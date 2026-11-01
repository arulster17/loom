"""A mock serving pod: the bench mock engine on loopback behind the pod proxy.

The M1 stand-in for a RunPod serving pod (contracts §9): the engine
(`python -m loom_bench.mock`) listens on 127.0.0.1:8000 only, and the pod proxy on
0.0.0.0:8443 is the only way in, with the pod's certificate and token from the dev
certificate directory (`loom-devcerts`). If the engine exits, the pod exits.

    loom-mockpod --pod-id mock-pod-1 --model qwen3-8b
    loom-mockpod --healthcheck          # container healthcheck: engine up, proxy listening

The engine gets a timing profile per model (below) so streams look like the real
replica's at low load; `--engine-set FIELD=VALUE` overrides any `MockConfig` field.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import logging
import os
import signal
import socket
import sys
import tempfile
from pathlib import Path
from typing import Any

import httpx
import yaml

from loom_podproxy.proxy import PodProxy, ProxyConfig, parse_networks, server_ssl_context

log = logging.getLogger("loom_mockpod")

ENGINE_HOST = "127.0.0.1"
ENGINE_PORT = 8000
PROXY_PORT = 8443

# Approximate low-load shapes of the measured replicas (reports/8b-config-sweep-v1,
# reports/70b-h100-v2): inter-token latency ~ step_base_ms + decode_ms_per_seq x running
# sequences; time to first token ~ step_base_ms + prefill_ms_per_token x prompt tokens.
# Plausible, not calibrated: the mock is for plumbing and load shape, never for numbers.
ENGINE_PROFILES: dict[str, dict[str, Any]] = {
    "qwen3-8b": {"step_base_ms": 20.0, "decode_ms_per_seq": 0.25, "prefill_ms_per_token": 0.1},
    "llama-3.3-70b-instruct-fp8": {
        "step_base_ms": 25.0,
        "decode_ms_per_seq": 0.3,
        "prefill_ms_per_token": 0.15,
    },
}


def engine_config(models: list[str], overrides: list[tuple[str, Any]]) -> dict[str, Any]:
    from loom_bench.registry import load_registry

    registry = load_registry()
    known = {m.id: m for m in registry.models}
    config: dict[str, Any] = {"models": models, "mean_output_tokens": 200}
    config.update(ENGINE_PROFILES.get(models[0], {}))
    if models[0] in known:
        config["max_model_len"] = known[models[0]].max_context
    config.update(overrides)
    return config


def _override(text: str) -> tuple[str, Any]:
    field, sep, value = text.partition("=")
    if not sep or not field:
        raise argparse.ArgumentTypeError(f"expected FIELD=VALUE, got {text!r}")
    return field.strip(), yaml.safe_load(value)


async def _wait_engine(
    proc: asyncio.subprocess.Process, engine_port: int, timeout_s: float = 60
) -> None:
    deadline = asyncio.get_running_loop().time() + timeout_s
    async with httpx.AsyncClient(trust_env=False, timeout=2) as http:
        while True:
            if proc.returncode is not None:
                raise SystemExit(f"engine exited with {proc.returncode} during start")
            with contextlib.suppress(httpx.HTTPError):
                if (await http.get(f"http://{ENGINE_HOST}:{engine_port}/health")).is_success:
                    return
            if asyncio.get_running_loop().time() > deadline:
                raise SystemExit("engine did not become healthy")
            await asyncio.sleep(0.2)


async def run(args: argparse.Namespace) -> int:
    certs: Path = args.certs_dir
    config = engine_config(args.model, args.engine_set)
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        yaml.safe_dump(config, f)
        config_path = f.name
    proc = await asyncio.create_subprocess_exec(
        sys.executable,
        "-m",
        "loom_bench.mock",
        "--host",
        ENGINE_HOST,
        "--port",
        str(args.engine_port),
        "--config",
        config_path,
        "--log-level",
        "warning",
    )
    proxy: PodProxy | None = None
    try:
        await _wait_engine(proc, args.engine_port)
        proxy = PodProxy(
            ProxyConfig(
                token=(certs / f"{args.pod_id}.token").read_text().strip(),
                upstream=f"http://{ENGINE_HOST}:{args.engine_port}",
                allowed_networks=parse_networks(args.allow_ip),
            ),
            server_ssl_context(
                certs / f"{args.pod_id}.crt", certs / f"{args.pod_id}.key", certs / "ca.crt"
            ),
        )
        await proxy.start("0.0.0.0", args.port)
        log.info(
            "%s serving %s: proxy :%d -> engine %s:%d",
            args.pod_id,
            ", ".join(args.model),
            args.port,
            ENGINE_HOST,
            args.engine_port,
        )
        stop = asyncio.Event()
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            with contextlib.suppress(NotImplementedError):
                loop.add_signal_handler(sig, stop.set)
        engine_done = asyncio.create_task(proc.wait())
        stopped = asyncio.create_task(stop.wait())
        await asyncio.wait({engine_done, stopped}, return_when=asyncio.FIRST_COMPLETED)
        stopped.cancel()
        if engine_done.done():
            log.error("engine exited with %s; stopping the pod", proc.returncode)
            return 1
        return 0
    finally:
        if proxy is not None:
            await proxy.close()
        if proc.returncode is None:
            proc.terminate()
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(proc.wait(), 10)
            if proc.returncode is None:
                proc.kill()
        os.unlink(config_path)


def healthcheck(port: int, engine_port: int = ENGINE_PORT) -> int:
    try:
        url = f"http://{ENGINE_HOST}:{engine_port}/health"
        r = httpx.get(url, timeout=2, trust_env=False)
        if not r.is_success:
            return 1
        with socket.create_connection(("127.0.0.1", port), timeout=2):
            pass
    except (httpx.HTTPError, OSError):
        return 1
    return 0


def main(argv: list[str] | None = None) -> None:
    env = os.environ.get
    p = argparse.ArgumentParser(prog="loom-mockpod", description=__doc__.splitlines()[0])
    p.add_argument("--pod-id", default=env("LOOM_MOCKPOD_ID", "mock-pod-1"))
    p.add_argument(
        "--model",
        action="append",
        help="served model id (repeatable; default $LOOM_MOCKPOD_MODELS, comma-separated)",
    )
    p.add_argument(
        "--engine-set",
        action="append",
        type=_override,
        default=[],
        metavar="FIELD=VALUE",
        help="override a MockConfig field of the engine, e.g. --engine-set time_scale=0.1",
    )
    p.add_argument("--certs-dir", type=Path, default=Path(env("LOOM_DEV_CERTS", ".dev/certs")))
    p.add_argument("--port", type=int, default=int(env("LOOM_MOCKPOD_PORT", str(PROXY_PORT))))
    p.add_argument(
        "--allow-ip",
        action="append",
        default=[v for v in [env("LOOM_PODPROXY_ALLOW_IPS")] if v],
        metavar="CIDR",
    )
    p.add_argument("--engine-port", type=int, default=ENGINE_PORT, help="engine port on loopback")
    p.add_argument("--healthcheck", action="store_true")
    p.add_argument("--log-level", default=env("LOOM_MOCKPOD_LOG_LEVEL", "info"))
    args = p.parse_args(argv)
    if args.healthcheck:
        raise SystemExit(healthcheck(args.port, args.engine_port))
    if not args.model:
        args.model = [m for m in env("LOOM_MOCKPOD_MODELS", "qwen3-8b").split(",") if m]
    logging.basicConfig(
        level=args.log_level.upper(), format="%(asctime)s %(levelname)s %(name)s %(message)s"
    )
    raise SystemExit(asyncio.run(run(args)))


if __name__ == "__main__":
    main()
