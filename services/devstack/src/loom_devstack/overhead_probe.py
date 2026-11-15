"""Gateway overhead on the compose network (M1 acceptance item d).

Runs inside a container on the compose network (the acceptance script pipes this file to
`python -` in the poolsim container, which has httpx and the dev certificates), so both
paths start from the same place:

- direct: `https://<pod>:8443/v1/chat/completions` with the gateway's client certificate
  and the pod token, i.e. exactly the gateway's own upstream request (mTLS, TLS 1.3);
- gateway: `http://gateway:8080/v1/chat/completions` with an API key.

Requests alternate, over kept-alive connections; the added overhead is the difference of
the medians. Prints one JSON object. Only the standard library and httpx: no Loom imports,
so it runs on any image that has httpx.

    python overhead_probe.py --api-key loom_sk_... --pairs 300
"""

from __future__ import annotations

import argparse
import json
import ssl
import statistics
import time
from pathlib import Path

import httpx


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="overhead_probe")
    p.add_argument("--gateway", default="http://gateway:8080")
    p.add_argument("--pod", default="mock-pod-1")
    p.add_argument("--model", default="qwen3-8b")
    p.add_argument("--certs", type=Path, default=Path("/certs"))
    p.add_argument("--api-key", required=True)
    p.add_argument("--pairs", type=int, default=300)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--max-tokens", type=int, default=1)
    args = p.parse_args(argv)

    certs: Path = args.certs
    ctx = ssl.create_default_context(cafile=str(certs / "ca.crt"))
    ctx.load_cert_chain(str(certs / "gateway.crt"), str(certs / "gateway.key"))
    token = (certs / f"{args.pod}.token").read_text().strip()
    body = {
        "model": args.model,
        "messages": [{"role": "user", "content": "Hello there."}],
        "max_tokens": args.max_tokens,
    }
    direct_url = f"https://{args.pod}:8443/v1/chat/completions"
    gw_url = f"{args.gateway.rstrip('/')}/v1/chat/completions"

    with (
        httpx.Client(verify=ctx, timeout=30, trust_env=False) as direct,
        httpx.Client(timeout=30, trust_env=False) as gw,
    ):

        def via_pod() -> float:
            t0 = time.perf_counter()
            r = direct.post(direct_url, json=body, headers={"X-Loom-Pod-Token": token})
            dt = time.perf_counter() - t0
            if r.status_code != 200:
                raise SystemExit(f"direct: {r.status_code} {r.text[:200]}")
            return dt

        def via_gateway() -> float:
            t0 = time.perf_counter()
            r = gw.post(gw_url, json=body, headers={"Authorization": f"Bearer {args.api_key}"})
            dt = time.perf_counter() - t0
            if r.status_code != 200:
                raise SystemExit(f"gateway: {r.status_code} {r.text[:200]}")
            return dt

        for _ in range(args.warmup):
            via_pod()
            via_gateway()
        d: list[float] = []
        g: list[float] = []
        for i in range(args.pairs):
            # Alternate which goes first so drift in the engine's step clock is shared.
            if i % 2:
                d.append(via_pod())
                g.append(via_gateway())
            else:
                g.append(via_gateway())
                d.append(via_pod())

    ms = 1000.0

    def q(xs: list[float], k: int) -> float:
        return statistics.quantiles(xs, n=100)[k - 1] * ms

    print(
        json.dumps(
            {
                "pairs": args.pairs,
                "direct_p50_ms": statistics.median(d) * ms,
                "gateway_p50_ms": statistics.median(g) * ms,
                "overhead_p50_ms": (statistics.median(g) - statistics.median(d)) * ms,
                "direct_p90_ms": q(d, 90),
                "gateway_p90_ms": q(g, 90),
                "overhead_p90_ms": q(g, 90) - q(d, 90),
                "paired_diff_p50_ms": statistics.median(
                    gi - di for gi, di in zip(g, d, strict=True)
                )
                * ms,
            }
        )
    )


if __name__ == "__main__":
    main()
