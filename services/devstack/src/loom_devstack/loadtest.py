"""Gateway load test: drive the public API with real API keys, report TTFT and tokens/s.

Uses the Benchmark Lab's streaming client (`loom_bench.client.openai_stream.send`), its
workload generator and its run summary, so the numbers mean what they mean in the Lab:
TTFT from send to first content chunk, TPOT, inter-token latency, per-request output
tokens/s, aggregate output tokens/s over the run. Every request streams with
`include_usage`, so token counts are the gateway's (the engine's) usage, not ours.

    loom-loadtest --base-url http://localhost:8080/v1 --api-key loom_sk_... \\
        --model qwen3-8b --concurrency 8 --requests 200

Closed loop: `--concurrency` workers each send their next request when the previous one
ends; worker i uses key i mod the number of keys. Mind the gateway's limits (contracts
§6): the per-key default is 8 in flight and 60 requests/min, the per-org default 16 and
120/min, so a run above those needs several keys and orgs (or raised limits), else it
measures 429s. Status counts are reported per HTTP status.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from loom_bench.client.openai_stream import PreparedRequest, new_record, send
from loom_bench.metrics.summary import RunSummary, summarize_run
from loom_bench.records import RequestRecord
from loom_bench.tokenize import SimpleTokenizer
from loom_bench.workloads import build_requests, load_profile, parse_profile

# OpenAI request fields the public API accepts; engine-only knobs a Lab profile may set
# (ignore_eos, min_tokens) are dropped unless --engine-params.
OPENAI_FIELDS = frozenset(
    {"messages", "prompt", "max_tokens", "temperature", "top_p", "stop", "seed", "n"}
)


@dataclass(frozen=True)
class LoadTestConfig:
    base_url: str
    api_keys: tuple[str, ...]
    model: str
    workload: str = "chat-short"
    concurrency: int = 4
    requests: int = 100
    warmup: int = 4
    max_tokens: int | None = None
    request_timeout_s: float = 120.0
    engine_params: bool = False
    seed: int = 0


# A short chat shape that stays inside the default limits at low concurrency.
BUILTIN_WORKLOADS: dict[str, dict[str, Any]] = {
    "chat-short": {
        "name": "chat-short",
        "description": "Short chat turns: ~256-token prompts, up to ~128-token answers.",
        "kind": "synthetic",
        "content": "synthetic",
        "endpoint": "chat",
        "input_len": 256,
        "output_len": 128,
        "range_ratio": 0.5,
        "ignore_eos": True,
        "seed": 0,
    },
}


def prepare(config: LoadTestConfig) -> list[PreparedRequest]:
    if config.workload in BUILTIN_WORKLOADS:
        profile = parse_profile(BUILTIN_WORKLOADS[config.workload])
    else:
        profile = load_profile(config.workload)
    n = config.warmup + config.requests
    requests = build_requests(profile, SimpleTokenizer(), n, rng_seed=config.seed)
    for req in requests:
        payload = dict(req.payload)
        if config.max_tokens is not None:
            payload["max_tokens"] = config.max_tokens
            req.max_tokens = config.max_tokens
        if not config.engine_params:
            payload = {k: v for k, v in payload.items() if k in OPENAI_FIELDS}
        req.payload = payload
    return requests


async def run_load(
    config: LoadTestConfig, *, transport: httpx.AsyncBaseTransport | None = None
) -> list[RequestRecord]:
    if not config.api_keys:
        raise ValueError("at least one API key is needed")
    requests = prepare(config)
    queue = iter(enumerate(requests))
    records: list[RequestRecord] = []

    async def worker(worker_id: int, http: httpx.AsyncClient, t0: float) -> None:
        key = config.api_keys[worker_id % len(config.api_keys)]
        for i, req in queue:
            rec = new_record(req, t0)
            rec.warmup = i < config.warmup
            rec.meta["key_index"] = worker_id % len(config.api_keys)
            records.append(rec)
            await send(
                http,
                config.base_url,
                req,
                t0,
                timeout_s=config.request_timeout_s,
                api_key=key,
                model=config.model,
                record=rec,
            )

    limits = httpx.Limits(
        max_connections=config.concurrency, max_keepalive_connections=config.concurrency
    )
    async with httpx.AsyncClient(
        transport=transport,
        limits=limits,
        timeout=httpx.Timeout(config.request_timeout_s),
        trust_env=False,
    ) as http:
        t0 = time.perf_counter()
        async with asyncio.TaskGroup() as tg:
            for w in range(config.concurrency):
                tg.create_task(worker(w, http, t0))
    return records


@dataclass(frozen=True)
class Report:
    summary: RunSummary
    http_status: dict[str, int]
    errors: list[str]

    def to_json(self) -> dict[str, Any]:
        return {
            "summary": self.summary.model_dump(mode="json"),
            "http_status": self.http_status,
            "errors": self.errors,
        }


def report(records: Sequence[RequestRecord]) -> Report:
    measured = [r for r in records if not r.warmup]
    summary = summarize_run(measured, window_s=None, gpus=1)
    statuses = Counter(str(r.http_status) if r.http_status else "none" for r in measured)
    errors = sorted({r.error for r in measured if r.error})[:10]
    return Report(summary, dict(sorted(statuses.items())), errors)


def _fmt(value: float | None, digits: int = 1) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def render(rep: Report, config: LoadTestConfig) -> str:
    s = rep.summary
    lines = [
        f"model {config.model}  workload {config.workload}  concurrency {config.concurrency}"
        f"  keys {len(config.api_keys)}  measured requests {s.n_total}"
        f" (+{config.warmup} warmup)  window {s.window_s:.1f} s",
        f"status: {s.counts}  http: {rep.http_status}  error rate {_fmt(s.error_rate, 4)}",
        "",
        f"{'metric':<28}{'p50':>10}{'p90':>10}{'p95':>10}{'p99':>10}{'mean':>10}",
    ]
    for name, dist in (
        ("TTFT ms", s.ttft_ms),
        ("TPOT ms", s.tpot_ms),
        ("ITL ms", s.itl_ms),
        ("E2E ms", s.e2e_ms),
        ("output tok/s per request", s.per_request_output_tok_s),
    ):
        lines.append(
            f"{name:<28}{_fmt(dist.p50):>10}{_fmt(dist.p90):>10}{_fmt(dist.p95):>10}"
            f"{_fmt(dist.p99):>10}{_fmt(dist.mean):>10}"
        )
    t = s.throughput
    lines += [
        "",
        f"throughput: {t.request_rate:.2f} req/s, {t.output_tok_s:.1f} output tok/s, "
        f"{t.input_tok_s:.1f} input tok/s",
    ]
    if s.requests_missing_usage:
        lines.append(f"WARNING: {s.requests_missing_usage} successful requests had no usage")
    for err in rep.errors:
        lines.append(f"error: {err}")
    return "\n".join(lines)


def _keys(values: Sequence[str], key_file: Path | None) -> tuple[str, ...]:
    keys = [k.strip() for v in values for k in v.split(",") if k.strip()]
    if key_file is not None:
        keys += [ln.strip() for ln in key_file.read_text().splitlines() if ln.strip()]
    return tuple(keys)


def main(argv: list[str] | None = None) -> None:
    env = os.environ.get
    p = argparse.ArgumentParser(prog="loom-loadtest", description=__doc__.splitlines()[0])
    p.add_argument("--base-url", default=env("LOOM_LOADTEST_BASE_URL", "http://localhost:8080/v1"))
    p.add_argument(
        "--api-key",
        action="append",
        default=[v for v in [env("LOOM_LOADTEST_API_KEYS")] if v],
        help="API key (repeatable or comma-separated; default $LOOM_LOADTEST_API_KEYS)",
    )
    p.add_argument("--api-key-file", type=Path, help="file with one API key per line")
    p.add_argument("--model", default="qwen3-8b")
    p.add_argument(
        "--workload",
        default="chat-short",
        help="chat-short (built in), a Lab profile name (fixed-128-128, ...) or a YAML path",
    )
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--requests", type=int, default=100, help="measured requests")
    p.add_argument("--warmup", type=int, default=4)
    p.add_argument("--max-tokens", type=int)
    p.add_argument("--timeout-s", type=float, default=120.0)
    p.add_argument(
        "--engine-params",
        action="store_true",
        help="keep engine-only fields of the profile (ignore_eos, min_tokens)",
    )
    p.add_argument("--json", type=Path, help="also write the report as JSON here")
    p.add_argument(
        "--max-error-rate",
        type=float,
        help="exit 1 when the error rate (errors + timeouts over attempts) is above this",
    )
    args = p.parse_args(argv)
    keys = _keys(args.api_key, args.api_key_file)
    if not keys:
        p.error("give --api-key, --api-key-file or $LOOM_LOADTEST_API_KEYS")
    config = LoadTestConfig(
        base_url=args.base_url,
        api_keys=keys,
        model=args.model,
        workload=args.workload,
        concurrency=args.concurrency,
        requests=args.requests,
        warmup=args.warmup,
        max_tokens=args.max_tokens,
        request_timeout_s=args.timeout_s,
        engine_params=args.engine_params,
    )
    records = asyncio.run(run_load(config))
    rep = report(records)
    print(render(rep, config))
    if args.json:
        args.json.write_text(json.dumps(rep.to_json(), indent=2))
    rate = rep.summary.error_rate
    if args.max_error_rate is not None and (rate is None or rate > args.max_error_rate):
        sys.exit(1)


if __name__ == "__main__":
    main()
