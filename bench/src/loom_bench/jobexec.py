"""Execute one LoadJob: prepare requests, drive load, scrape the server meanwhile.

Runs in-process for the mock and local providers, and on the GPU host for
cloud providers (`bench job run --in job.json --out result.json`).
"""

from __future__ import annotations

import asyncio
import contextlib
import math
import os
import time
from datetime import UTC, datetime

import httpx

from loom_bench.jobs import LoadJob, LoadJobResult, TokenizerSpec
from loom_bench.loadgen.arrivals import parse_arrivals
from loom_bench.loadgen.base import ClosedLoopPlan, LoadPlan, OpenLoopPlan, get_load_generator
from loom_bench.metrics.gpu import NVIDIA_SMI_ARGS
from loom_bench.records import LoadMode
from loom_bench.tokenize import HFTokenizer, SimpleTokenizer, Tokenizer
from loom_bench.workloads import build_requests
from loom_bench.workloads.profiles import parse_profile

# Duration-bound closed-loop runs need requests prepared up front: this many per
# worker per second of run is 40x headroom for a 1k-token answer at 100 tok/s.
# A run that still runs out reports `exhausted` in its meta.
CLOSED_LOOP_REQUESTS_PER_SLOT_S = 4


def make_tokenizer(spec: TokenizerSpec) -> Tokenizer:
    if spec.kind == "simple":
        return SimpleTokenizer()
    if spec.repo is None or spec.revision is None:
        raise ValueError("hf tokenizer needs repo and a pinned revision")
    return HFTokenizer(spec.repo, spec.revision, token=os.environ.get("HF_TOKEN"))


def _plan(job: LoadJob) -> tuple[LoadPlan, int]:
    """Load plan and how many requests it needs."""
    if job.mode is LoadMode.OPEN_LOOP:
        if job.arrival is None or job.duration_s is None:
            raise ValueError("open-loop jobs need arrival and duration_s")
        offsets = parse_arrivals(job.arrival).schedule(job.duration_s, job.seed)
        plan = OpenLoopPlan(
            arrivals=offsets,
            duration_s=job.duration_s,
            warmup_s=job.warmup_s,
            max_inflight=job.max_inflight,
            drain_timeout_s=job.drain_timeout_s,
            load_value=job.load_value,
        )
        return plan, len(offsets)
    concurrency = int(job.load_value)
    if job.num_requests is not None:
        n = job.warmup_requests + job.num_requests
    elif job.duration_s is not None:
        n = job.warmup_requests + math.ceil(
            concurrency * job.duration_s * CLOSED_LOOP_REQUESTS_PER_SLOT_S
        )
    else:
        raise ValueError("closed-loop jobs need num_requests or duration_s")
    plan = ClosedLoopPlan(
        concurrency=concurrency,
        num_requests=job.num_requests,
        duration_s=job.duration_s if job.num_requests is None else None,
        warmup_requests=job.warmup_requests,
    )
    return plan, n


async def _scrape(
    url: str, interval_s: float, t0: float, stop: asyncio.Event
) -> tuple[list[tuple[float, str]], int]:
    """Scrape `url` every `interval_s` until `stop`, then once more."""
    scrapes: list[tuple[float, str]] = []
    errors = 0
    async with httpx.AsyncClient(timeout=max(interval_s, 5.0), trust_env=False) as http:
        while True:
            done = stop.is_set()
            try:
                resp = await http.get(url)
                resp.raise_for_status()
                scrapes.append((time.perf_counter() - t0, resp.text))
            except httpx.HTTPError:
                errors += 1
            if done:
                return scrapes, errors
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(stop.wait(), interval_s)


async def _start_gpu_sampler(interval_s: float) -> asyncio.subprocess.Process | None:
    try:
        return await asyncio.create_subprocess_exec(
            *NVIDIA_SMI_ARGS,
            f"--loop-ms={max(int(interval_s * 1000), 100)}",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
    except FileNotFoundError:
        return None


async def _stop_gpu_sampler(proc: asyncio.subprocess.Process) -> str:
    if proc.returncode is None:
        proc.terminate()
    out, _ = await proc.communicate()
    return out.decode(errors="replace")


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


async def execute_load_job(job: LoadJob) -> LoadJobResult:
    tokenizer = make_tokenizer(job.tokenizer)
    profile = parse_profile(job.workload)
    plan, n = _plan(job)
    requests = build_requests(profile, tokenizer, n, rng_seed=job.seed)
    if job.extra_body:
        for req in requests:
            req.payload = {**req.payload, **job.extra_body}
    generator = get_load_generator(job.loadgen)
    if hasattr(generator, "bind"):  # wrapped engine tools need the workload, not just requests
        from loom_bench.loadgen.external import WorkloadContext

        generator = generator.bind(WorkloadContext.from_job(job))

    meta: dict[str, object] = {"requests_prepared": n, "tokenizer": tokenizer.name}
    stop = asyncio.Event()
    t0 = time.perf_counter()
    scraper = (
        asyncio.create_task(_scrape(job.metrics_url, job.scrape_interval_s, t0, stop))
        if job.metrics_url
        else None
    )
    gpu = await _start_gpu_sampler(job.scrape_interval_s) if job.sample_gpu else None
    if job.sample_gpu and gpu is None:
        meta["gpu_sampling"] = "nvidia-smi not found"

    started_at = _now_iso()
    try:
        result = await generator.run(
            job.base_url,
            requests,
            plan,
            request_timeout_s=job.request_timeout_s,
            model=job.served_model,
            keep_output=job.keep_output,
        )
    except BaseException:
        if scraper is not None:
            scraper.cancel()
        if gpu is not None and gpu.returncode is None:
            gpu.kill()
        raise
    finished_at = _now_iso()
    stop.set()
    scrapes: list[tuple[float, str]] = []
    if scraper is not None:
        scrapes, meta["scrape_errors"] = await scraper
    smi = await _stop_gpu_sampler(gpu) if gpu is not None else None

    return LoadJobResult(
        run_id=job.run_id,
        mode=result.mode,
        load_value=result.load_value,
        t_measure_start_s=result.t_measure_start_s,
        t_measure_end_s=result.t_measure_end_s,
        client_saturated_count=result.client_saturated_count,
        records=[r.to_row() for r in result.records],
        scrapes=scrapes,
        nvidia_smi_csv=smi,
        started_at=started_at,
        finished_at=finished_at,
        meta={**result.meta, **meta, "loadgen": generator.name},
    )
