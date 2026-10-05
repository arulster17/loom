"""Execute one LoadJob: hand it to its load generator, scrape the server meanwhile.

Runs in-process for the mock and local providers, and on the GPU host for
cloud providers (`bench job run --in job.json --out result.json`).
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from datetime import UTC, datetime

import httpx

from loom_bench.jobs import LoadJob, LoadJobResult
from loom_bench.loadgen.base import get_load_generator, prepare_load
from loom_bench.metrics.gpu import NVIDIA_SMI_ARGS


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
    ctx = prepare_load(job)
    generator = get_load_generator(job.loadgen)

    meta: dict[str, object] = {
        "requests_prepared": len(ctx.requests),
        "tokenizer": ctx.tokenizer.name,
    }
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
        result = await generator.run(ctx)
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
        timeline=result.timeline,
        client_saturated_count=result.client_saturated_count,
        records=[r.to_row() for r in result.records],
        scrapes=scrapes,
        nvidia_smi_csv=smi,
        started_at=started_at,
        finished_at=finished_at,
        meta={**result.meta, **meta, "loadgen": generator.name},
    )
