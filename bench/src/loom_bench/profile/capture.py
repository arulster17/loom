"""Profile windows: switch the engine's profiler on during a short load job, fetch the
trace, break it down.

Two halves:

- `run_capture` runs inside `bench job run` on the GPU host, next to the load
  generator: at `start_after_s` it POSTs `/start_profile`, at `stop_after_s`
  `/stop_profile`, which flushes the trace. The start body is the engine's own
  (`start_body`): vLLM's per-session bounds `max_iterations` / `delay_iterations` /
  `profile_prefix`, or TensorRT-LLM's `output_dir` / `num_steps` / `start_step` (it
  records `num_steps` steps, writes the trace and stops by itself; /stop_profile then
  returns once that write is done). Times are reported against the job's t0, the
  clock the request records use.
- `run_profiles` runs in the runner after a cell's measured load points: one window
  per (workload, load value) in the experiment's `profile:` section. It builds the job,
  has the provider run it and collect the trace (`collect_profile`: on RunPod a root
  script archives the engine's trace directory and uploads it through a presigned S3
  URL; on the mock, the simulated engine's directory), stores it under
  `results/<experiment>/profiles/<run id>/` with the breakdown, and records a
  `profile` event. Nothing here touches `bench_runs`: a window's latencies are inflated
  by the profiler, so it never counts as a load run, for goodput or anything else.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import TYPE_CHECKING, Any

import httpx
import numpy as np

from loom_bench.profile.spec import ProfileApi, ProfileCapture, ProfileSpec

if TYPE_CHECKING:
    from loom_bench.experiment import Cell, WorkloadEntry
    from loom_bench.providers.base import Endpoint, Host
    from loom_bench.records import RequestRecord
    from loom_bench.workloads import WorkloadProfile

log = logging.getLogger(__name__)


def engine_root(base_url: str) -> str:
    """The server root of an OpenAI base URL (`http://h:p/v1` -> `http://h:p`)."""
    root = base_url.rstrip("/")
    return root[: -len("/v1")] if root.endswith("/v1") else root


async def _post(http: httpx.AsyncClient, path: str, body: dict[str, Any] | None) -> tuple[int, str]:
    resp = await http.post(path, json=body) if body is not None else await http.post(path)
    return resp.status_code, resp.text[:300]


def start_body(capture: ProfileCapture) -> dict[str, Any]:
    """The /start_profile body for the capture's engine API."""
    if capture.api == "trtllm":
        # TensorRT-LLM 1.4.0rc2 StartProfileRequest (extra fields are a 422). It has no
        # prefix: the trace is trtllm-trace-<time>-<id>-rank-<rank>.json in output_dir.
        return {
            "output_dir": capture.output_dir,
            "num_steps": capture.max_iterations,
            "start_step": capture.delay_iterations,
        }
    return {
        "profile_prefix": capture.prefix,
        "max_iterations": capture.max_iterations,
        "delay_iterations": capture.delay_iterations,
    }


async def run_capture(base_url: str, capture: ProfileCapture, t0: float) -> dict[str, Any]:
    """Start and stop the engine's profiler on schedule; never raises (the load job's
    result must come back either way). Returns what happened, for the job's meta."""
    out: dict[str, Any] = {"prefix": capture.prefix, "api": capture.api}
    body = start_body(capture)
    timeout = httpx.Timeout(30.0, read=capture.stop_timeout_s)
    async with httpx.AsyncClient(
        base_url=engine_root(base_url), timeout=timeout, trust_env=False
    ) as http:
        try:
            await asyncio.sleep(max(0.0, t0 + capture.start_after_s - time.perf_counter()))
            out["start_s"] = time.perf_counter() - t0
            status, text = await _post(http, "/start_profile", body)
            if capture.api == "vllm" and 400 <= status < 500 and status != 409:
                # A vLLM without per-session options: start with its defaults. Never
                # TensorRT-LLM: its default output_dir is not the collected directory.
                out["start_options_refused"] = f"{status}: {text}"
                status, text = await _post(http, "/start_profile", None)
            out["start_status"] = status
            if status >= 300:
                out["error"] = f"/start_profile answered {status}: {text}"
                return out
            await asyncio.sleep(max(0.0, t0 + capture.stop_after_s - time.perf_counter()))
            out["stop_s"] = time.perf_counter() - t0
            status, text = await _post(http, "/stop_profile", None)
            out["stop_status"] = status
            out["stopped_s"] = time.perf_counter() - t0
            if status >= 300:
                out["error"] = f"/stop_profile answered {status}: {text}"
        except (httpx.HTTPError, OSError) as e:
            out["error"] = f"{type(e).__name__}: {e}"
    return out


def window_latency(records: list[RequestRecord], start_s: float, stop_s: float) -> dict[str, Any]:
    """Inter-token gaps while the profiler was on against the same load just before it
    (the settle window's second half): what profiling cost the steps it recorded."""
    inside, before = [], []
    settle_from = start_s / 2
    for r in records:
        if r.first_token_at_s is None or not r.itl_s:
            continue
        times = r.first_token_at_s + np.cumsum(r.itl_s)
        gaps = np.asarray(r.itl_s)
        inside.append(gaps[(times >= start_s) & (times < stop_s)])
        before.append(gaps[(times >= settle_from) & (times < start_s)])

    def stats(parts: list[np.ndarray]) -> dict[str, float | int | None]:
        g = np.concatenate(parts) if parts else np.array([])
        if g.size == 0:
            return {"tokens": 0, "itl_p50_ms": None, "itl_mean_ms": None}
        return {
            "tokens": int(g.size),
            "itl_p50_ms": float(np.median(g) * 1000),
            "itl_mean_ms": float(g.mean() * 1000),
        }

    a, b = stats(inside), stats(before)
    out: dict[str, Any] = {"profiled": a, "before": b}
    if a["itl_p50_ms"] and b["itl_p50_ms"]:
        out["profiler_slowdown"] = float(a["itl_p50_ms"]) / float(b["itl_p50_ms"])  # type: ignore[arg-type]
    return out


async def collect_profile(provider: Any, host: Host, run_id: str, spec: ProfileSpec) -> bytes:
    collect = getattr(provider, "collect_profile", None)
    if collect is None:
        raise RuntimeError(
            f"provider {getattr(provider, 'name', provider)!r} cannot collect traces"
        )
    data = await collect(host, run_id, spec.trace_dir, spec.max_trace_mb * 1024 * 1024)
    return bytes(data)


def capture_for(spec: ProfileSpec, run_id: str, api: ProfileApi = "vllm") -> ProfileCapture:
    return ProfileCapture(
        start_after_s=spec.settle_s,
        stop_after_s=spec.settle_s + spec.capture_s,
        prefix=f"loom-{run_id}",
        max_iterations=spec.max_iterations,
        delay_iterations=spec.delay_iterations,
        stop_timeout_s=spec.stop_timeout_s,
        api=api,
        output_dir=spec.trace_dir if api == "trtllm" else None,
    )


def profile_api(cell: Cell) -> ProfileApi:
    """The profiler API a cell's engine speaks: its launch engine's, or on the mock the
    one its simulated engine imitates (`MockConfig.profile_api`)."""
    engine = cell.launch.engine
    if engine == "mock":
        return cell.mock.profile_api if cell.mock is not None else "vllm"
    if engine == "trtllm":
        return "trtllm"
    return "vllm"


def window_load(entry: WorkloadEntry, spec: ProfileSpec) -> Any:
    """The workload's load spec cut down to one profile window (no warmup accounting:
    nothing in it is measured)."""
    from loom_bench.records import LoadMode

    update: dict[str, Any] = {
        "duration_s": spec.job_s,
        "num_requests": None,
        "drain_timeout_s": spec.drain_timeout_s,
    }
    if entry.load.mode is LoadMode.OPEN_LOOP:
        update["warmup_s"] = 0.0
    else:
        update["warmup_requests"] = None
    return entry.load.model_copy(update=update)


def windows(
    spec: ProfileSpec, entries: list[tuple[WorkloadEntry, WorkloadProfile]]
) -> list[tuple[WorkloadEntry, WorkloadProfile, float]]:
    by_name = {e.name: (e, p) for e, p in entries}
    out = []
    for w in spec.windows:
        entry, profile = by_name[w.workload]
        for value in w.load_values or entry.load.values or []:
            out.append((entry, profile, float(value)))
    return out


async def run_profiles(ex: Any, host: Host, cell: Cell, endpoint: Endpoint) -> None:
    """Every profile window of `ex.exp.profile` on this cell (`ex`: the runner's
    executor). A failed window is recorded and skipped; budget and host loss propagate."""
    from loom_bench.budget import BudgetExceeded
    from loom_bench.experiment import derive_seed
    from loom_bench.providers.base import HostLost

    spec: ProfileSpec = ex.exp.profile
    for entry, profile, value in windows(spec, list(ex.workloads)):
        load = window_load(entry, spec)
        label = f"{cell.key} / {entry.name} @ {value:g} (profile)"
        ex.guard.check_next(spec.window_s(ex.est.run_overhead_s(profile)), what=f"profile {label}")
        seed = derive_seed(ex.exp.seed, entry.name, value, "profile")
        job = ex._job(cell, endpoint, load, profile, value, seed)
        capture = capture_for(spec, job.run_id, profile_api(cell))
        job = job.model_copy(update={"profile": capture, "sample_gpu": False})
        out_dir = ex.run_dir / "profiles" / job.run_id
        try:
            result = await ex.guard.guarded(ex.provider.run_job(host, job))
            meta = dict(result.meta.get("profile") or {})
            if meta.get("error"):
                raise RuntimeError(meta["error"])
            blob = await ex.guard.guarded(collect_profile(ex.provider, host, job.run_id, spec))
        except (BudgetExceeded, HostLost):
            raise
        except Exception as e:
            log.exception("profile window %s failed", label)
            ex.event(
                "profile_failed",
                cell=cell.key,
                workload=entry.name,
                load=value,
                run_id=job.run_id,
                error=f"{type(e).__name__}: {e}"[-2000:],
            )
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        trace_path = out_dir / "trace.tar.gz"
        trace_path.write_bytes(blob)
        latency = window_latency(
            result.request_records(), meta.get("start_s", 0.0), meta.get("stop_s", 0.0)
        )
        doc: dict[str, Any] = {
            "run_id": job.run_id,
            "cell": cell.key,
            "config_hash": cell.config_hash,
            "workload": entry.name,
            "load_mode": str(load.mode),
            "load_value": value,
            "capture": meta,
            "spec": spec.model_dump(mode="json"),
            "client_latency": latency,
            "trace_bytes": len(blob),
            # a closed-loop window that ran out of prepared requests idled the engine
            "load_exhausted": result.meta.get("exhausted"),
            "note": "profile window: not a load run; excluded from goodput and every report",
        }
        try:
            from loom_bench.profile.trace import analyze_path

            breakdown = analyze_path(trace_path, steps=spec.max_iterations)
            doc["breakdown"] = breakdown.as_dict()
        except (ValueError, OSError) as e:
            doc["breakdown_error"] = str(e)
        (out_dir / "window.json").write_text(json.dumps(doc, indent=2, default=str))
        rank0 = (doc.get("breakdown") or {}).get("ranks", [{}])[0]
        ex.event(
            "profile",
            cell=cell.key,
            workload=entry.name,
            load=value,
            run_id=job.run_id,
            trace=str(trace_path),
            trace_bytes=len(blob),
            per_step_ms=rank0.get("per_step_ms"),
            steps=rank0.get("steps"),
            profiler_slowdown=latency.get("profiler_slowdown"),
            error=doc.get("breakdown_error"),
        )
