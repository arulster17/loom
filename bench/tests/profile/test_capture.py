"""Profile windows end to end on the mock backend, the spec rules, and the capture client."""

import asyncio
import json
import os
import sqlite3
import tarfile
import time
from pathlib import Path

import httpx
import pytest
from pydantic import ValidationError

from loom_bench.budget import load_budget
from loom_bench.experiment import Experiment
from loom_bench.mock.config import MockConfig
from loom_bench.mock.server import create_app
from loom_bench.prices import load_prices
from loom_bench.profile.capture import engine_root, run_capture, window_latency
from loom_bench.profile.mock_trace import SHARES
from loom_bench.profile.spec import ProfileCapture
from loom_bench.records import RequestRecord, RequestStatus
from loom_bench.registry import load_registry
from loom_bench.runner import RunnerContext, plan_experiment, run_experiment
from loom_bench.store.db import upgrade

pytestmark = pytest.mark.timeout(240)


def mock_profile_doc(**over: object) -> dict:
    doc: dict = {
        "name": "unit-profile",
        "description": "mock experiment with profile windows",
        "model": "qwen3-8b",
        "provider": {"kind": "mock", "hourly_price": "$1", "time_scale": 0.3},
        "variants": [{"name": "a"}, {"name": "b", "mock": {"step_base_ms": 9.0}}],
        "workloads": [
            {
                "profile": "fixed-128-128",
                "overrides": {"input_len": 32, "output_len": 256},
                "load": {
                    "mode": "closed_loop",
                    "values": [2, 4],
                    "num_requests": 6,
                    "warmup_requests": 1,
                    "scrape_interval_s": 0.05,
                },
            }
        ],
        "repetitions": 2,
        "slo": {"ttft_ms": {"p95": 1000}, "max_error_rate": 0.01},
        "profile": {
            "windows": [{"workload": "fixed-128-128", "load_values": [4]}],
            "settle_s": 0.3,
            "capture_s": 0.4,
            "tail_s": 0.2,
            "max_iterations": 12,
            "drain_timeout_s": 2,
        },
        "budget": {"max_spend": "$1", "ttl_minutes": 10, "accrual_interval_s": 0.2},
    }
    doc.update(over)
    return doc


@pytest.fixture
def ctx(tmp_path: Path) -> RunnerContext:
    url = f"sqlite:///{tmp_path / 'loom.db'}"
    upgrade(url)
    return RunnerContext(
        db_url=url,
        out_dir=tmp_path / "results",
        registry=load_registry(),
        prices=load_prices(),
        budget=load_budget(),
    )


async def test_profile_windows_on_the_mock_never_become_load_runs(ctx: RunnerContext) -> None:
    exp = Experiment.model_validate(mock_profile_doc())
    outcome = await run_experiment(exp, ctx)
    assert outcome.exit_code == 0, outcome
    exp_dir = ctx.out_dir / str(outcome.experiment_id)
    windows = sorted((exp_dir / "profiles").iterdir())
    assert len(windows) == 2  # one window per cell
    for w in windows:
        doc = json.loads((w / "window.json").read_text())
        assert doc["load_value"] == 4 and "not a load run" in doc["note"]
        assert doc["capture"]["start_status"] == 200 and doc["capture"]["stop_status"] == 200
        with tarfile.open(w / "trace.tar.gz") as t:
            assert any(n.endswith(".pt.trace.json.gz") for n in t.getnames())
        assert "breakdown" in doc, doc.get("breakdown_error")
        (rank,) = doc["breakdown"]["ranks"]
        assert 0 < rank["steps"] <= 12
        per = rank["per_step_ms"]
        total = sum(per[c] for _, c, _ in SHARES)
        for _, cat, share in SHARES:  # the mock's known split comes back out
            assert per[cat] / total == pytest.approx(share / 0.9, rel=1e-4)
        assert doc["client_latency"]["profiled"]["tokens"] > 0
    # bench_runs: exactly the measured load runs (2 cells x 2 values x 2 reps).
    con = sqlite3.connect(ctx.db_url.removeprefix("sqlite:///"))
    assert con.execute("select count(*) from bench_runs").fetchone()[0] == 8
    con.close()
    run_ids = {p.name for p in (exp_dir / "runs").iterdir()}
    assert not run_ids & {w.name for w in windows}
    events = [json.loads(line) for line in (exp_dir / "events.jsonl").read_text().splitlines()]
    prof = [e for e in events if e["kind"] == "profile"]
    assert len(prof) == 2 and all(e["steps"] for e in prof)
    assert not [e for e in events if e["kind"] == "profile_failed"]


def test_plan_counts_the_profile_windows(ctx: RunnerContext) -> None:
    with_prof = Experiment.model_validate(mock_profile_doc())
    without = Experiment.model_validate({**mock_profile_doc(), "profile": None})
    _, plan_a = plan_experiment(with_prof, ctx)
    _, plan_b = plan_experiment(without, ctx)
    steps = [s for h in plan_a.hosts for s in h.steps if s.kind == "profile"]
    assert len(steps) == 2  # one per cell
    assert plan_a.total_seconds > plan_b.total_seconds
    assert plan_a.n_runs_max == plan_b.n_runs_max  # windows are not runs


def runpod_doc(**profile_over: object) -> dict:
    args = {
        "profiler_config": {
            "profiler": "torch",
            "torch_profiler_dir": "/var/lib/loom/profiles",
            "torch_profiler_with_stack": False,
        }
    }
    return {
        **mock_profile_doc(),
        "provider": {"kind": "runpod"},
        "variants": [{"name": "a", "engine": {"args": args}}],
        "profile": {"windows": [{"workload": "fixed-128-128"}], **profile_over},
    }


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda d: d["profile"]["windows"][0].update(workload="nope"), "not a workload label"),
        (lambda d: d["profile"]["windows"][0].update(load_values=[3]), "not in its load.values"),
        (lambda d: d["variants"][0].pop("engine"), "needs engine.args.profiler_config"),
        (
            lambda d: d["variants"][0]["engine"]["args"]["profiler_config"].update(
                torch_profiler_dir="/tmp/x"
            ),
            "must be the profile",
        ),
        (
            lambda d: d["variants"][0]["engine"]["args"]["profiler_config"].pop(
                "torch_profiler_with_stack"
            ),
            "with_stack: false",
        ),
        (lambda d: d.update(provider={"kind": "aws_ec2"}), "runpod and mock providers only"),
    ],
)
def test_profile_section_rules(mutate, match: str) -> None:
    Experiment.model_validate(runpod_doc())  # the base document is valid
    doc = json.loads(json.dumps(runpod_doc(), default=str))
    mutate(doc)
    with pytest.raises(ValidationError, match=match):
        Experiment.model_validate(doc)


def test_a_searched_workload_cannot_be_profiled() -> None:
    doc = runpod_doc()
    doc["workloads"][0]["load"] = {
        "mode": "open_loop",
        "search": {"lo": 1, "hi": 2},
        "duration_s": 5,
    }
    with pytest.raises(ValidationError, match=r"must use load\.values"):
        Experiment.model_validate(doc)


def test_engine_root() -> None:
    assert engine_root("http://127.0.0.1:8000/v1") == "http://127.0.0.1:8000"
    assert engine_root("http://h:1/v1/") == "http://h:1"


class _Server:
    """The mock app on a real port, so run_capture talks HTTP to it."""

    def __init__(self, config: MockConfig) -> None:
        import socket
        import threading

        import uvicorn

        self.sock = socket.socket()
        self.sock.bind(("127.0.0.1", 0))
        self.port = self.sock.getsockname()[1]
        self.server = uvicorn.Server(uvicorn.Config(create_app(config), log_level="critical"))
        self.thread = threading.Thread(
            target=self.server.run, kwargs={"sockets": [self.sock]}, daemon=True
        )
        self.thread.start()
        while not self.server.started:
            time.sleep(0.01)

    def close(self) -> None:
        self.server.should_exit = True
        self.thread.join(timeout=10)
        self.sock.close()


async def test_capture_starts_and_stops_with_bounds(tmp_path: Path) -> None:
    srv = _Server(MockConfig(time_scale=0.01, profile_dir=str(tmp_path)))
    try:
        cap = ProfileCapture(
            start_after_s=0.05, stop_after_s=0.2, prefix="loom-abc", max_iterations=5
        )
        t0 = time.perf_counter()
        base = f"http://127.0.0.1:{srv.port}/v1"
        async with httpx.AsyncClient(base_url=base, timeout=10) as http:

            async def load() -> None:
                while time.perf_counter() - t0 < 0.4:
                    await http.post(
                        "/completions",
                        json={"model": "mock-model", "prompt": "hi", "max_tokens": 8},
                    )

            out, _ = await asyncio.gather(run_capture(base, cap, t0), load())
        assert out["start_status"] == 200 and out["stop_status"] == 200, out
        assert 0.05 <= out["start_s"] < out["stop_s"]
        (trace,) = os.listdir(tmp_path)
        assert trace.startswith("loom-abc_rank0.")
    finally:
        srv.close()


async def test_capture_reports_a_disabled_profiler_without_raising() -> None:
    srv = _Server(MockConfig(time_scale=0.01))  # no profile_dir: 404, like vLLM without it
    try:
        cap = ProfileCapture(start_after_s=0.01, stop_after_s=0.02, prefix="p", max_iterations=1)
        out = await run_capture(f"http://127.0.0.1:{srv.port}/v1", cap, time.perf_counter())
        assert "error" in out and "404" in out["error"]
        assert "start_options_refused" in out  # retried without the body first
    finally:
        srv.close()


async def test_capture_survives_an_unreachable_engine() -> None:
    cap = ProfileCapture(start_after_s=0.01, stop_after_s=0.02, prefix="p", max_iterations=1)
    out = await run_capture("http://127.0.0.1:9/v1", cap, time.perf_counter())
    assert "error" in out


def test_window_latency_compares_profiled_and_settled_steps() -> None:
    def rec(gaps: list[float]) -> RequestRecord:
        return RequestRecord("r", RequestStatus.OK, 0.0, first_token_at_s=0.0, itl_s=gaps)

    # 0-2 s at 10 ms per token, then 2-3 s at 20 ms (the profiler on)
    r = rec([0.01] * 200 + [0.02] * 50)
    out = window_latency([r], 2.0, 3.0)
    assert out["before"]["itl_p50_ms"] == pytest.approx(10.0)
    assert out["profiled"]["itl_p50_ms"] == pytest.approx(20.0)
    assert out["profiler_slowdown"] == pytest.approx(2.0)


def test_shipped_specs_plan_under_their_caps(ctx: RunnerContext) -> None:
    from loom_bench.experiment import EXPERIMENTS_DIR, load_experiment

    totals = {}
    for name in ("qwen3-8b-profile-runpod", "runpod-smoke-profile"):
        exp = load_experiment(EXPERIMENTS_DIR / f"{name}.yaml")
        _, plan = plan_experiment(exp, ctx)
        assert plan.ok, (name, plan.refusals)
        assert [s.kind for s in plan.hosts[0].steps].count("profile") == 4  # 2 cells x 2
        totals[name] = (plan.total_micros, plan.ttl_worst_micros)
    # the proposal: everything, smoke included, under $6 even if every pod lives to its TTL
    assert sum(w for _, w in totals.values()) < 6_000_000


def test_smoke_mirrors_the_real_profile_run() -> None:
    """Every path the real run takes, the smoke takes: same model, provider, variants
    (checkpoint, KV dtype, engine args with the profiler config), workload profiles and
    load modes, repetitions, SLO, and the profile section's trace settings."""
    from loom_bench.experiment import EXPERIMENTS_DIR, load_experiment

    real = load_experiment(EXPERIMENTS_DIR / "qwen3-8b-profile-runpod.yaml")
    smoke = load_experiment(EXPERIMENTS_DIR / "runpod-smoke-profile.yaml")
    assert smoke.smoke and not real.smoke
    assert smoke.model == real.model and smoke.provider == real.provider
    assert smoke.variants == real.variants
    assert [(w.profile, w.load.mode) for w in smoke.workloads] == [
        (w.profile, w.load.mode) for w in real.workloads
    ]
    assert smoke.repetitions == real.repetitions and smoke.slo == real.slo
    assert real.profile is not None and smoke.profile is not None
    assert [w.workload for w in smoke.profile.windows] == [w.workload for w in real.profile.windows]
    for key in ("trace_dir", "max_trace_mb", "delay_iterations"):
        assert getattr(smoke.profile, key) == getattr(real.profile, key)
    assert real.quality is None and smoke.quality is None
