"""Acceptance: the gate blocks a deliberately broken config end to end.

The mock backend stands in for the engine. "Over-aggressive quantization" is
the mock with `degrade=0.3` (30% of prompts get wrong arithmetic or truncated
JSON) and heavy logit noise; the baseline is the clean mock. Every request goes
over HTTP through the real client, tasks, divergence measurement and gate.
"""

import socket
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager

import pytest

pytest.importorskip("loom_bench.mock")
uvicorn = pytest.importorskip("uvicorn")

from loom_bench.mock.config import MockConfig  # noqa: E402
from loom_bench.mock.server import create_app  # noqa: E402
from loom_bench.provenance import build_provenance  # noqa: E402
from loom_bench.quality.gate import Verdict  # noqa: E402
from loom_bench.quality.runner import (  # noqa: E402
    gate_against_baseline,
    record_gate,
    record_suite_result,
    run_divergence,
    run_suite,
)
from loom_bench.quality.suite import Suite  # noqa: E402
from loom_bench.store.db import session_scope, upgrade  # noqa: E402
from loom_bench.store.models import BenchEvalRun, BenchGateDecision  # noqa: E402
from loom_bench.store.repo import create_experiment  # noqa: E402

MODEL = "mock-model"
CLEAN = {"time_scale": 0.001}
BROKEN = {"time_scale": 0.001, "degrade": 0.3, "logprob_noise": 3.0}


@contextmanager
def serve(**overrides) -> Iterator[str]:
    """Run the mock on a free port in a background thread; yields its /v1 base URL."""
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    config = MockConfig(models=[MODEL], **overrides)
    server = uvicorn.Server(uvicorn.Config(create_app(config), log_level="critical"))
    thread = threading.Thread(target=server.run, kwargs={"sockets": [sock]}, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while not server.started:
        assert time.monotonic() < deadline, "mock server did not start"
        time.sleep(0.005)
    try:
        yield f"http://127.0.0.1:{port}/v1"
    finally:
        server.should_exit = True
        thread.join(timeout=5)
        sock.close()


def suite(n: int, *, min_samples: int = 300) -> Suite:
    return Suite.model_validate(
        {
            "suite": "mock-acceptance",
            "model": MODEL,
            "seed": 1234,
            "gate": {"threshold": 0.01, "min_samples": min_samples, "n_boot": 2000},
            "tasks": [
                {"name": "arithmetic", "kind": "toy_arithmetic", "params": {"n": n, "seed": 7}},
                {
                    "name": "json_schema",
                    "kind": "json_schema",
                    "threshold": 0.06,
                    "min_samples": 50,
                },
            ],
            "divergence": {"max_new_tokens": 32, "top_k": 5, "max_kl": 0.05, "min_top1": 0.95},
        }
    )


@pytest.fixture(scope="module")
def baseline_url() -> Iterator[str]:
    with serve(**CLEAN) as url:
        yield url


@pytest.fixture(scope="module")
def same_url() -> Iterator[str]:
    """A second server with the baseline's exact config (a no-op config change)."""
    with serve(**CLEAN) as url:
        yield url


@pytest.fixture(scope="module")
def broken_url() -> Iterator[str]:
    with serve(**BROKEN) as url:
        yield url


async def _gate(suite_: Suite, baseline_url: str, candidate_url: str, tmp_path):
    base = await run_suite(suite_, baseline_url, MODEL, workdir=tmp_path / "base")
    cand = await run_suite(suite_, candidate_url, MODEL, workdir=tmp_path / "cand")
    div = await run_divergence(suite_, baseline_url, candidate_url, MODEL)
    return base, cand, div, gate_against_baseline(base, cand, suite_, div)


async def test_gate_blocks_over_aggressive_quantization(baseline_url, broken_url, tmp_path):
    s = suite(400)
    base, cand, div, decision = await _gate(s, baseline_url, broken_url, tmp_path)

    assert base.tasks["arithmetic"].estimate.mean == 1.0
    assert cand.tasks["arithmetic"].estimate.mean == pytest.approx(0.7, abs=0.08)
    by_task = {t.task: t for t in decision.tasks}
    assert by_task["arithmetic"].verdict is Verdict.FAIL
    assert by_task["arithmetic"].ci_high < -0.01
    assert by_task["json_schema"].verdict is Verdict.FAIL
    assert by_task["json_schema"].regressed > 0 and by_task["json_schema"].improved == 0
    assert div.kl.point > 0.05 and div.top1.point < 0.95
    assert decision.divergence.verdict is Verdict.FAIL
    assert decision.decision is Verdict.FAIL
    assert decision.blocked


async def test_identical_config_passes(baseline_url, same_url, tmp_path):
    s = suite(400)
    _, _, div, decision = await _gate(s, baseline_url, same_url, tmp_path)

    assert [t.verdict for t in decision.tasks] == [Verdict.PASS, Verdict.PASS]
    assert all(t.delta == 0.0 for t in decision.tasks)
    assert div.kl.point == pytest.approx(0.0, abs=1e-9) and div.top1.point == 1.0
    assert decision.sanity.verdict is Verdict.PASS
    assert decision.decision is Verdict.PASS
    assert not decision.blocked


async def test_small_sample_is_inconclusive(baseline_url, same_url, tmp_path):
    # 100 items clear a min_samples of 50, but cannot resolve a 1-point margin:
    # the CI is at least ±3/100.
    s = suite(100, min_samples=50)
    base = await run_suite(s, baseline_url, MODEL, workdir=tmp_path / "b", only=["arithmetic"])
    cand = await run_suite(s, same_url, MODEL, workdir=tmp_path / "c", only=["arithmetic"])
    decision = gate_against_baseline(base, cand, s)

    (arith,) = decision.tasks
    assert arith.delta == 0.0
    assert arith.verdict is Verdict.INCONCLUSIVE
    assert "more samples" in arith.reason
    assert decision.decision is Verdict.INCONCLUSIVE
    assert decision.blocked

    # Below min_samples the gate does not even look at the deltas.
    tiny = await run_suite(
        suite(20), baseline_url, MODEL, workdir=tmp_path / "t", only=["arithmetic"]
    )
    tiny_decision = gate_against_baseline(tiny, tiny, suite(20))
    assert tiny_decision.tasks[0].verdict is Verdict.INCONCLUSIVE
    assert "min_samples" in tiny_decision.tasks[0].reason


async def test_results_and_decision_persist(baseline_url, broken_url, tmp_path):
    s = suite(400)
    base, cand, _, decision = await _gate(s, baseline_url, broken_url, tmp_path)
    url = f"sqlite:///{tmp_path / 'loom.db'}"
    upgrade(url)
    base_prov = build_provenance({"engine": "mock", "degrade": 0.0})
    cand_prov = build_provenance({"engine": "mock", "degrade": 0.3})
    with session_scope(url) as session:
        exp = create_experiment(
            session, name="gate", spec={"suite": s.suite}, git=base_prov.git, budget_micros=None
        )
        record_suite_result(
            session,
            experiment_id=exp.id,
            config_hash=base_prov.config_hash,
            result=base,
            provenance=base_prov,
        )
        rows = record_suite_result(
            session,
            experiment_id=exp.id,
            config_hash=cand_prov.config_hash,
            result=cand,
            provenance=cand_prov,
        )
        row = record_gate(
            session,
            experiment_id=exp.id,
            baseline_config_hash=base_prov.config_hash,
            candidate_config_hash=cand_prov.config_hash,
            decision=decision,
        )
    with session_scope(url) as session:
        assert session.query(BenchEvalRun).count() == 4
        stored = session.get(BenchGateDecision, row.id)
        assert stored.decision == "fail"
        assert stored.details["blocked"] is True
        assert stored.details["divergence"]["verdict"] == "fail"
        arith = next(r for r in rows if r.task == "arithmetic")
        assert arith.n == 400 and arith.ci_low < arith.score < arith.ci_high
        assert arith.provenance["eval"]["suite"] == "mock-acceptance"
