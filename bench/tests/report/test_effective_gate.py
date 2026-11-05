"""Which gate decision stands when a config was gated more than once."""

import uuid
from datetime import UTC, datetime

from loom_bench.report.analyze import effective_gate, quality_for
from loom_bench.store.models import BenchGateDecision

from .factories import gate_row

BASE, CAND = "b" * 64, "c" * 64
STRICT = {"gsm8k": "pass", "ifeval": "inconclusive", "tool_calling_strict": "fail"}


def _gate(decision: str, day: int, tasks: dict[str, str]) -> BenchGateDecision:
    row = gate_row(
        BASE, CAND, decision, {"tasks": [{"task": t, "verdict": v} for t, v in tasks.items()]}
    )
    row.id = uuid.uuid5(uuid.NAMESPACE_URL, f"gate/{day}/{decision}")
    row.created_at = datetime(2026, 11, day, tzinfo=UTC)
    return row


def test_a_later_narrower_gate_does_not_clear_an_earlier_failure() -> None:
    # 55b35655 (TRT-LLM FP8): phase0-strict FAIL on tool_calling_strict on 11-04, then a
    # GSM8K-only PASS on 11-05 (qwen3-8b-trtllm-fp8-cudagraph-aws). The FAIL stands.
    full_fail, gsm8k_pass = _gate("fail", 4, STRICT), _gate("pass", 5, {"gsm8k": "pass"})
    assert effective_gate([gsm8k_pass, full_fail]) is full_fail
    q = quality_for(CAND, [], [full_fail, gsm8k_pass])
    assert q is not None and q.gate == "fail"
    assert {c.name for c in q.checks} == set(STRICT)


def test_a_later_gate_that_reran_every_task_replaces_the_earlier_one() -> None:
    full_fail = _gate("fail", 4, STRICT)
    rerun = _gate("pass", 6, {t: "pass" for t in STRICT})
    assert effective_gate([full_fail, rerun]) is rerun


def test_a_later_failure_stands_over_an_earlier_pass() -> None:
    rerun = _gate("pass", 6, {t: "pass" for t in STRICT})
    later_fail = _gate("fail", 7, {"gsm8k": "fail"})
    assert effective_gate([rerun, later_fail]) is later_fail
    assert effective_gate([]) is None
