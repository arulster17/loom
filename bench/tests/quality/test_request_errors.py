"""A task whose requests are rejected fails loudly instead of scoring 0 vs 0."""

import functools

import httpx
import pytest

from loom_bench.quality import runner as quality_runner
from loom_bench.quality.client import EvalClient
from loom_bench.quality.runner import run_suite
from loom_bench.quality.suite import Suite
from loom_bench.quality.tasks.base import (
    MAX_REJECTED_FRACTION,
    EvalTaskFailed,
    ItemResult,
    check_request_errors,
)

VLLM_400 = '"auto" tool choice requires --enable-auto-tool-choice and --tool-call-parser to be set'


def ok(i: int) -> ItemResult:
    return ItemResult(item_id=f"i{i}", score=1.0)


def err(i: int, status: int | None) -> ItemResult:
    return ItemResult(
        item_id=f"i{i}", score=0.0, meta={"error": f"HTTP {status}: boom {i}", "status": status}
    )


def test_clean_task_passes():
    check_request_errors("t", [ok(i) for i in range(4)])


def test_a_few_rejections_stay_scored():
    n = 20
    allowed = int(MAX_REJECTED_FRACTION * n)
    items = [err(i, 400) for i in range(allowed)] + [ok(i) for i in range(allowed, n)]
    check_request_errors("t", items)


def test_rejections_above_the_threshold_fail_with_status_and_first_error():
    n = 20
    bad = int(MAX_REJECTED_FRACTION * n) + 1
    items = [err(i, 400) for i in range(bad)] + [ok(i) for i in range(bad, n)]
    with pytest.raises(EvalTaskFailed, match=r"t: 3 of 20 requests rejected \(HTTP 400\)"):
        check_request_errors("t", items)


@pytest.mark.parametrize("status", [408, 429, 500, None])
def test_retryable_errors_below_all_stay_scored(status):
    check_request_errors("t", [err(0, status), err(1, status), ok(2), ok(3)])


@pytest.mark.parametrize("status", [500, None])
def test_every_request_failing_fails_the_task(status):
    with pytest.raises(EvalTaskFailed, match="t: all 3 requests failed"):
        check_request_errors("t", [err(i, status) for i in range(3)])


async def test_suite_fails_when_the_engine_rejects_tool_calls(tmp_path, monkeypatch):
    """The 51ad57b0 smoke bug: no tool-call parser, every tool request got HTTP 400."""

    def handler(request):
        return httpx.Response(400, json={"error": {"message": VLLM_400}})

    monkeypatch.setattr(
        quality_runner,
        "EvalClient",
        functools.partial(EvalClient, transport=httpx.MockTransport(handler), max_retries=0),
    )
    suite = Suite.model_validate(
        {
            "suite": "tools",
            "model": "m",
            "seed": 1,
            "tasks": [
                {"name": "tool_calling", "kind": "tool_calling", "params": {"limit": 4}},
            ],
        }
    )
    with pytest.raises(EvalTaskFailed) as info:
        await run_suite(suite, "http://t/v1", "m", workdir=tmp_path)
    message = str(info.value)
    assert message.startswith("tool_calling: 4 of 4 requests rejected (HTTP 400)")
    assert "--tool-call-parser" in message
