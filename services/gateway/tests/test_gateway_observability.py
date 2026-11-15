"""Metrics and logs: the Prometheus series exist and move; logs never carry prompts,
completions or keys."""

from __future__ import annotations

import logging

import openai
import pytest

from loom_gateway.logs import JsonFormatter

pytestmark = [pytest.mark.postgres, pytest.mark.redis]

SECRET_PROMPT = "my bank PIN is 4931, do not repeat it"


async def test_metrics_and_logs(stack, caplog):
    t = await stack.tenant()
    client = stack.openai(t)
    stack.engine.behavior.token_delay_s = 0.01
    with caplog.at_level(logging.DEBUG):
        await client.chat.completions.create(
            model=stack.small, messages=[{"role": "user", "content": SECRET_PROMPT}]
        )
        stream = await client.chat.completions.create(
            model=stack.small,
            messages=[{"role": "user", "content": SECRET_PROMPT}],
            stream=True,
        )
        async for _ in stream:
            pass
        await stack.wait_events(2)
        with pytest.raises(openai.AuthenticationError):
            await stack.openai("loom_sk_" + "x" * 12 + "_" + "y" * 40).chat.completions.create(
                model=stack.small, messages=[{"role": "user", "content": SECRET_PROMPT}]
            )

    fmt = JsonFormatter()
    lines = [fmt.format(r) for r in caplog.records if r.name.startswith("loom_gateway")]
    assert any('"msg":"request"' in ln for ln in lines)
    assert any('"msg":"rejected"' in ln for ln in lines)
    for ln in lines:
        assert "4931" not in ln and "PIN" not in ln
        assert "tok1" not in ln  # completions either
        assert t.key not in ln and t.key[20:] not in ln

    text = stack.gateway.metrics.render().decode()
    m = stack.small
    for needle in (
        f'loom_gateway_requests_total{{endpoint="chat",model="{m}",status="200"}} 2.0',
        'loom_gateway_requests_total{endpoint="chat",model="-",status="401"} 1.0',
        'loom_gateway_errors_total{code="invalid_api_key"} 1.0',
        f'loom_gateway_ttft_seconds_count{{model="{m}"}} 1.0',
        f'loom_gateway_stream_tokens_per_second_count{{model="{m}"}} 1.0',
        f'loom_gateway_inflight_requests{{model="{m}"}} 0.0',
        'loom_gateway_overhead_seconds_count{phase="pre_dispatch"} 2.0',
        'loom_gateway_overhead_seconds_count{phase="settle"} 2.0',
        f'loom_gateway_tokens_total{{kind="completion",model="{m}"}} 10.0',
    ):
        assert needle in text, needle
