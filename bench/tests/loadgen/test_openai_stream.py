import asyncio
import json
import time

import httpx
import pytest
from fake_server import BASE_URL, DONE, chat_chunk, scripted, sse, usage_chunk

from loom_bench.client.openai_stream import PreparedRequest, new_record, send
from loom_bench.records import RequestStatus


def chat_req(**kw) -> PreparedRequest:
    base = dict(
        request_id="r1",
        endpoint="chat",
        payload={"messages": [{"role": "user", "content": "hi"}], "max_tokens": 3},
        expected_prompt_tokens=1,
        max_tokens=3,
        meta={"k": "v"},
    )
    base.update(kw)
    return PreparedRequest(**base)


async def run(handler, req=None, **kw):
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        t0 = time.perf_counter()
        kw.setdefault("timeout_s", 5.0)
        return await send(http, BASE_URL, req or chat_req(), t0, **kw)


async def test_parses_stream_timings_usage_and_finish_reason():
    handler = scripted(
        [
            0.05,
            chat_chunk(role="assistant"),  # role-only delta carries no tokens
            chat_chunk("Hel"),
            0.03,
            chat_chunk("lo"),
            0.03,
            chat_chunk("!", "stop"),
            usage_chunk(prompt=12, completion=3, cached=8),
            DONE,
        ]
    )
    rec = await run(handler, scheduled_at_s=0.0, keep_output=True)
    assert rec.status is RequestStatus.OK
    assert rec.http_status == 200
    assert rec.prompt_tokens == 12 and rec.completion_tokens == 3
    assert rec.cached_prompt_tokens == 8
    assert rec.finish_reason == "stop"
    assert rec.output_text == "Hello!"
    assert rec.ttft_s is not None and rec.ttft_s >= 0.045
    assert len(rec.itl_s) == 2 and all(g >= 0.025 for g in rec.itl_s)
    assert rec.finished_at_s >= rec.first_token_at_s + sum(rec.itl_s) - 1e-6
    assert rec.expected_prompt_tokens == 1 and rec.max_tokens == 3
    assert rec.meta == {"k": "v"} and rec.scheduled_at_s == 0.0


async def test_output_not_kept_by_default():
    rec = await run(scripted([chat_chunk("secret", "stop"), usage_chunk(1, 1), DONE]))
    assert rec.ok and rec.output_text is None


async def test_request_body_forces_streaming_with_usage_and_model():
    seen = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers.get("authorization")
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, content=DONE)

    req = chat_req(payload={"messages": [], "stream": False, "stream_options": {"x": 1}})
    await run(handler, req, model="m1", api_key="sk-test")
    assert seen["url"] == "http://fake/v1/chat/completions"
    assert seen["auth"] == "Bearer sk-test"
    assert seen["body"]["stream"] is True
    assert seen["body"]["stream_options"] == {"x": 1, "include_usage": True}
    assert seen["body"]["model"] == "m1"
    assert req.payload["stream"] is False  # caller's payload untouched


async def test_completions_endpoint_and_tool_call_deltas_count_as_tokens():
    comp = [
        sse({"choices": [{"index": 0, "text": "a", "finish_reason": None}]}),
        sse({"choices": [{"index": 0, "text": "b", "finish_reason": "length"}]}),
        DONE,
    ]
    rec = await run(scripted(comp), chat_req(endpoint="completions"), keep_output=True)
    assert rec.ok and rec.output_text == "ab" and len(rec.itl_s) == 1

    tool = [chat_chunk(tool_calls=[{"index": 0, "function": {"arguments": "{"}}]), 0.02]
    tool += [chat_chunk(tool_calls=[{"index": 0, "function": {"arguments": "}"}}]), DONE]
    rec = await run(scripted(tool))
    assert rec.first_token_at_s is not None and len(rec.itl_s) == 1


@pytest.mark.parametrize(
    ("body", "message"),
    [
        (
            b'{"error": {"message": "prompt too long", "type": "BadRequestError"}}',
            "prompt too long",
        ),
        (b'{"object": "error", "message": "bad model"}', "bad model"),
        (b'{"detail": "Not Found"}', "Not Found"),
        (b"upstream exploded", "upstream exploded"),
    ],
)
async def test_http_errors_record_status_and_message(body, message):
    async def handler(request):
        return httpx.Response(400, content=body)

    rec = await run(handler)
    assert rec.status is RequestStatus.ERROR
    assert rec.http_status == 400 and rec.error == message
    assert rec.first_token_at_s is None and rec.finished_at_s is not None


async def test_error_chunk_mid_stream():
    chunks = [chat_chunk("a"), sse({"error": {"message": "engine died"}}), DONE]
    rec = await run(scripted(chunks))
    assert rec.status is RequestStatus.ERROR and rec.error == "engine died"


async def test_malformed_and_truncated_streams_are_errors():
    rec = await run(scripted([b"data: {not json\n\n"]))
    assert rec.status is RequestStatus.ERROR and rec.error == "malformed stream chunk"
    rec = await run(scripted([chat_chunk("a")]))
    assert rec.status is RequestStatus.ERROR and "ended" in rec.error
    rec = await run(scripted([chat_chunk("a", "stop")]))  # no [DONE] but finished
    assert rec.ok


async def test_connection_error():
    def handler(request):
        raise httpx.ConnectError("refused")

    rec = await run(handler)
    assert rec.status is RequestStatus.ERROR and rec.error == "ConnectError: refused"
    assert rec.http_status is None


async def test_timeout_mid_stream():
    rec = await run(scripted([chat_chunk("a"), 10.0, DONE]), timeout_s=0.1)
    assert rec.status is RequestStatus.TIMEOUT
    assert rec.first_token_at_s is not None
    assert 0.09 <= rec.e2e_s < 1.0


async def test_cancel_mid_stream_marks_record_aborted_and_propagates():
    req = chat_req()
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(scripted([chat_chunk("a"), 10.0]))
    ) as http:
        t0 = time.perf_counter()
        rec = new_record(req, t0, scheduled_at_s=0.0)
        assert rec.status is RequestStatus.ABORTED
        task = asyncio.create_task(send(http, BASE_URL, req, t0, timeout_s=5, record=rec))
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert rec.status is RequestStatus.ABORTED
    assert rec.first_token_at_s is not None and rec.finished_at_s is not None
