"""The stream client against TensorRT-LLM 1.2.1's SSE shape.

Chunks are built the way tensorrt_llm/serve/postprocess_handlers.py (v1.2.1) writes them,
not captured from a server: the role-only first chunk ends `} \\n\\n` (a space before the
blank line); with `stream_options.include_usage` every chunk also carries usage, because
1.2.1 defaults `continuous_usage_stats` to true; the last token chunk may have empty
content with the finish reason; extra choice fields (`avg_decoded_tokens_per_iter`,
`stop_reason`); then a usage chunk with no choices and `[DONE]`."""

import json
import time

import httpx
from fake_server import BASE_URL, scripted

from loom_bench.client.openai_stream import PreparedRequest, send
from loom_bench.records import RequestStatus

MODEL = "b968826d9c46dd6066d109eabc6255188de91218"  # 1.2.1 names a snapshot dir by its name


def usage(prompt: int, completion: int, cached: int = 0) -> dict:
    return {
        "prompt_tokens": prompt,
        "total_tokens": prompt + completion,
        "completion_tokens": completion,
        "prompt_tokens_details": {"cached_tokens": cached},
    }


def chunk(delta: dict, completion: int, finish: str | None = None, *, tail: str = "") -> bytes:
    choice: dict = {"index": 0, "delta": delta, "avg_decoded_tokens_per_iter": 1.0}
    if finish is not None:
        choice.update(finish_reason=finish, stop_reason=None)
    body = {
        "id": "chatcmpl-1",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": MODEL,
        "choices": [choice],
        "usage": usage(12, completion, cached=4),
    }
    return f"data: {json.dumps(body)}{tail}\n\n".encode()


def final_usage(completion: int) -> bytes:
    body = {"id": "chatcmpl-1", "object": "chat.completion.chunk", "created": 1,
            "model": MODEL, "choices": [], "usage": usage(12, completion, cached=4)}  # fmt: skip
    return f"data: {json.dumps(body)}\n\n".encode()


async def test_trtllm_stream_shape_is_parsed():
    handler = scripted(
        [
            chunk({"role": "assistant"}, 0, tail=" "),  # role only, trailing space
            0.02,
            chunk({"content": "Hel"}, 1),
            0.02,
            chunk({"content": "lo"}, 2),
            0.02,
            chunk({"content": ""}, 3, "length"),  # no text, only the finish reason
            final_usage(3),
            b"data: [DONE]\n\n",
        ]
    )
    req = PreparedRequest(
        request_id="r1",
        endpoint="chat",
        payload={"messages": [{"role": "user", "content": "hi"}], "max_tokens": 3},
        max_tokens=3,
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        rec = await send(http, BASE_URL, req, time.perf_counter(), timeout_s=5.0, keep_output=True)
    assert rec.status is RequestStatus.OK
    assert rec.output_text == "Hello"
    # The last usage seen wins: the final chunk's totals, not an earlier running count.
    assert (rec.prompt_tokens, rec.completion_tokens, rec.cached_prompt_tokens) == (12, 3, 4)
    assert rec.finish_reason == "length"
    # Two content chunks: one inter-token gap (the empty finish chunk is not a token).
    assert rec.first_token_at_s is not None and len(rec.itl_s) == 1
