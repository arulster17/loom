"""Teacher-forced prompt logprobs from inside trtllm-serve, for Loom's divergence check.

Loom's divergence check (quality/divergence.py) scores a text through `/v1/completions`
with `echo: true, logprobs: k, max_tokens: 1` and reads, for every prompt token, the top-k
distribution that predicted it. vLLM and SGLang answer that; trtllm-serve does not (1.2.1
rejects `logprobs` on completions, 1.4.0rc2 returns generated-token logprobs only, and no
version exposes prompt logprobs over HTTP). Its Python LLM API does: `SamplingParams(
prompt_logprobs=k)` on the PyTorch backend (tensorrt_llm/sampling_params.py and
executor/base_worker.py `_compute_pytorch_prompt_logprobs` at v1.4.0rc2; covered there by
tests/unittest/llmapi/test_llm_pytorch.py `test_llm_return_logprobs`).

So the engine is started through `main()` here instead of the `trtllm-serve` script: it
installs an ASGI middleware on trtllm-serve's own FastAPI app that answers exactly those
echo+logprobs completions requests from the server's own `LLM` object (`generate_async`
with `prompt_logprobs`), and passes every other request through untouched. The scores
therefore come from the served engine itself: same process, GPU, weights, precision,
KV-cache dtype and kernels as the load runs and eval tasks, not a second model instance.
Each scoring request gets a fresh `cache_salt`, so no KV block is reused and every prompt
position's logits are computed (prompt logprobs come from context logits).

The answer is written in vLLM v0.30.0's form (`echo_logprobs`): the tokens and top-k keys
rendered with `detokenize.incremental_text` from each token id's raw bytes, `text_offset`
as vLLM computes it, the token itself first in each top-k map. That is the reference's own
form, so nothing has to be reconstructed on the scoring side.

On the pod this file runs as `python3 -c` source right after detokenize.py's source
(`engines.TRTLLM_SERVE_SOURCE`), inside the TensorRT-LLM image, where loom_bench is not
installed: no `from __future__` import, and tensorrt_llm is imported only inside functions.
"""

import contextlib
import json
import time
import uuid
from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Any

# On the pod, detokenize.py's source precedes this file's and defines incremental_text.
with contextlib.suppress(ImportError):
    from loom_bench.detokenize import incremental_text

ROUTE = "/v1/completions"
# The divergence check asks for 5; the cap keeps a stray request from asking for a huge k.
MAX_K = 20

Score = Callable[[Mapping[str, Any]], Awaitable[dict[str, Any]]]


def byte_decoder() -> dict[str, int]:
    """GPT-2 byte-level BPE's character -> byte table (`bytes_to_unicode`, inverted): how
    a byte-level token string (Qwen, Llama 3) spells its raw bytes."""
    printable = [*range(ord("!"), ord("~") + 1), *range(0xA1, 0xAD), *range(0xAE, 0x100)]
    chars = list(printable)
    extra = 0
    for b in range(256):
        if b not in printable:
            printable.append(b)
            chars.append(256 + extra)
            extra += 1
    return {chr(c): b for b, c in zip(printable, chars, strict=True)}


def hf_token_bytes(tokenizer: Any) -> Callable[[int], bytes]:
    """A token id's raw bytes, from a Hugging Face byte-level BPE tokenizer. Added tokens
    (`<|im_end|>`, ...) are not byte-mapped: their text is their bytes."""
    table = byte_decoder()
    added = set(getattr(tokenizer, "added_tokens_decoder", {}) or {})

    def token_bytes(token_id: int) -> bytes:
        piece = tokenizer.convert_ids_to_tokens(int(token_id))
        if token_id not in added and all(c in table for c in piece):
            return bytes(table[c] for c in piece)
        return str(piece).encode("utf-8")

    return token_bytes


def wants_prompt_logprobs(body: Mapping[str, Any]) -> bool:
    """A completions request this module answers: echo with top-k logprobs on one prompt."""
    return (
        bool(body.get("echo"))
        and bool(body.get("logprobs"))
        and isinstance(body.get("prompt"), str)
    )


def echo_logprobs(
    ids: Sequence[int],
    entries: Sequence[Mapping[int, float]],
    k: int,
    token_bytes: Callable[[int], bytes],
) -> dict[str, Any]:
    """vLLM v0.30.0's completions `logprobs` for an echoed prompt plus one generated token.

    `ids` are the prompt's token ids followed by the generated one; `entries[j]` is the
    distribution that predicted `ids[j + 1]` (TensorRT-LLM's `prompt_logprobs[j]`, which
    holds the top-k and the actual token), so the first token has none, as in vLLM.
    """
    if len(entries) != len(ids) - 1:
        raise ValueError(f"{len(entries)} logprob entries for {len(ids)} tokens")
    units = [token_bytes(t) for t in ids]
    tokens = [incremental_text(u, units[:i]) for i, u in enumerate(units)]
    token_logprobs: list[float | None] = [None]
    tops: list[dict[str, float] | None] = [None]
    for j, entry in enumerate(entries):
        i, actual = j + 1, ids[j + 1]
        if actual not in entry:
            raise ValueError(f"position {i}: no logprob for the scored token {actual}")
        # vLLM writes the token itself first, then the top-k by rank; keys that render
        # alike collide and the last one written stays.
        top = {tokens[i]: float(entry[actual])}
        ranked = sorted(entry.items(), key=lambda kv: -kv[1])[:k]
        for token_id, lp in ranked:
            top[incremental_text(token_bytes(token_id), units[:i])] = float(lp)
        token_logprobs.append(float(entry[actual]))
        tops.append(top)
    offsets, pos = [], 0
    for t in tokens:
        offsets.append(pos)
        pos += len(t)
    return {
        "tokens": tokens,
        "token_logprobs": token_logprobs,
        "top_logprobs": tops,
        "text_offset": offsets,
    }


def _error(status: int, message: str) -> tuple[int, dict[str, Any]]:
    kind = "invalid_request_error" if status < 500 else "server_error"
    return status, {"error": {"message": message, "type": kind, "param": None, "code": status}}


class PromptLogprobsMiddleware:
    """ASGI middleware: echo+logprobs `POST /v1/completions` go to `score`; everything
    else reaches the wrapped app with its body intact."""

    def __init__(self, app: Any, score: Score) -> None:
        self.app = app
        self.score = score

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if (
            scope.get("type") != "http"
            or scope.get("method") != "POST"
            or scope.get("path") != ROUTE
        ):
            await self.app(scope, receive, send)
            return
        chunks: list[bytes] = []
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            chunks.append(message.get("body", b""))
            if not message.get("more_body", False):
                break
        body = b"".join(chunks)
        try:
            request = json.loads(body)
        except ValueError:
            request = None
        if not isinstance(request, dict) or not wants_prompt_logprobs(request):
            replayed = False

            async def replay() -> dict[str, Any]:
                nonlocal replayed
                if not replayed:
                    replayed = True
                    return {"type": "http.request", "body": body, "more_body": False}
                return await receive()

            await self.app(scope, replay, send)
            return
        try:
            status, payload = 200, await self.score(request)
        except ValueError as e:
            status, payload = _error(400, f"loom prompt logprobs: {e}")
        except Exception as e:  # reported to the client, not swallowed
            status, payload = _error(500, f"loom prompt logprobs: {type(e).__name__}: {e}")
        data = json.dumps(payload).encode()
        headers = [(b"content-type", b"application/json"), (b"content-length", b"%d" % len(data))]
        await send({"type": "http.response.start", "status": status, "headers": headers})
        await send({"type": "http.response.body", "body": data})


def make_scorer(server: Any, sampling_params: Callable[..., Any]) -> Score:
    """The scorer for a trtllm-serve `OpenAIServer`: its `generator` (the LLM), its
    tokenizer (the one its completions route encodes with) and its served model name."""
    tokenizer = getattr(server.tokenizer, "tokenizer", server.tokenizer)
    token_bytes = hf_token_bytes(tokenizer)

    async def score(request: Mapping[str, Any]) -> dict[str, Any]:
        k = int(request["logprobs"])
        if not 1 <= k <= MAX_K:
            raise ValueError(f"logprobs must be 1..{MAX_K}, got {k}")
        prompt = request["prompt"]
        ids = [int(t) for t in tokenizer.encode(prompt)]
        params = sampling_params(
            max_tokens=1, temperature=0.0, prompt_logprobs=k, seed=request.get("seed")
        )
        # A fresh salt: no KV block reuse, so every position's context logits are computed.
        result = await server.generator.generate_async(
            ids, sampling_params=params, cache_salt=uuid.uuid4().hex
        ).aresult()
        out = result.outputs[0]
        generated = [int(t) for t in out.token_ids][:1]
        if not generated or out.prompt_logprobs is None:
            raise RuntimeError("the engine returned no generated token or no prompt logprobs")
        entries = [
            {int(t): float(getattr(lp, "logprob", lp)) for t, lp in entry.items()}
            for entry in out.prompt_logprobs
        ]
        logprobs = echo_logprobs(ids + generated, entries, k, token_bytes)
        return {
            "id": f"cmpl-loom-{uuid.uuid4().hex}",
            "object": "text_completion",
            "created": int(time.time()),
            "model": server.model,
            "choices": [
                {
                    "index": 0,
                    "text": prompt + (out.text or ""),
                    "logprobs": logprobs,
                    "finish_reason": "length",
                }
            ],
            "usage": {
                "prompt_tokens": len(ids),
                "completion_tokens": len(generated),
                "total_tokens": len(ids) + len(generated),
            },
        }

    return score


def install() -> None:
    """Wrap `OpenAIServer.__init__` so every server it builds carries the middleware."""
    from tensorrt_llm import SamplingParams  # type: ignore[import-not-found]
    from tensorrt_llm.serve import openai_server  # type: ignore[import-not-found]

    cls = openai_server.OpenAIServer
    original = cls.__init__

    def __init__(self: Any, *args: Any, **kwargs: Any) -> None:
        original(self, *args, **kwargs)
        missing = [a for a in ("app", "generator", "tokenizer", "model") if not hasattr(self, a)]
        if missing:
            raise RuntimeError(
                f"loom prompt logprobs: OpenAIServer has no {missing}; "
                "loom_bench/trtllm_scorer.py needs updating for this TensorRT-LLM version"
            )
        self.app.add_middleware(PromptLogprobsMiddleware, score=make_scorer(self, SamplingParams))

    cls.__init__ = __init__


def main() -> None:
    """`trtllm-serve` with the prompt-logprobs route: the same CLI, after `install()`."""
    install()
    from tensorrt_llm.commands.serve import main as serve_main  # type: ignore[import-not-found]

    serve_main()


if __name__ == "__main__":
    main()
