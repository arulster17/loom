"""Loom's prompt-logprobs route for trtllm-serve (loom_bench/trtllm_scorer.py), offline.

TensorRT-LLM itself is not importable here, so its pieces are stood in for by fakes that
follow its v1.4.0rc2 contracts: `generate_async(...).aresult()` returns outputs whose
`prompt_logprobs[j]` is the distribution that predicted prompt token j + 1 (the last entry
predicting the generated token), holding the top-k plus the actual token when it is not
among them, as `Logprob` objects (executor/result.py `compute_prompt_logprobs`,
base_worker.py `_compute_pytorch_prompt_logprobs`); the stock completions route rejects
`logprobs` (1.2.1) or returns generated-token logprobs only (1.4.0rc2).

The end-to-end case scores byte-split text (" √2", " 这句话") through the divergence
check twice: a vLLM-style reference built with the mock's vLLM renderer (an implementation
independent of the scorer's), and the fake TensorRT-LLM server behind the middleware, both
over the same toy model. Identical distributions must come out identical (KL 0, top-1
100%); an off-by-one alignment must not.
"""

import asyncio
import hashlib
import json
import math
import sys
import types
from dataclasses import dataclass
from typing import Any

import httpx
import pytest
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from loom_bench.engines import TRTLLM_SERVE_SOURCE
from loom_bench.mock.bytelevel import completion_logprobs, identity
from loom_bench.mock.logprobs import TokenLogprob
from loom_bench.quality.client import EvalClient
from loom_bench.quality.divergence import capture_reference, score_against_reference
from loom_bench.trtllm_scorer import (
    SCORE_CONCURRENCY,
    PromptLogprobsMiddleware,
    byte_decoder,
    echo_logprobs,
    hf_token_bytes,
    make_scorer,
)

K = 5
PROMPTS = {
    "The square root of two is": " written √2, and in Chinese 这句话 means this sentence.",
    "Café menus in München": " list straße prices in € and ∞ patience.",
}

# --- a toy byte-level tokenizer and model -------------------------------------------

TO_UNICODE = {b: c for c, b in byte_decoder().items()}


def pieces(text: str) -> list[bytes]:
    """Words with their leading space; a word holding a multi-byte character is split
    inside that character (after its lead byte), so both halves are fragments that are
    not valid UTF-8 alone, as Qwen3's byte-level BPE splits " √" and CJK."""
    out: list[bytes] = []
    for i, word in enumerate(text.split(" ")):
        if not word:
            continue
        data = (" " if i else "").encode() + word.encode()
        lead = next((j for j, b in enumerate(data) if b >= 0x80), None)
        out += [data] if lead is None else [data[: lead + 1], data[lead + 1 :]]
    return out


class ToyTokenizer:
    """Hugging Face's interface, as the scorer uses it, over a growing byte vocabulary."""

    added_tokens_decoder: dict[int, Any] = {0: "<|im_end|>"}  # noqa: RUF012

    def __init__(self) -> None:
        self.vocab: dict[bytes, int] = {b"<|im_end|>": 0}
        for text in [*PROMPTS, *PROMPTS.values(), " alpha beta gamma delta 予定 \xe4\xb8"]:
            self.encode(text)
        self.vocab.setdefault(b"\xe4\xb8", len(self.vocab))  # a CJK lead-byte fragment

    def id_of(self, unit: bytes) -> int:
        return self.vocab.setdefault(unit, len(self.vocab))

    def encode(self, text: str) -> list[int]:
        return [self.id_of(u) for u in pieces(text)]

    def unit(self, token_id: int) -> bytes:
        return next(u for u, i in self.vocab.items() if i == token_id)

    def convert_ids_to_tokens(self, token_id: int) -> str:
        if token_id in self.added_tokens_decoder:
            return self.added_tokens_decoder[token_id]
        return "".join(TO_UNICODE[b] for b in self.unit(token_id))


def toy_distribution(tok: ToyTokenizer, prefix: list[int], target: int) -> dict[int, float]:
    """Top-K logprobs at the position after `prefix`, the scripted next token `target`
    usually first. Deterministic in the prefix, so both servers see the same model."""
    seed = int(hashlib.sha256(json.dumps(prefix).encode()).hexdigest(), 16)
    pool = sorted(set(tok.vocab.values()) - {target})
    ranked = [target] + [pool[(seed >> (8 * i)) % len(pool)] for i in range(K + 2)]
    if seed % 4 == 0:  # sometimes the scripted token is not the model's top-1
        ranked[0], ranked[1] = ranked[1], ranked[0]
    if seed % 5 == 0:  # and sometimes not in the top-k at all
        ranked = ranked[1:] + ranked[:1]
    ranked = list(dict.fromkeys(ranked))
    return {t: -0.2 - 1.1 * r - (seed % 7) / 10 for r, t in enumerate(ranked)}


def logprob_of(tok: ToyTokenizer, prefix: list[int], target: int, token: int) -> float:
    return toy_distribution(tok, prefix, target).get(token, -12.0)


def script(tok: ToyTokenizer, text: str) -> list[int]:
    """Token ids of `text`, then the generated token (end of turn)."""
    return [*tok.encode(text), 0]


def top_k(dist: dict[int, float]) -> list[tuple[int, float]]:
    return sorted(dist.items(), key=lambda kv: -kv[1])[:K]


# --- the vLLM-style reference ---------------------------------------------------------


def reference_app(tok: ToyTokenizer) -> FastAPI:
    app = FastAPI()

    @app.post("/v1/completions")
    async def completions(request: Request) -> JSONResponse:
        body = await request.json()
        prompt = body["prompt"]
        if not body.get("echo"):
            return JSONResponse({"choices": [{"text": PROMPTS[prompt], "finish_reason": "stop"}]})
        ids = script(tok, prompt)
        entries: list[TokenLogprob | None] = [None]
        for j in range(1, len(ids)):
            dist = toy_distribution(tok, ids[:j], ids[j])
            entries.append(
                TokenLogprob(
                    token=identity(tok.unit(ids[j])),
                    logprob=logprob_of(tok, ids[:j], ids[j], ids[j]),
                    top=tuple((identity(tok.unit(t)), lp) for t, lp in top_k(dist)),
                )
            )
        lps = completion_logprobs([tok.unit(i) for i in ids], entries, incremental=True)
        return JSONResponse({"choices": [{"text": prompt, "logprobs": lps}]})

    return app


# --- the fake TensorRT-LLM server -------------------------------------------------------


@dataclass
class Logprob:
    logprob: float
    rank: int


class FakeSamplingParams:
    def __init__(self, **kwargs: Any) -> None:
        self.__dict__.update(kwargs)


class FakeGenerator:
    """`LLM.generate_async` with TensorRT-LLM's prompt_logprobs layout (`shift`=1); a
    `shift` of 0 is the bug of reading each entry one position early."""

    def __init__(self, tok: ToyTokenizer, shift: int = 1) -> None:
        self.tok, self.shift, self.calls = tok, shift, []

    def generate_async(self, ids: list[int], *, sampling_params: Any, cache_salt: str) -> Any:
        self.calls.append((ids, sampling_params, cache_salt))
        tok, k = self.tok, sampling_params.prompt_logprobs
        text = next(p + c for p, c in PROMPTS.items() if script(tok, p + c)[: len(ids)] == ids)
        full = script(tok, text)
        entries = []
        for t in range(len(ids)):
            # Entry t: the distribution after full[: t + 1] (shift 1), or, with the bug,
            # one position early; plus the actual next token when not in the top-k.
            at = max(t + self.shift, 1)
            dist = toy_distribution(tok, full[:at], full[at])
            entry = {i: Logprob(lp, r + 1) for r, (i, lp) in enumerate(top_k(dist)[:k])}
            nxt = full[t + 1]
            if nxt not in entry:
                entry[nxt] = Logprob(logprob_of(tok, full[:at], full[at], nxt), 99)
            entries.append(entry)
        out = types.SimpleNamespace(token_ids=[0], text="", prompt_logprobs=entries)

        class Promise:
            async def aresult(self) -> Any:
                return types.SimpleNamespace(outputs=[out])

        return Promise()


def stock_trtllm_app() -> FastAPI:
    """trtllm-serve's completions route as far as these requests go: logprobs refused."""
    app = FastAPI()

    @app.post("/v1/completions")
    async def completions(request: Request) -> JSONResponse:
        body = await request.json()
        if body.get("logprobs"):
            return JSONResponse({"error": {"message": "logprobs is not supported"}}, 400)
        return JSONResponse({"choices": [{"text": " stock", "finish_reason": "length"}]})

    return app


def fake_server(tok: ToyTokenizer, shift: int = 1) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        app=stock_trtllm_app(),
        generator=FakeGenerator(tok, shift),
        tokenizer=types.SimpleNamespace(tokenizer=tok),
        model="qwen3-8b",
    )


def trtllm_client(server: types.SimpleNamespace) -> EvalClient:
    app = PromptLogprobsMiddleware(server.app, make_scorer(server, FakeSamplingParams))
    return EvalClient(
        "http://trt/v1", "qwen3-8b", transport=httpx.ASGITransport(app=app), max_retries=0
    )


# --- tests --------------------------------------------------------------------------------


def test_byte_decoder_is_gpt2s_table():
    table = byte_decoder()
    assert len(table) == 256 and sorted(table.values()) == list(range(256))
    assert (table["Ġ"], table["Ċ"], table["!"]) == (0x20, 0x0A, 0x21)


def test_token_bytes_map_byte_level_pieces_and_keep_added_tokens():
    tok = ToyTokenizer()
    token_bytes = hf_token_bytes(tok)
    sqrt_lead, rest = tok.encode(" √2")  # b" \xe2", b"\x88\x9a2"
    assert token_bytes(sqrt_lead) == b" \xe2" and token_bytes(rest) == b"\x88\x9a2"
    assert token_bytes(0) == b"<|im_end|>"


def test_echo_logprobs_is_vllms_form():
    # The scorer's rendering against the mock's vLLM renderer, on byte-split tokens.
    tok = ToyTokenizer()
    ids = script(tok, next(iter(PROMPTS)) + next(iter(PROMPTS.values())))
    trt_entries = [toy_distribution(tok, ids[: j + 1], ids[j + 1]) for j in range(len(ids) - 1)]
    for j, entry in enumerate(trt_entries):  # TRT-LLM adds the actual token if not in top-k
        kept = dict(top_k(entry))
        kept.setdefault(ids[j + 1], entry.get(ids[j + 1], -12.0))
        trt_entries[j] = kept
    ours = echo_logprobs(ids, trt_entries, K, hf_token_bytes(tok))
    vllm_entries: list[TokenLogprob | None] = [None] + [
        TokenLogprob(
            token=identity(tok.unit(ids[j + 1])),
            logprob=trt_entries[j][ids[j + 1]],
            top=tuple((identity(tok.unit(t)), lp) for t, lp in top_k(trt_entries[j])),
        )
        for j in range(len(ids) - 1)
    ]
    assert ours == completion_logprobs([tok.unit(i) for i in ids], vllm_entries, incremental=True)
    assert "" in ours["tokens"]  # a fragment token renders "" before its character


def test_echo_logprobs_rejects_misaligned_entries():
    tok = ToyTokenizer()
    ids = tok.encode("alpha beta")
    with pytest.raises(ValueError, match="logprob entries"):
        echo_logprobs(ids, [{ids[1]: -1.0}] * len(ids), K, hf_token_bytes(tok))
    with pytest.raises(ValueError, match="no logprob for the scored token"):
        echo_logprobs(ids, [{ids[0]: -1.0}], K, hf_token_bytes(tok))


async def test_divergence_through_the_route_matches_an_identical_reference():
    tok = ToyTokenizer()
    ref_client = EvalClient(
        "http://ref/v1", "qwen3-8b", transport=httpx.ASGITransport(app=reference_app(tok))
    )
    reference = await capture_reference(ref_client, list(PROMPTS), top_k=K)
    server = fake_server(tok)
    result = await score_against_reference(trtllm_client(server), reference, n_boot=200)
    assert result.n_prompts == len(PROMPTS) and result.n_positions > 20
    assert result.kl.point == pytest.approx(0.0, abs=1e-9)
    assert result.top1.point == 1.0
    # One engine call per scored text, each with prompt logprobs, no KV reuse across them.
    calls = server.generator.calls
    assert len(calls) == len(PROMPTS)
    assert {c[1].prompt_logprobs for c in calls} == {K} and {c[1].max_tokens for c in calls} == {1}
    assert len({c[2] for c in calls}) == len(calls)


async def test_scoring_requests_reach_the_engine_one_at_a_time():
    """46938d1f: trtllm-bf16's LLM shut down under 16 concurrent prompt-logprobs requests
    (its smokes sent 4). The client still sends 16 at once; the engine sees one."""
    tok = ToyTokenizer()
    ref_client = EvalClient(
        "http://ref/v1", "qwen3-8b", transport=httpx.ASGITransport(app=reference_app(tok))
    )
    reference = await capture_reference(ref_client, list(PROMPTS), top_k=K)
    server = fake_server(tok)
    inner = server.generator.generate_async
    live = peak = 0

    def generate_async(ids: list[int], **kw: Any) -> Any:
        promise = inner(ids, **kw)

        class Counted:
            async def aresult(self) -> Any:
                nonlocal live, peak
                live += 1
                peak = max(peak, live)
                await asyncio.sleep(0.01)
                try:
                    return await promise.aresult()
                finally:
                    live -= 1

        return Counted()

    server.generator.generate_async = generate_async
    assert SCORE_CONCURRENCY == 1
    result = await score_against_reference(trtllm_client(server), reference, n_boot=200)
    assert result.n_prompts == len(PROMPTS) > 1
    assert peak == 1


async def test_an_off_by_one_alignment_does_not_pass_as_identical():
    tok = ToyTokenizer()
    ref_client = EvalClient(
        "http://ref/v1", "qwen3-8b", transport=httpx.ASGITransport(app=reference_app(tok))
    )
    reference = await capture_reference(ref_client, list(PROMPTS), top_k=K)
    result = await score_against_reference(
        trtllm_client(fake_server(tok, shift=0)), reference, n_boot=200
    )
    assert result.kl.point > 0.5 or result.top1.point < 0.9


async def test_other_requests_reach_trtllm_serve_untouched():
    server = fake_server(ToyTokenizer())
    app = PromptLogprobsMiddleware(server.app, make_scorer(server, FakeSamplingParams))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as c:
        plain = await c.post("/v1/completions", json={"prompt": "hi", "max_tokens": 1})
        assert plain.json()["choices"][0]["text"] == " stock"
        # logprobs without echo is TensorRT-LLM's own business (refused by 1.2.1).
        gen_only = await c.post("/v1/completions", json={"prompt": "hi", "logprobs": 5})
        assert gen_only.status_code == 400 and "not supported" in gen_only.text
        assert server.generator.calls == []
        bad_k = await c.post(
            "/v1/completions", json={"prompt": "alpha", "echo": True, "logprobs": 99}
        )
        assert bad_k.status_code == 400 and "loom prompt logprobs" in bad_k.text


def test_the_pod_program_installs_the_route_and_runs_trtllm_serve(monkeypatch):
    # TRTLLM_SERVE_SOURCE as the pod runs it (`python3 -c`), against stand-in modules.
    built: list[Any] = []
    served: list[bool] = []

    class OpenAIServer:
        def __init__(self, generator: Any, model: str) -> None:
            self.app, self.generator, self.model = stock_trtllm_app(), generator, model
            self.tokenizer = types.SimpleNamespace(tokenizer=ToyTokenizer())
            built.append(self)

    trt = types.ModuleType("tensorrt_llm")
    trt.SamplingParams = FakeSamplingParams  # type: ignore[attr-defined]
    serve_pkg = types.ModuleType("tensorrt_llm.serve")
    openai_server = types.ModuleType("tensorrt_llm.serve.openai_server")
    openai_server.OpenAIServer = OpenAIServer  # type: ignore[attr-defined]
    serve_pkg.openai_server = openai_server  # type: ignore[attr-defined]
    commands = types.ModuleType("tensorrt_llm.commands")
    serve_cmd = types.ModuleType("tensorrt_llm.commands.serve")
    serve_cmd.main = lambda: served.append(True)  # type: ignore[attr-defined]
    for name, mod in {
        "tensorrt_llm": trt,
        "tensorrt_llm.serve": serve_pkg,
        "tensorrt_llm.serve.openai_server": openai_server,
        "tensorrt_llm.commands": commands,
        "tensorrt_llm.commands.serve": serve_cmd,
    }.items():
        monkeypatch.setitem(sys.modules, name, mod)
    # loom_bench is not installed on the pod: the program must not need it.
    monkeypatch.setitem(sys.modules, "loom_bench.detokenize", None)

    exec(compile(TRTLLM_SERVE_SOURCE, "<trtllm-serve wrapper>", "exec"), {"__name__": "__main__"})
    assert served == [True]
    server = OpenAIServer(generator=FakeGenerator(ToyTokenizer()), model="qwen3-8b")
    stack = [m.cls.__name__ for m in server.app.user_middleware]
    assert stack == ["PromptLogprobsMiddleware"]
    assert built and math.isfinite(len(TRTLLM_SERVE_SOURCE))


def test_a_server_without_the_expected_attributes_fails_loudly(monkeypatch):
    class OpenAIServer:
        def __init__(self) -> None:
            self.app = FastAPI()

    trt = types.ModuleType("tensorrt_llm")
    trt.SamplingParams = FakeSamplingParams  # type: ignore[attr-defined]
    openai_server = types.ModuleType("tensorrt_llm.serve.openai_server")
    openai_server.OpenAIServer = OpenAIServer  # type: ignore[attr-defined]
    serve_pkg = types.ModuleType("tensorrt_llm.serve")
    serve_pkg.openai_server = openai_server  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "tensorrt_llm", trt)
    monkeypatch.setitem(sys.modules, "tensorrt_llm.serve", serve_pkg)
    monkeypatch.setitem(sys.modules, "tensorrt_llm.serve.openai_server", openai_server)
    from loom_bench.trtllm_scorer import install

    install()
    with pytest.raises(RuntimeError, match="needs updating for this TensorRT-LLM version"):
        OpenAIServer()
