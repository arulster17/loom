"""Divergence in two halves: capture the reference once, score candidates later.

The oracle is the original simultaneous measurement (both endpoints up, each
prompt continued by the reference and scored by both sides at once); capture +
score, including a JSON round trip of the capture, must give the same result.
"""

import asyncio
from collections.abc import Iterator, Sequence

import pytest

pytest.importorskip("uvicorn")

from loom_bench.quality.client import EvalClient
from loom_bench.quality.divergence import (
    DivergenceResult,
    PromptDivergence,
    ReferenceLogprobs,
    aggregate,
    capture_reference,
    compare_positions,
    load_prompts,
    measure_divergence,
    score_against_reference,
    scored_positions,
)

from .mock_serve import MODEL, serve

TOP_K, NEW_TOKENS, SEED, N_BOOT = 5, 16, 1234, 500
PROMPTS = load_prompts()[:12]


@pytest.fixture(scope="module")
def clean_url() -> Iterator[str]:
    with serve(time_scale=0.001) as url:
        yield url


@pytest.fixture(scope="module")
def noisy_url() -> Iterator[str]:
    with serve(time_scale=0.001, degrade=0.3, logprob_noise=3.0) as url:
        yield url


def client(url: str) -> EvalClient:
    return EvalClient(url, MODEL, seed=SEED)


async def simultaneous(ref_url: str, cand_url: str, prompts: Sequence[str]) -> DivergenceResult:
    async with client(ref_url) as ref, client(cand_url) as cand:

        async def score(c: EvalClient, text: str):
            return (await c.complete(text, max_tokens=1, echo=True, logprobs=TOP_K)).logprobs

        async def one(prompt: str) -> PromptDivergence | None:
            cont = await ref.complete(prompt, max_tokens=NEW_TOKENS)
            if not cont.text:
                return None
            text = prompt + cont.text
            ref_lp, cand_lp = await asyncio.gather(score(ref, text), score(cand, text))
            span = (len(prompt), len(text))
            return compare_positions(
                scored_positions(ref_lp, *span), scored_positions(cand_lp, *span)
            )

        results = await asyncio.gather(*(one(p) for p in prompts))
    kept = [r for r in results if r is not None]
    return aggregate(
        kept, top_k=TOP_K, n_skipped=len(results) - len(kept), n_boot=N_BOOT, seed=SEED
    )


async def capture(url: str) -> ReferenceLogprobs:
    async with client(url) as ref:
        return await capture_reference(ref, PROMPTS, top_k=TOP_K, max_new_tokens=NEW_TOKENS)


async def score(url: str, reference: ReferenceLogprobs) -> DivergenceResult:
    async with client(url) as cand:
        return await score_against_reference(cand, reference, n_boot=N_BOOT, seed=SEED)


async def test_capture_then_score_equals_the_simultaneous_measurement(clean_url, noisy_url):
    expected = await simultaneous(clean_url, noisy_url, PROMPTS)
    reference = await capture(clean_url)
    stored = ReferenceLogprobs.model_validate_json(reference.model_dump_json())
    assert stored == reference
    assert await score(noisy_url, stored) == expected
    assert expected.kl.point > 0.05 and expected.top1.point < 0.95


async def test_measure_divergence_is_capture_plus_score(clean_url, noisy_url):
    expected = await simultaneous(clean_url, noisy_url, PROMPTS)
    async with client(clean_url) as ref, client(noisy_url) as cand:
        got = await measure_divergence(
            ref, cand, PROMPTS, top_k=TOP_K, max_new_tokens=NEW_TOKENS, n_boot=N_BOOT, seed=SEED
        )
    assert got == expected


@pytest.fixture(scope="module")
def no_offsets_url() -> Iterator[str]:
    # Same weights as `clean_url`, but every text_offset is -1, as SGLang reports them.
    with serve(time_scale=0.001, text_offsets=False) as url:
        yield url


async def test_a_server_without_text_offsets_scores_against_one_with_them(
    clean_url, no_offsets_url
):
    # The 2026-10-06 smoke run (b97dea4c): a vLLM reference scored against SGLang
    # failed because SGLang's -1 offsets selected no positions.
    reference = await capture(clean_url)
    got = await score(no_offsets_url, reference)
    same = await score(clean_url, reference)
    assert got.n_positions == same.n_positions > 0
    assert got.kl.point == pytest.approx(0.0, abs=1e-9) and got.top1.point == 1.0


async def test_a_reference_captured_without_text_offsets(clean_url, no_offsets_url):
    reference = await capture(no_offsets_url)
    assert reference.prompts and all(p.positions for p in reference.prompts if p.continuation)
    assert reference.prompts == (await capture(clean_url)).prompts
    got = await score(clean_url, reference)
    assert got.kl.point == pytest.approx(0.0, abs=1e-9) and got.top1.point == 1.0


async def test_one_capture_scores_many_candidates(clean_url, noisy_url):
    reference = await capture(clean_url)
    assert reference.model == MODEL and reference.top_k == TOP_K
    assert [p.prompt for p in reference.prompts] == PROMPTS
    assert all(p.positions for p in reference.prompts if p.continuation)

    same = await score(clean_url, reference)
    assert same.kl.point == pytest.approx(0.0, abs=1e-9) and same.top1.point == 1.0
    noisy = await score(noisy_url, reference)
    assert noisy.n_prompts == same.n_prompts and noisy.n_positions == same.n_positions
    assert noisy.kl.point > 0.05


@pytest.fixture(scope="module")
def byte_level_urls() -> Iterator[tuple[str, str]]:
    # A byte-level tokenizer splitting multi-byte characters across tokens, rendered as
    # vLLM does (offsets known, "" then the whole character) and as SGLang does (offsets
    # -1, every token alone, fragments as latin-1 bytes): the 565b8d3f sweep's failure.
    with (
        serve(time_scale=0.001, byte_level=True) as vllm,
        serve(time_scale=0.001, byte_level=True, text_offsets=False) as sglang,
    ):
        yield vllm, sglang


async def test_byte_split_characters_score_across_renderings(byte_level_urls):
    vllm, sglang = byte_level_urls
    reference = await capture(vllm)
    tokens = [x.token for p in reference.prompts for x in p.positions]
    assert "" in tokens and any(ord(c) > 127 for t in tokens for c in t)
    same = await score(vllm, reference)
    got = await score(sglang, reference)
    assert got.n_positions == same.n_positions > 0
    assert got.kl.point == pytest.approx(0.0, abs=1e-9) and got.top1.point == 1.0


async def test_a_byte_split_reference_captured_per_token(byte_level_urls):
    vllm, sglang = byte_level_urls
    reference = await capture(sglang)
    assert reference.prompts == (await capture(vllm)).prompts
    got = await score(vllm, reference)
    assert got.kl.point == pytest.approx(0.0, abs=1e-9) and got.top1.point == 1.0
