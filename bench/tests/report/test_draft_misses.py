"""Draft-miss analysis: the head's vocabulary from its safetensors file (local or by
ranged requests), and per-request output statistics against tokens per chunk."""

import json
import struct

import httpx
import pytest

from loom_bench.records import RequestRecord, RequestStatus
from loom_bench.report.draft_misses import (
    analyze_outputs,
    draft_vocab_from_file,
    draft_vocab_from_hub,
    markdown,
)

WORDS = ["the", "cat", "sat", "on", "mat", "a", "dog", "ran", "zxq", "qqv", "wvb"]
IDS = {w: i for i, w in enumerate(WORDS)}
DRAFT = frozenset(range(8))  # the last three words are outside the draft vocabulary


def safetensors(tensors: dict[str, tuple[str, list[int], bytes]]) -> bytes:
    """A minimal safetensors file: 8-byte header length, JSON header, data."""
    header, data = {"__metadata__": {"format": "pt"}}, b""
    for name, (dtype, shape, raw) in tensors.items():
        header[name] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [len(data), len(data) + len(raw)],
        }
        data += raw
    blob = json.dumps(header).encode()
    return struct.pack("<Q", len(blob)) + blob + data


def head_with_d2t(draft_to_target: list[int]) -> bytes:
    offsets = [t - i for i, t in enumerate(draft_to_target)]
    weights = ("BF16", [2, 2], b"\0" * 8)
    d2t = ("I64", [len(offsets)], struct.pack(f"<{len(offsets)}q", *offsets))
    return safetensors({"d2t": d2t, "embed.weight": weights})


def test_vocab_from_d2t_and_from_t2d(tmp_path):
    f = tmp_path / "model.safetensors"
    f.write_bytes(head_with_d2t([0, 3, 4, 10, 42]))
    assert draft_vocab_from_file(f) == {0, 3, 4, 10, 42}
    mask = bytes([1, 0, 1, 1, 0])
    g = tmp_path / "t2d.safetensors"
    g.write_bytes(safetensors({"t2d": ("BOOL", [5], mask)}))
    assert draft_vocab_from_file(g) == {0, 2, 3}
    h = tmp_path / "none.safetensors"
    h.write_bytes(safetensors({"w": ("BF16", [1], b"\0\0")}))
    with pytest.raises(ValueError, match="no d2t or t2d"):
        draft_vocab_from_file(h)


def test_vocab_from_the_hub_reads_only_header_and_d2t():
    blob = head_with_d2t([1, 5, 7]) + b"\1" * 10_000  # weights it must not download
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/o/head/resolve/" + "a" * 40 + "/model.safetensors"
        rng = request.headers["Range"]
        seen.append(rng)
        start, end = (int(x) for x in rng.removeprefix("bytes=").split("-"))
        return httpx.Response(206, content=blob[start : end + 1])

    vocab = draft_vocab_from_hub("o/head", "a" * 40, transport=httpx.MockTransport(handler))
    assert vocab == {1, 5, 7}
    assert len(seen) == 3  # header length, header, d2t
    assert sum(int(r.split("-")[1]) - int(r.split("=")[1].split("-")[0]) + 1 for r in seen) < 500


def record(
    rid: str, text: str, chunks: int, *, ok: bool = True, tokens: int | None = None
) -> RequestRecord:
    n = tokens if tokens is not None else len(text.split())
    return RequestRecord(
        request_id=rid,
        status=RequestStatus.OK if ok else RequestStatus.ERROR,
        sent_at_s=0.0,
        first_token_at_s=0.1,
        finished_at_s=1.0,
        itl_s=[0.01] * (chunks - 1),
        prompt_tokens=5,
        completion_tokens=n,
        output_text=text,
    )


def encode(text: str) -> list[int]:
    return [IDS[w] for w in text.split()]


def test_low_acceptance_requests_are_the_ones_outside_the_draft_vocabulary():
    recs = [
        ("run1", record("a", "the cat sat on the mat a dog ran " * 3, chunks=10)),  # 2.7 tok/chunk
        ("run1", record("b", "the dog sat on a mat the cat ran " * 3, chunks=12)),
        ("run2", record("c", "zxq qqv zxq wvb the zxq qqv zxq wvb a", chunks=10)),  # 1.0
        ("run2", record("d", "zxq zxq zxq zxq zxq zxq zxq zxq", chunks=8)),  # a loop
        ("run2", record("e", "", chunks=1, tokens=5)),  # no stored text
        ("run2", record("f", "the cat", chunks=1, ok=False)),  # failed: ignored
    ]
    rep = analyze_outputs(
        recs, encode=encode, decode_token=lambda i: WORDS[i], draft_vocab=DRAFT, threshold=1.2
    )
    assert [r.request_id for r in rep.requests] == ["c", "d", "b", "a"]  # low first
    by = {r.request_id: r for r in rep.requests}
    assert by["c"].tokens_per_chunk == 1.0 and by["c"].outside_share == pytest.approx(0.8)
    assert by["c"].top_outside[:1] == ["zxq"] and by["c"].run_id == "run2"
    assert by["d"].distinct_share == pytest.approx(1 / 8) and by["d"].top_token_share == 1.0
    assert by["a"].outside_share == 0.0
    assert rep.low.n == 2 and rep.rest.n == 2 and rep.skipped_no_text == 1
    assert rep.low.outside_share_median == pytest.approx(0.9)
    assert rep.rest.outside_share_median == 0.0
    text = rep.finding()
    assert text.startswith("2 of 4 requests had at most 1.2 tokens per chunk")
    assert "draft vocabulary: 90.0% (other requests 0.0%)" in text
    md = markdown(rep)
    assert "| c | 1.00 | 10 | 80.0% |" in md and "1 requests had no stored text" in md


def test_no_low_requests_and_an_empty_vocabulary():
    rep = analyze_outputs(
        [("r", record("a", "the cat sat on", chunks=2))],
        encode=encode,
        decode_token=lambda i: WORDS[i],
        draft_vocab=DRAFT,
    )
    assert rep.low.n == 0 and rep.low.outside_share_median is None
    assert rep.finding().startswith("No request had 1.2 tokens per chunk or fewer")
    with pytest.raises(ValueError, match="empty draft vocabulary"):
        analyze_outputs([], encode=encode, decode_token=str, draft_vocab=frozenset())
