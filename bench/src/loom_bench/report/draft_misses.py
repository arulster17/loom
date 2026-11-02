"""What the draft could not predict: speculative runs' stored output text, per request.

An EAGLE3 head proposes only tokens in its draft vocabulary (RedHatAI's heads: the 32,000
most common, against 151,936 for Qwen3 and 128,256 for Llama 3). A request whose output
leaves that vocabulary can never have a draft accepted there. With a variant's
`store_output_text`, load runs keep each request's generated text, and this module lines
up, per request:

- tokens per chunk (completion tokens / streamed chunks: ~1 means no draft accepted);
- the share of output tokens outside the draft vocabulary, and which ones;
- repetition: distinct tokens / tokens, and the most common token's share (a greedy
  continuation stuck in a loop).

then compares the low-acceptance requests with the rest. Output text is re-tokenized,
which is the generated ids up to the tokenizer's decode/encode round trip.

The draft vocabulary comes from the head's `d2t` tensor (draft id i stands for target id
i + d2t[i]), read from its safetensors file: locally, or from Hugging Face with two
ranged requests (the header and d2t, ~256 KB, not the 2 GB of weights).
"""

from __future__ import annotations

import json
import statistics
import struct
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path

import httpx
from pydantic import BaseModel

from loom_bench.records import RequestRecord

# safetensors dtypes the vocabulary tensors use.
_INT64 = "I64"
_BOOL = "BOOL"
LOW_ACCEPTANCE = 1.2  # tokens per chunk at or below this count as "low"


def _vocab_from(header: dict, read: Callable[[int, int], bytes]) -> frozenset[int]:
    """Target ids the draft can propose, from a parsed safetensors header and a reader of
    the data section (offset, length)."""
    if "d2t" in header:
        meta = header["d2t"]
        if meta["dtype"] != _INT64 or len(meta["shape"]) != 1:
            raise ValueError(f"d2t: expected a 1-D I64 tensor, got {meta['dtype']} {meta['shape']}")
        start, end = meta["data_offsets"]
        raw = read(start, end - start)
        offsets = struct.unpack(f"<{meta['shape'][0]}q", raw)
        return frozenset(i + off for i, off in enumerate(offsets))
    if "t2d" in header:
        meta = header["t2d"]
        if meta["dtype"] != _BOOL:
            raise ValueError(f"t2d: expected BOOL, got {meta['dtype']}")
        start, end = meta["data_offsets"]
        return frozenset(i for i, b in enumerate(read(start, end - start)) if b)
    raise ValueError("no d2t or t2d tensor: the head's vocabulary is the target's whole one")


def draft_vocab_from_file(path: str | Path) -> frozenset[int]:
    """The draft vocabulary of a local safetensors file."""
    with Path(path).open("rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(n))

        def read(offset: int, length: int) -> bytes:
            f.seek(8 + n + offset)
            return f.read(length)

        return _vocab_from(header, read)


def draft_vocab_from_hub(
    repo: str,
    revision: str,
    filename: str = "model.safetensors",
    token: str | None = None,
    transport: httpx.BaseTransport | None = None,
) -> frozenset[int]:
    """The draft vocabulary of a pinned Hugging Face head, fetched with ranged requests."""
    url = f"https://huggingface.co/{repo}/resolve/{revision}/{filename}"
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    with httpx.Client(follow_redirects=True, timeout=60, transport=transport) as http:

        def get(start: int, length: int) -> bytes:
            r = http.get(url, headers={**headers, "Range": f"bytes={start}-{start + length - 1}"})
            r.raise_for_status()
            if len(r.content) != length:
                raise ValueError(f"{url}: asked {length} bytes at {start}, got {len(r.content)}")
            return r.content

        (n,) = struct.unpack("<Q", get(0, 8))
        header = json.loads(get(8, n))
        return _vocab_from(header, lambda offset, length: get(8 + n + offset, length))


class RequestOutput(BaseModel):
    request_id: str
    run_id: str | None = None
    tokens_per_chunk: float
    tokens: int  # output tokens after re-tokenizing the stored text
    outside_share: float  # share of output tokens the draft can never propose
    distinct_share: float  # distinct tokens / tokens
    top_token_share: float  # the most common token's share of the output
    top_outside: list[str]  # the most frequent outside-vocabulary tokens, as text


class GroupStats(BaseModel):
    n: int
    outside_share_median: float | None
    distinct_share_median: float | None
    top_token_share_median: float | None


class DraftMissReport(BaseModel):
    threshold: float
    vocab_size: int
    requests: list[RequestOutput]  # low acceptance first, then by tokens per chunk
    low: GroupStats
    rest: GroupStats
    skipped_no_text: int  # successful requests whose run did not store text

    def finding(self) -> str:
        """One plain sentence comparing the low-acceptance requests with the rest."""
        lo, rest = self.low, self.rest
        if lo.n == 0:
            return (
                f"No request had {self.threshold:g} tokens per chunk or fewer ({rest.n} with text)."
            )

        def pct(x: float | None) -> str:
            return "n/a" if x is None else f"{x:.1%}"

        return (
            f"{lo.n} of {lo.n + rest.n} requests had at most {self.threshold:g} tokens per "
            f"chunk. Median share of their output outside the {self.vocab_size:,}-token "
            f"draft vocabulary: {pct(lo.outside_share_median)} (other requests "
            f"{pct(rest.outside_share_median)}); distinct tokens {pct(lo.distinct_share_median)} "
            f"({pct(rest.distinct_share_median)}); most common token "
            f"{pct(lo.top_token_share_median)} ({pct(rest.top_token_share_median)})."
        )


def _median(xs: Sequence[float]) -> float | None:
    return statistics.median(xs) if xs else None


def _group(rows: Sequence[RequestOutput]) -> GroupStats:
    return GroupStats(
        n=len(rows),
        outside_share_median=_median([r.outside_share for r in rows]),
        distinct_share_median=_median([r.distinct_share for r in rows]),
        top_token_share_median=_median([r.top_token_share for r in rows]),
    )


def analyze_outputs(
    records: Iterable[tuple[str | None, RequestRecord]],
    *,
    encode: Callable[[str], list[int]],
    decode_token: Callable[[int], str],
    draft_vocab: frozenset[int],
    threshold: float = LOW_ACCEPTANCE,
    top: int = 5,
) -> DraftMissReport:
    """Per-request output analysis of (run id, record) pairs: successful, measured
    requests with stored text. `encode` maps text to target token ids."""
    if not draft_vocab:
        raise ValueError("empty draft vocabulary")
    rows: list[RequestOutput] = []
    skipped = 0
    for run_id, r in records:
        if not r.ok or r.warmup or not r.completion_tokens:
            continue
        if not r.output_text:
            skipped += 1
            continue
        ids = encode(r.output_text)
        if not ids:
            continue
        counts = Counter(ids)
        outside = Counter({i: c for i, c in counts.items() if i not in draft_vocab})
        rows.append(
            RequestOutput(
                request_id=r.request_id,
                run_id=run_id,
                tokens_per_chunk=r.completion_tokens / (len(r.itl_s) + 1),
                tokens=len(ids),
                outside_share=sum(outside.values()) / len(ids),
                distinct_share=len(counts) / len(ids),
                top_token_share=counts.most_common(1)[0][1] / len(ids),
                top_outside=[decode_token(i) for i, _ in outside.most_common(top)],
            )
        )
    rows.sort(key=lambda x: (x.tokens_per_chunk > threshold, x.tokens_per_chunk))
    low = [x for x in rows if x.tokens_per_chunk <= threshold + 1e-9]
    rest = [x for x in rows if x.tokens_per_chunk > threshold + 1e-9]
    return DraftMissReport(
        threshold=threshold,
        vocab_size=len(draft_vocab),
        requests=rows,
        low=_group(low),
        rest=_group(rest),
        skipped_no_text=skipped,
    )


def markdown(report: DraftMissReport, limit: int = 20) -> str:
    """The finding, then the lowest-acceptance requests as a table."""
    lines = [
        report.finding(),
        "",
        "| Request | Tokens per chunk | Tokens | Outside draft vocab | Distinct | Top token "
        "| Most frequent outside tokens |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in report.requests[:limit]:
        outside = ", ".join(repr(t) for t in r.top_outside)
        lines.append(
            f"| {r.request_id} | {r.tokens_per_chunk:.2f} | {r.tokens} | {r.outside_share:.1%} "
            f"| {r.distinct_share:.1%} | {r.top_token_share:.1%} | {outside} |"
        )
    if report.skipped_no_text:
        lines += ["", f"{report.skipped_no_text} requests had no stored text (not analysed)."]
    return "\n".join(lines)
