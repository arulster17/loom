"""Speculative decoding on a leaderboard board: acceptance and speed-up by load.

Shown only on boards where some config speculates (its runs carry vLLM's
spec-decode counters, `metrics.prometheus`). For every load each config ran, one row per
config:

- **Accepted per step**: 1 + accepted / drafts, the tokens a request gains from one
  verify step that drafted (vLLM's "mean acceptance length", bonus token included).
  Steps where an n-gram lookup found nothing to draft are not in it.
- **Acceptance**: accepted / drafted tokens, then the per-position rates (the share of
  drafts whose token at position i was kept, which needs every earlier one kept too).
- **Tokens per chunk**: client-side, completion tokens per streamed chunk, averaged per
  request (mean of the run, then the lowest request): an engine streams once per step,
  so a request no draft helps sits at 1 however good the mean is.
- **TPOT p50 / p95**, **ITL p50** (the time between chunks: one engine step), and
  **TPOT p50 speed-up** against the board's one config without speculation (the
  reference), at the same load.

`crossover` names, per speculating config, the first common load at which its TPOT p50
is no longer below the reference's (point estimates; the CIs are in the table): where
speculation stops helping as the load grows.
"""

from __future__ import annotations

from collections.abc import Sequence

from pydantic import BaseModel

from loom_bench.records import LoadMode
from loom_bench.report.analyze import ConfigResult, LoadPoint
from loom_bench.report.format import NA, est, load_value, md_table, num
from loom_bench.stats import Estimate

ACCEPTED_LENGTH = "server.spec_mean_accepted_length"
ACCEPTANCE_RATE = "server.spec_acceptance_rate"
BY_POSITION = "server.spec_acceptance_by_position."
TOKENS_PER_CHUNK_MEAN = "tokens_per_chunk.mean"
TOKENS_PER_CHUNK_MIN = "tokens_per_chunk.min"
TPOT_P50 = "tpot_ms.p50"
TPOT_P95 = "tpot_ms.p95"
ITL_P50 = "itl_ms.p50"

HEADERS = (
    "Load",
    "Config",
    "Accepted per step",
    "Draft acceptance (by position)",
    "Tokens per chunk: mean, lowest request",
    "TPOT p50",
    "TPOT p95",
    "ITL p50 (step)",
    "TPOT p50 speed-up",
)

SPECDEC_NOTE = (
    "Speculative decoding by load. Accepted per step is 1 + accepted / drafts (vLLM's mean "
    "acceptance length, the target's own token included, over steps that drafted); draft "
    "acceptance is accepted / drafted tokens, then the share of drafts kept at each "
    "position. Tokens per chunk is measured by the client (completion tokens per streamed "
    "chunk, one chunk per engine step): its lowest request shows content the draft cannot "
    "predict, which the server's mean hides. ITL is the time between chunks, so under "
    "speculation it is the verify step's time, not a token's. Speed-up is the reference's "
    "TPOT p50 over the config's at the same load (point estimates)."
)


def speculates(r: ConfigResult) -> bool:
    return any(p.aggregate.get(ACCEPTED_LENGTH) is not None for p in r.points)


def at_goodput(r: ConfigResult, key: str) -> Estimate | None:
    """A metric at the goodput point (None without goodput or the metric)."""
    p = r.goodput_point
    return None if p is None else p.aggregate.get(key)


class SpecDecRow(BaseModel):
    load: float
    config: str
    accepted_length: Estimate | None
    acceptance_rate: Estimate | None
    by_position: list[Estimate]
    tokens_per_chunk_mean: Estimate | None
    tokens_per_chunk_min: Estimate | None
    tpot_p50: Estimate | None
    tpot_p95: Estimate | None
    itl_p50: Estimate | None
    speedup: float | None  # reference TPOT p50 / this config's, same load


class SpecDecTable(BaseModel):
    workload: str
    load_mode: LoadMode
    reference: str | None  # the board's one config without speculation
    rows: list[SpecDecRow]
    crossover: dict[str, str]  # speculating config -> sentence


def _positions(p: LoadPoint) -> list[Estimate]:
    found = {
        k.removeprefix(BY_POSITION): e
        for k, e in p.aggregate.metrics.items()
        if k.startswith(BY_POSITION)
    }
    return [found[k] for k in sorted(found, key=lambda k: (len(k), k))]


def _row(r: ConfigResult, p: LoadPoint, ref: dict[float, LoadPoint]) -> SpecDecRow:
    a = p.aggregate
    tpot = a.get(TPOT_P50)
    base = ref[p.load].aggregate.get(TPOT_P50) if p.load in ref else None
    speedup = base.mean / tpot.mean if base is not None and tpot is not None and tpot.mean else None
    return SpecDecRow(
        load=p.load,
        config=r.name,
        accepted_length=a.get(ACCEPTED_LENGTH),
        acceptance_rate=a.get(ACCEPTANCE_RATE),
        by_position=_positions(p),
        tokens_per_chunk_mean=a.get(TOKENS_PER_CHUNK_MEAN),
        tokens_per_chunk_min=a.get(TOKENS_PER_CHUNK_MIN),
        tpot_p50=tpot,
        tpot_p95=a.get(TPOT_P95),
        itl_p50=a.get(ITL_P50),
        speedup=speedup,
    )


def _crossover(rows: Sequence[SpecDecRow], mode: LoadMode) -> str:
    """Where a config's TPOT p50 stops beating the reference, from its compared loads."""
    compared = [r for r in rows if r.speedup is not None]
    if not compared:
        return "no load in common with the reference"
    trail = ", ".join(f"{r.speedup:.2f}x at {load_value(r.load, mode)}" for r in compared)
    first_loss = next((i for i, r in enumerate(compared) if (r.speedup or 0) <= 1.0), None)
    if first_loss is None:
        verdict = f"faster at every compared load (up to {load_value(compared[-1].load, mode)})"
    elif first_loss == 0:
        verdict = (
            f"no faster even at the lowest compared load ({load_value(compared[0].load, mode)})"
        )
    else:
        lo, hi = compared[first_loss - 1].load, compared[first_loss].load
        verdict = f"stops helping between {load_value(lo, mode)} and {load_value(hi, mode)}"
    return f"TPOT p50 speed-up {trail}: {verdict}"


def specdec_table(results: Sequence[ConfigResult]) -> SpecDecTable | None:
    """The board's speculative-decoding table; None when no config speculates."""
    if not results or not any(speculates(r) for r in results):
        return None
    plain = [r for r in results if not speculates(r)]
    reference = plain[0] if len(plain) == 1 else None
    ref = {p.load: p for p in reference.points} if reference else {}
    # The reference first at each load, then the board's order.
    order = [*([reference.name] if reference else []), *(r.name for r in results)]
    rows = [_row(r, p, ref) for r in results for p in r.points]
    rows.sort(key=lambda row: (row.load, order.index(row.config)))
    first = results[0]
    crossover = (
        {
            r.name: _crossover([row for row in rows if row.config == r.name], first.load_mode)
            for r in results
            if speculates(r)
        }
        if reference
        else {}
    )
    return SpecDecTable(
        workload=first.workload,
        load_mode=first.load_mode,
        reference=reference.name if reference else None,
        rows=rows,
        crossover=crossover,
    )


def _acceptance_text(row: SpecDecRow) -> str:
    if row.acceptance_rate is None:
        return NA
    text = est(row.acceptance_rate, 2)
    if row.by_position:
        text += " (" + ", ".join(num(e.mean, 2) for e in row.by_position) + ")"
    return text


def specdec_rows(t: SpecDecTable) -> list[list[str]]:
    """Table cells, as strings (markdown and HTML render the same text)."""
    out = []
    for row in t.rows:
        chunk = (
            NA
            if row.tokens_per_chunk_mean is None
            else f"{est(row.tokens_per_chunk_mean, 2)}; {est(row.tokens_per_chunk_min, 2)}"
        )
        out.append(
            [
                load_value(row.load, t.load_mode),
                row.config,
                est(row.accepted_length, 2),
                _acceptance_text(row),
                chunk,
                est(row.tpot_p50, 1, "ms"),
                est(row.tpot_p95, 1, "ms"),
                est(row.itl_p50, 1, "ms"),
                NA if row.speedup is None else f"{row.speedup:.2f}x",
            ]
        )
    return out


def specdec_lines(t: SpecDecTable) -> list[str]:
    """Sentences under the table: the reference, then each config's crossover."""
    if t.reference is None:
        return ["No single config without speculation on this board, so no speed-up is shown."]
    return [f"Reference (no speculation): {t.reference}."] + [
        f"{name}: {text}." for name, text in t.crossover.items()
    ]


def specdec_markdown(t: SpecDecTable) -> str:
    lines = "\n".join(f"- {s}" for s in specdec_lines(t))
    return f"{md_table(list(HEADERS), specdec_rows(t))}\n\n{lines}"
