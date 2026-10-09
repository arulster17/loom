"""Plain-English summary at the top of a leaderboard.

Per model and workload: what one replica costs at the SLO ($/1M input, $/1M output under
the measured prefill-time split, and blended at the workload's own token mix), which
config that figure is for and why, what limits it (the SLO target that failed at the
next load, or at the lowest load when none passed), the quality the gate verified, and
the caveats, stated with their numbers. Where no config has a cost at the declared SLO
and an alternative-SLO analysis is given, its figure is quoted and labelled as such.

Everything is generated from the results, so the same data always gives the same text.
Nothing here changes a check, an SLO or the ranking: it quotes them.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from fractions import Fraction
from typing import Any

from pydantic import BaseModel

from loom_bench.competitiveness import like_for_like, price_at_mix
from loom_bench.cost import MicrosRange
from loom_bench.prices import Competitors
from loom_bench.registry import Registry
from loom_bench.report.analyze import UNTRUSTING, ConfigResult, LoadPoint, gate_label
from loom_bench.report.format import (
    describe_slo,
    est,
    goodput_bracket,
    html_env,
    load_value,
    md_table,
    usd,
    usd_ci,
)
from loom_bench.report.leaderboard import (
    Leaderboard,
    LeaderboardReport,
    LeaderboardRow,
    RowStatus,
    board_title,
    headline_row,
    no_headline_reason,
    quality_text,
    quality_verified,
    split_ranges,
)
from loom_bench.slo import slo_met

# Engine environment settings worth naming when they explain a limit.
ENV_NOTES = {
    ("NCCL_P2P_DISABLE", "1"): (
        "The engine ran with NCCL_P2P_DISABLE=1 (no GPU peer-to-peer on this host), so "
        "tensor-parallel all-reduces go through host memory."
    ),
}


class SummaryLine(BaseModel):
    """One config of one workload in the summary table."""

    config: str
    quoted: bool  # the figure the workload's text quotes
    standing: str
    goodput: str
    cost_input: MicrosRange | None
    cost_output: MicrosRange | None
    cost_blended: MicrosRange | None
    quality: str


class WorkloadSummary(BaseModel):
    title: str
    workload: str
    shape: str
    lines: list[SummaryLine]
    points: list[str]


class ModelSummary(BaseModel):
    model: str
    hardware: list[str]
    workloads: list[WorkloadSummary]
    quality: list[str]


class Summary(BaseModel):
    heading: str
    intro: str
    models: list[ModelSummary]


def standing(row: LeaderboardRow) -> str:
    if row.rank is not None:
        return f"rank {row.rank}"
    if row.status is RowStatus.UNTRUSTED:
        return "untrusted, not ranked"
    return row.status.value


def shape_text(r: ConfigResult | None) -> str:
    shape = r.tokens_per_request() if r else None
    if shape is None:
        return "tokens per request not measured"
    return f"{shape[0]:,.0f} input / {shape[1]:,.0f} output tokens per request"


def cost_sentence(r: ConfigResult) -> str:
    cin, cout, blended = split_ranges(r)
    return (
        f"{usd_ci(cin)} per 1M input tokens, {usd_ci(cout)} per 1M output tokens, "
        f"{usd_ci(blended)} per 1M tokens blended at this mix, holding the SLO up to "
        f"{goodput_bracket(r.goodput)}"
    )


def _metric_label(metric: str) -> str:
    if metric == "error_rate":
        return "error rate"
    name, _, pctl = metric.partition(".")
    return f"{name.removesuffix('_ms').upper()} {pctl}"


def failed_checks(r: ConfigResult, point: LoadPoint) -> list[str]:
    """Each SLO target missed at `point`, with the measured value and its CI."""
    out = []
    for c in slo_met(r.goodput.slo, point.aggregate).checks:
        if c.passed:
            continue
        e = point.aggregate.get(c.metric)
        if c.metric == "error_rate":
            value = "not measured" if e is None else f"{e.mean:.2%}"
            out.append(f"error rate {value} against the {c.target:.2%} limit")
            continue
        text = f"{_metric_label(c.metric)} is {est(e, 1, 'ms')} against the {c.target:g} ms target"
        if e is not None and e.mean <= c.target and c.used_upper_bound:
            text += " (the SLO is judged on the CI upper bound, which is over it)"
        out.append(text)
    return out


def env_note(r: ConfigResult) -> str | None:
    if r.gpus < 2:
        return None
    config = r.provenance.get("config")
    launch = config.get("launch") if isinstance(config, Mapping) else None
    env = launch.get("env") if isinstance(launch, Mapping) else None
    if not isinstance(env, Mapping):
        return None
    notes = [text for (k, v), text in ENV_NOTES.items() if str(env.get(k)) == v]
    return " ".join(notes) or None


def limit_sentence(r: ConfigResult) -> str:
    g = r.goodput
    by_load = {p.load: p for p in r.points}
    if g.max_load is not None and g.first_failing_load is not None:
        point = by_load[g.first_failing_load]
        misses = "; ".join(failed_checks(r, point)) or "the SLO verdict failed"
        text = f"What limits {r.name}: at {load_value(point.load, r.load_mode)}, {misses}."
    elif g.max_load is not None:
        text = (
            f"{r.name} met the SLO at every tested load: goodput is at least "
            f"{load_value(g.max_load, r.load_mode)}, so its cost is an upper bound."
        )
    else:
        point = r.points[0]
        checks = failed_checks(r, point)
        text = (
            f"{r.name} misses the SLO even at the lowest tested load, "
            f"{load_value(point.load, r.load_mode)}: {'; '.join(checks)}."
        )
        latency = [c for c in checks if not c.startswith("error rate")]
        if latency and len(latency) == len(checks):
            text += (
                " Missing a latency target at the lowest tested load points to the "
                "per-request latency of this config on this hardware, not to load."
            )
    note = env_note(r)
    return f"{text} {note}" if note else text


def caveat_sentence(r: ConfigResult) -> str | None:
    reasons = [w.message for w in r.warnings if w.kind in UNTRUSTING]
    if not reasons:
        return None
    return (
        f"Caveat: {r.name} is untrusted, so the leaderboard does not rank it: "
        f"{'; '.join(reasons)}. Treat its figure as indicative and rerun before relying on it."
    )


def _scores(r: ConfigResult) -> str:
    q = r.quality
    if q is None or not q.tasks:
        return "no task scores"
    return ", ".join(f"{t.task} {t.score:.3f}" for t in q.tasks)


def quality_sentence(r: ConfigResult, names: Mapping[str, str]) -> str:
    q = r.quality
    quant = r.label.quantization
    precision = "unquantized" if quant in (None, "none") else quant
    if q is None or (q.gate is None and not q.tasks):
        return f"{r.name} ({precision}): quality not evaluated."
    if q.gate is None:
        return (
            f"{r.name} ({precision}): scored, not gated (no other config of this model was "
            f"gated against it): {_scores(r)}."
        )
    if q.gate == "baseline":
        return (
            f"{r.name} ({precision}): the reference the quality gate compares against; "
            f"{_scores(r)}."
        )
    base = names.get(q.baseline_config_hash or "", (q.baseline_config_hash or "?")[:12])
    passed = [c.name for c in q.checks if c.verdict == "pass"]
    others = [f"{c.name} {c.verdict} ({c.reason})" for c in q.checks if c.verdict != "pass"]
    detail = []
    if passed:
        detail.append(f"{', '.join(passed)} pass")
    detail += others
    text = f"{r.name} ({precision}): quality gate vs {base} is {gate_label(q.gate)}"
    text += f": {'; '.join(detail)}." if detail else "."
    if q.gate == "inconclusive":
        text += (
            f" An inconclusive gate blocks the config: its quality is not shown to match {base}."
        )
    elif q.gate == "fail":
        text += " It is not ranked."
    return text


def _model_id(registry: Registry | None, repo: str) -> str | None:
    if registry is None:
        return None
    return next((m.id for m in registry.models if m.hf.repo == repo), None)


def _market_model_id(registry: Registry | None, model_id: str | None) -> str | None:
    """The registry id whose list prices apply: a quantized entry's base model."""
    if registry is None or model_id is None:
        return model_id
    return registry.market_model_id(model_id)


def market_sentence(
    r: ConfigResult,
    model_id: str | None,
    competitors: Competitors | None,
    label: str = "",
    *,
    base_model: bool = False,
) -> str | None:
    """Our blended cost against each public list price taken at this workload's mix.

    `model_id` is the registry entry whose listings apply; `base_model` says it is the
    base model of the quantized entry `r` serves (`Registry.market_model_id`). A disclosed
    competitor precision is marked like for like, or not, against ours.
    """
    if competitors is None or model_id is None:
        return None
    shape = r.tokens_per_request()
    _, _, blended = split_ranges(r)
    if shape is None or blended is None or blended.value is None or sum(shape) <= 0:
        return None
    share = Fraction(shape[0]) / Fraction(sum(shape))
    entries = competitors.entries_for(model_id)
    if not entries:
        return f"Market{label}: no public list price recorded for this model."
    ours = r.label.quantization
    parts, eligible = [], []
    for p, e in entries:
        mix = price_at_mix(e.input_per_mtok, e.output_per_mtok, share)
        precision: str | None = e.quantization
        if e.quantization is not None and ours is not None:
            precision = f"{e.quantization}, {like_for_like(e.quantization, ours)}"
        tags = [t for t in (precision, "aggregator" if p.aggregator else None) if t]
        if e.availability != "listed":
            tags.append(f"availability {e.availability}")
        if mix > 0:
            tags.append(f"our cost {blended.value / mix:.2f}×")
        parts.append(f"{p.name} {usd(mix)} ({', '.join(tags)})")
        if not p.aggregator and e.availability == "listed":
            eligible.append((mix, p.name))
    whose = (
        f" of {model_id} (the model this config serves at another precision; providers "
        "price the model)"
        if base_model
        else ""
    )
    text = (
        f"Market{label}: public list prices{whose} at this mix, per 1M tokens: "
        f"{'; '.join(parts)}. Our blended cost is {usd(blended.value)}"
    )
    if eligible:
        low, who = min(eligible)
        text += f", {blended.value / low:.2f}× the lowest eligible list price ({who}, {usd(low)})."
    else:
        text += (
            "; no listed price is eligible for the flags (aggregators and unverified "
            "listings are left out)."
        )
    return text


def _workload(
    b: Leaderboard,
    alt: Leaderboard | None,
    alt_slo: str | None,
    model_id: str | None,
    competitors: Competitors | None,
    alternative_view: bool = False,
    base_model: bool = False,
) -> WorkloadSummary:
    """`model_id` is the registry entry whose public list prices apply; `base_model`
    says it is the base model of this board's quantized entry."""
    head = headline_row(b.rows)
    lines = [
        SummaryLine(
            config=row.result.name,
            quoted=row is head,
            standing=standing(row),
            goodput=goodput_bracket(row.result.goodput),
            cost_input=split_ranges(row.result)[0],
            cost_output=split_ranges(row.result)[1],
            cost_blended=split_ranges(row.result)[2],
            quality=quality_text(row.result),
        )
        for row in b.rows
    ]
    points: list[str] = []
    if head is not None:
        r = head.result
        text = f"{r.name}: {cost_sentence(r)}."
        leader = b.rows[0]
        if leader is not head and leader.rank is not None:
            text += (
                f" {leader.result.name} ranks first but its quality is not verified "
                f"({quality_text(leader.result)}), so the figure quoted is {r.name}'s "
                f"({quality_text(r)}, {standing(head)})."
            )
        elif head.rank is None:
            text += " No config on this board is trusted, so this figure is indicative only."
        unquantized_ungated = (r.quality is None or r.quality.gate is None) and (
            r.label.quantization in (None, "none")
        )
        if not quality_verified(r) and not unquantized_ungated:
            text += (
                f" Its quality is not verified against the reference ({quality_text(r)}); "
                "see Quality below."
            )
        points.append(text)
        if head.goodput_ties:
            points.append(
                f"{r.name} and {', '.join(head.goodput_ties)} are tied: their goodput brackets "
                "overlap, so the load search cannot separate them on cost; the latency at equal "
                "load table compares them."
            )
        points.append(limit_sentence(r))
    else:
        which = "this view's SLO" if alternative_view else "the declared SLO"
        points.append(f"No cost at {which}: {no_headline_reason(b.rows)}.")
        points += [limit_sentence(row.result) for row in b.rows]
    for row in b.rows:
        caveat = caveat_sentence(row.result)
        if caveat:
            points.append(caveat)
    alt_head = headline_row(alt.rows) if alt is not None and head is None else None
    if head is None and alt is not None:
        if alt_head is not None:
            r = alt_head.result
            points.append(
                f"Alternative SLO, NOT the declared one ({alt_slo}), for reference only: "
                f"{r.name} costs {cost_sentence(r)}."
            )
            caveat = caveat_sentence(r)
            if caveat:
                points.append(f"Under the alternative SLO: {caveat}")
        else:
            points.append(
                f"No config has a cost under the alternative SLO ({alt_slo}) either: "
                f"{no_headline_reason(alt.rows)}."
            )
    if head is not None:
        market = market_sentence(head.result, model_id, competitors, base_model=base_model)
    elif alt_head is not None:
        market = market_sentence(
            alt_head.result,
            model_id,
            competitors,
            " (at the alternative-SLO cost)",
            base_model=base_model,
        )
    else:
        market = None
    if market:
        points.append(market)
    quoted = head.result if head else (b.rows[0].result if b.rows else None)
    return WorkloadSummary(
        title=board_title(b),
        workload=b.workload,
        shape=shape_text(quoted),
        lines=lines,
        points=points,
    )


def _hardware(results: Sequence[ConfigResult]) -> list[str]:
    out: dict[str, None] = {}
    for r in results:
        lab = r.label
        where = " ".join(x for x in (lab.cloud, lab.instance_type) if x)
        gpus = f"{lab.gpu_count or '?'}×{lab.gpu_type}" if lab.gpu_type else "unknown GPU"
        price = r.prices.on_demand
        storage = f" incl. {r.prices.storage_gb} GB storage" if r.prices.storage_gb else ""
        priced = f"{usd(price)}/h on-demand{storage}" if price is not None else "no list price"
        out[f"{gpus} ({where}, {priced})"] = None
    return list(out)


INTRO = (
    "What one replica costs us to serve at the SLO ({slo}), at the on-demand list price "
    "including its storage, at the highest tested load that met the SLO; 95% confidence "
    "intervals in brackets. $/1M input and $/1M output split the replica's cost by measured "
    "prefill time (methodology at the end); $/1M blended is the replica's cost over all "
    "tokens at that workload's own input:output mix and needs no split. Each workload quotes "
    "one config: trusted first, then quality verified (the gate's reference or a gate pass), "
    "then leaderboard rank; a config that failed the quality gate is never quoted. Nothing "
    "here relaxes a check: untrusted figures are quoted only with their reason, and the "
    "ranking is the leaderboard's."
)


def build_summary(
    report: LeaderboardReport,
    *,
    alt: LeaderboardReport | None = None,
    competitors: Competitors | None = None,
    registry: Registry | None = None,
    view_note: str | None = None,
) -> Summary:
    """A summary of `report`; `alt` is an alternative-SLO analysis of the same runs, quoted
    (labelled) only where no config has a cost at the declared SLO. `view_note` prefixes
    the intro, e.g. for the alternative-SLO report itself."""
    alt_boards = {(b.model, b.workload, b.load_mode): b for b in (alt.boards if alt else [])}
    alt_slo = None
    if alt is not None and alt.boards and alt.boards[0].rows:
        alt_slo = describe_slo(alt.boards[0].rows[0].result.goodput.slo)
    slos = sorted({describe_slo(row.result.goodput.slo) for b in report.boards for row in b.rows})
    models: dict[str, list[Leaderboard]] = {}
    for b in report.boards:
        models.setdefault(b.model, []).append(b)
    out = []
    for model, boards in models.items():
        results = [row.result for b in boards for row in b.rows]
        model_id = _model_id(registry, model)
        market_id = _market_model_id(registry, model_id)
        names = {r.config_hash: r.name for r in results}
        seen: dict[str, ConfigResult] = {}
        for r in results:
            seen.setdefault(r.config_hash, r)
        out.append(
            ModelSummary(
                model=model,
                hardware=_hardware(results),
                workloads=[
                    _workload(
                        b,
                        alt_boards.get((b.model, b.workload, b.load_mode)),
                        alt_slo,
                        market_id,
                        competitors,
                        alternative_view=view_note is not None,
                        base_model=market_id != model_id,
                    )
                    for b in boards
                ],
                quality=[quality_sentence(r, names) for r in seen.values()],
            )
        )
    intro = INTRO.format(slo="; ".join(slos) or "none")
    if view_note:
        intro = f"{view_note} {intro}"
    return Summary(heading="Summary", intro=intro, models=out)


TABLE_HEADERS = (
    "Workload",
    "Config",
    "Standing",
    "Goodput at SLO",
    "$/1M input",
    "$/1M output",
    "$/1M blended",
    "Quality",
)


def _line_cells(w: WorkloadSummary, line: SummaryLine) -> list[Any]:
    return [
        w.workload,
        f"**{line.config}** (quoted)" if line.quoted else line.config,
        line.standing,
        line.goodput,
        usd_ci(line.cost_input),
        usd_ci(line.cost_output),
        usd_ci(line.cost_blended),
        line.quality,
    ]


def summary_markdown(s: Summary) -> str:
    parts = [f"## {s.heading}", "", s.intro]
    for m in s.models:
        parts += ["", f"### {m.model}", "", f"Hardware: {'; '.join(m.hardware)}.", ""]
        rows = [_line_cells(w, line) for w in m.workloads for line in w.lines]
        parts.append(md_table(TABLE_HEADERS, rows))
        for w in m.workloads:
            parts += ["", f"**{w.workload}** ({w.shape})", ""]
            parts += [f"- {p}" for p in w.points]
        parts += ["", "**Quality**", ""]
        parts += [f"- {q}" for q in m.quality]
    return "\n".join(parts)


def summary_html(s: Summary) -> str:
    return html_env().get_template("_summary.html.j2").render(summary=s, headers=TABLE_HEADERS)
