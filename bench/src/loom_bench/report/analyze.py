"""Analysis from the store: bench_runs rows → one priced goodput result per config.

A sweep is every completed run of one (config_hash, workload, load_mode). Its runs
are grouped by load value, repetitions are aggregated (mean ± t-CI per metric), the
SLO is checked conservatively per load point, goodput is the highest passing load
below the first failure, and cost at SLO prices that goodput at each of the replica's
hourly prices (`cost.replica_prices`: on-demand, which ranks results, spot,
committed 1y and as run). Runs whose status is not "completed" are excluded and
counted.

Untrusted results are still reported but carry warnings, and leaderboards rank them
after trusted ones. A result is untrusted when any load point has fewer than two
completed repetitions, goodput is not bracketed by a failing load, a headline metric
at the goodput point varies run to run by more than `cv_warn`, or successful
requests lacked a usage block (their tokens are missing from throughput and cost).
"""

from __future__ import annotations

import datetime as dt
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from enum import StrEnum
from statistics import median
from typing import Any, Literal, NamedTuple, cast

from pydantic import BaseModel

from loom_bench.cost import (
    CostAllocation,
    CostAtSlo,
    PriceColumn,
    ReplicaPrices,
    cost_from_goodput,
    replica_prices,
)
from loom_bench.metrics.aggregate import HEADLINE_METRICS, AggregateSummary, aggregate_runs
from loom_bench.metrics.summary import RunSummary
from loom_bench.money import Micros
from loom_bench.prices import PriceBook
from loom_bench.provenance import ContentKind, config_hash
from loom_bench.records import LoadMode
from loom_bench.slo import GoodputResult, Slo, find_goodput
from loom_bench.stats import Estimate
from loom_bench.store.models import (
    BenchColdStart,
    BenchEvalRun,
    BenchGateDecision,
    BenchRun,
    ColdStartKind,
)

COMPLETED = "completed"

# Engine-arg keys that carry tensor parallelism (vLLM, SGLang); shown as TPn, not as args.
TP_ARG_KEYS = ("tensor_parallel_size", "tensor-parallel-size", "tp_size", "tp-size", "tp")

PriceResolver = Callable[[Mapping[str, Any]], ReplicaPrices]


class SweepKey(NamedTuple):
    config_hash: str
    workload: str
    load_mode: LoadMode


class LoadPoint(BaseModel):
    """All completed repetitions at one load value."""

    load: float
    run_ids: list[str]
    repetitions: list[int | None]
    aggregate: AggregateSummary


class WarningKind(StrEnum):
    SINGLE_REPETITION = "single_repetition"
    UNBRACKETED_GOODPUT = "unbracketed_goodput"
    HIGH_CV = "high_cv"
    MISSING_USAGE = "missing_usage"
    NO_GOODPUT = "no_goodput"
    NO_PRICE = "no_price"
    EXCLUDED_RUNS = "excluded_runs"


UNTRUSTING = frozenset(
    {
        WarningKind.SINGLE_REPETITION,
        WarningKind.UNBRACKETED_GOODPUT,
        WarningKind.HIGH_CV,
        WarningKind.MISSING_USAGE,
    }
)


# Short reason per warning kind, for tables (the message carries the detail).
WARNING_LABELS: dict[WarningKind, str] = {
    WarningKind.SINGLE_REPETITION: "single repetition",
    WarningKind.UNBRACKETED_GOODPUT: "goodput not bracketed",
    WarningKind.HIGH_CV: "high run-to-run variance",
    WarningKind.MISSING_USAGE: "missing token usage",
    WarningKind.NO_GOODPUT: "no tested load met the SLO",
    WarningKind.NO_PRICE: "no hourly price",
    WarningKind.EXCLUDED_RUNS: "runs not completed were excluded",
}


class ResultWarning(BaseModel):
    kind: WarningKind
    message: str

    @property
    def label(self) -> str:
        return WARNING_LABELS[self.kind]


class ConfigLabel(BaseModel):
    engine: str | None
    engine_version: str | None
    quantization: str | None
    tp: int | None
    gpu_type: str | None
    gpu_count: int | None
    instance_type: str | None
    cloud: str | None
    region: str | None
    market: str | None
    engine_args: dict[str, Any]

    @property
    def text(self) -> str:
        parts = [" ".join(p for p in (self.engine, self.engine_version) if p) or "unknown engine"]
        if self.quantization:
            parts.append("unquantized" if self.quantization == "none" else self.quantization)
        if self.tp is not None:
            parts.append(f"TP{self.tp}")
        if self.gpu_type:
            parts.append(f"{self.gpu_count or '?'}×{self.gpu_type}")
        if self.instance_type:
            parts.append(
                f"{self.instance_type} ({self.market})" if self.market else self.instance_type
            )
        if self.engine_args:
            parts.append(", ".join(f"{k}={_arg(v)}" for k, v in sorted(self.engine_args.items())))
        return " · ".join(parts)


class TaskQuality(BaseModel):
    task: str
    task_version: str | None
    n: int
    score: float
    ci_low: float | None
    ci_high: float | None
    baseline_score: float | None
    delta: float | None  # score - baseline score on the same task


GateStatus = Literal["pass", "review", "fail", "inconclusive", "baseline"]
GATE_LABELS: dict[str, str] = {"review": "needs review"}


def gate_label(gate: str) -> str:
    return GATE_LABELS.get(gate, gate)


class GateDivergence(BaseModel):
    """A gate decision's logprob divergence and the limits it was held to."""

    kl: float  # mean KL(reference || candidate), nats
    top1: float  # top-1 agreement
    max_kl: float | None
    min_top1: float | None
    calibrated: bool  # limits from the baseline's measured noise floor
    noise_multiple: float
    self_kl: float | None
    self_top1: float | None

    @classmethod
    def from_details(cls, details: Mapping[str, Any]) -> GateDivergence | None:
        """From `GateDecision.details()`; None when divergence was not measured or the
        decision predates calibrated limits."""
        div = details.get("divergence") or {}
        result, limits = div.get("result"), div.get("limits")
        if result is None or limits is None:
            return None
        return cls(
            kl=result["kl"]["point"],
            top1=result["top1"]["point"],
            max_kl=limits["max_kl"],
            min_top1=limits["min_top1"],
            calibrated=limits["calibrated"],
            noise_multiple=limits["noise_multiple"],
            self_kl=limits["self_kl"],
            self_top1=limits["self_top1"],
        )

    def text(self) -> str:
        kl_lim = "" if self.max_kl is None else f" (limit {self.max_kl:.4f})"
        top1_lim = "" if self.min_top1 is None else f" (limit {self.min_top1:.1%})"
        if self.calibrated and self.self_kl is not None and self.self_top1 is not None:
            floor = (
                f"limits {self.noise_multiple:g}x the baseline's noise floor of "
                f"KL {self.self_kl:.4f}, top-1 {self.self_top1:.1%}, or the absolute ones"
            )
        else:
            floor = "uncalibrated: absolute limits, no noise floor measured"
        return f"KL {self.kl:.4f} nats{kl_lim}, top-1 {self.top1:.1%}{top1_lim}; {floor}"


class Quality(BaseModel):
    gate: GateStatus | None  # None: no gate decision involves this config
    baseline_config_hash: str | None
    tasks: list[TaskQuality]
    divergence: GateDivergence | None = None  # of the latest gate decision on this config

    def worst(self) -> TaskQuality | None:
        """The task with the most negative delta vs baseline."""
        deltas = [t for t in self.tasks if t.delta is not None]
        return min(deltas, key=lambda t: (t.delta, t.task)) if deltas else None


class ColdStartStat(BaseModel):
    median_s: float
    n: int


class ConfigResult(BaseModel):
    config_hash: str
    workload: str
    cell_key: str | None
    label: ConfigLabel
    model_repo: str | None
    model_revision: str | None
    content: ContentKind | None
    load_mode: LoadMode
    gpus: int
    points: list[LoadPoint]
    goodput: GoodputResult
    allocation: str
    prices: ReplicaPrices
    cost: CostAtSlo | None  # at the on-demand price: what results are ranked by
    spot_cost: CostAtSlo | None
    committed_1y_cost: CostAtSlo | None
    as_run_cost: CostAtSlo | None
    ttft_p95_ms: Estimate | None  # at the goodput point
    tpot_p95_ms: Estimate | None
    peak_output_tok_s: Estimate | None  # raw peak over all loads, SLO ignored
    peak_load: float | None
    warnings: list[ResultWarning]
    experiment_ids: list[str]
    run_ids: list[str]
    provenance_digests: list[str]
    git_shas: list[str]
    provenance: dict[str, Any]  # representative record (first run at the lowest load)
    quality: Quality | None = None

    @property
    def name(self) -> str:
        return self.cell_key or self.config_hash[:12]

    @property
    def hourly_micros(self) -> Micros | None:
        """The on-demand hourly price, the one results are ranked by."""
        return self.prices.on_demand

    def cost_at(self, column: PriceColumn) -> CostAtSlo | None:
        if column is PriceColumn.ON_DEMAND:
            return self.cost
        return cast(CostAtSlo | None, getattr(self, f"{column.value}_cost"))

    @property
    def key(self) -> SweepKey:
        """One result per key: a config can be benchmarked on several workloads."""
        return SweepKey(self.config_hash, self.workload, self.load_mode)

    @property
    def trusted(self) -> bool:
        if any(w.kind in UNTRUSTING for w in self.warnings):
            return False
        return self.goodput.max_load is None or self.goodput.trusted

    @property
    def main_warning(self) -> ResultWarning | None:
        """The warning that best explains the verdict: the first one that makes the result
        untrusted, else the first of any kind (no goodput, no price, ...)."""
        untrusting = (w for w in self.warnings if w.kind in UNTRUSTING)
        return next(untrusting, self.warnings[0] if self.warnings else None)

    @property
    def goodput_point(self) -> LoadPoint | None:
        return next((p for p in self.points if p.load == self.goodput.max_load), None)

    @property
    def goodput_output_tok_s_per_gpu(self) -> Estimate | None:
        est = self.goodput.output_tok_s
        if est is None:
            return None
        return est.scaled(1 / self.gpus)

    def reproduce_run_id(self) -> str:
        """A run whose provenance reproduces this sweep: first run at goodput, else first."""
        point = self.goodput_point or self.points[0]
        return point.run_ids[0]


def _arg(v: Any) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    return str(v)


def _section(prov: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = prov.get(name)
    return value if isinstance(value, Mapping) else {}


def _tp(
    engine_args: Mapping[str, Any], config: Mapping[str, Any], prov: Mapping[str, Any]
) -> int | None:
    recorded = _section(prov, "parallelism").get("tp")
    if recorded is not None:
        return int(recorded)
    for key in TP_ARG_KEYS:
        if key in engine_args:
            return int(engine_args[key])
    tp = _section(config, "parallelism").get("tp")
    return None if tp is None else int(tp)


def label_from_provenance(prov: Mapping[str, Any]) -> ConfigLabel:
    engine = _section(prov, "engine")
    hardware = _section(prov, "hardware")
    args = dict(engine.get("args") or {})
    return ConfigLabel(
        engine=engine.get("name"),
        engine_version=engine.get("version"),
        quantization=_section(prov, "model").get("quantization"),
        tp=_tp(args, _section(prov, "config"), prov),
        gpu_type=hardware.get("gpu_type"),
        gpu_count=hardware.get("gpu_count"),
        instance_type=hardware.get("instance_type"),
        cloud=prov.get("cloud"),
        region=prov.get("region"),
        market=prov.get("market"),
        engine_args={k: v for k, v in args.items() if k not in TP_ARG_KEYS},
    )


def provenance_digest(prov: Mapping[str, Any]) -> str:
    """sha256 of the stored record; equals `provenance_hash` of the Provenance it came from."""
    return config_hash(prov)


def default_price_resolver(
    price_book: PriceBook, *, allow_unverified: bool = False
) -> PriceResolver:
    """Every price column for a run from `price_book` and its provenance
    (`cost.replica_prices`): the one pricing used by reports, the site, the run
    summary and `goodput.json`."""

    def resolve(prov: Mapping[str, Any]) -> ReplicaPrices:
        return replica_prices(prov, price_book, allow_unverified=allow_unverified)

    return resolve


def _sweep_key(run: BenchRun) -> SweepKey:
    if run.load_mode is None or run.workload is None:
        raise ValueError(f"run {run.id} has no load_mode or workload")
    return SweepKey(run.config_hash, run.workload, LoadMode(run.load_mode))


def _group(runs: Iterable[BenchRun]) -> tuple[dict[SweepKey, list[BenchRun]], dict[SweepKey, int]]:
    completed: dict[SweepKey, list[BenchRun]] = defaultdict(list)
    excluded: dict[SweepKey, int] = defaultdict(int)
    for run in runs:
        if run.status != COMPLETED:
            if run.load_mode is not None and run.workload is not None:
                excluded[_sweep_key(run)] += 1
            continue
        if run.summary is None or run.load_value is None:
            raise ValueError(f"completed run {run.id} has no summary or load_value")
        completed[_sweep_key(run)].append(run)
    return completed, excluded


def _points(
    runs: Sequence[BenchRun], confidence: float, cv_warn: float
) -> tuple[list[LoadPoint], list[RunSummary], list[BenchRun]]:
    by_load: dict[float, list[BenchRun]] = defaultdict(list)
    for run in runs:
        by_load[float(run.load_value or 0.0)].append(run)
    points: list[LoadPoint] = []
    summaries: list[RunSummary] = []
    ordered: list[BenchRun] = []
    for load in sorted(by_load):
        reps = sorted(by_load[load], key=lambda r: (r.repetition is None, r.repetition, str(r.id)))
        sums = [RunSummary.model_validate(r.summary) for r in reps]
        points.append(
            LoadPoint(
                load=load,
                run_ids=[str(r.id) for r in reps],
                repetitions=[r.repetition for r in reps],
                aggregate=aggregate_runs(sums, confidence=confidence, cv_warn=cv_warn),
            )
        )
        summaries.extend(sums)
        ordered.extend(reps)
    return points, summaries, ordered


def load_points(
    runs: Iterable[BenchRun], *, confidence: float = 0.95, cv_warn: float = 0.10
) -> dict[SweepKey, list[LoadPoint]]:
    """Completed runs aggregated per sweep and load value, with no SLO or pricing."""
    completed, _ = _group(runs)
    return {key: _points(rows, confidence, cv_warn)[0] for key, rows in completed.items()}


def _unit(mode: LoadMode) -> str:
    return "req/s" if mode is LoadMode.OPEN_LOOP else "concurrent"


def _warnings(
    key: SweepKey,
    points: list[LoadPoint],
    summaries: list[RunSummary],
    goodput: GoodputResult,
    prices: ReplicaPrices,
    excluded: int,
    cv_warn: float,
) -> list[ResultWarning]:
    unit = _unit(key.load_mode)
    out: list[ResultWarning] = []

    def add(kind: WarningKind, message: str) -> None:
        out.append(ResultWarning(kind=kind, message=message))

    for p in points:
        if p.aggregate.n_runs < 2:
            add(
                WarningKind.SINGLE_REPETITION,
                f"load {p.load:g} {unit}: {p.aggregate.n_runs} completed repetition; "
                "no confidence interval",
            )
    if goodput.max_load is None:
        add(
            WarningKind.NO_GOODPUT,
            f"no tested load met the SLO (lowest tested {points[0].load:g} {unit})",
        )
    elif not goodput.bracketed:
        add(
            WarningKind.UNBRACKETED_GOODPUT,
            f"every tested load met the SLO; goodput is at least {goodput.max_load:g} {unit} "
            "and cost at SLO is an upper bound",
        )
    at = next((p for p in points if p.load == goodput.max_load), None)
    if at is not None:
        for metric in HEADLINE_METRICS:
            est = at.aggregate.get(metric)
            if est is not None and est.cv is not None and est.cv > cv_warn:
                add(
                    WarningKind.HIGH_CV,
                    f"{metric} at goodput: run-to-run CV {est.cv:.1%} exceeds {cv_warn:.0%}",
                )
    missing = sum(s.requests_missing_usage for s in summaries)
    if missing:
        add(
            WarningKind.MISSING_USAGE,
            f"{missing} successful requests had no usage block; their tokens are not counted",
        )
    if prices.on_demand is None:
        reason = prices.missing or "no on-demand price for this replica"
        add(WarningKind.NO_PRICE, f"{reason}; cost at SLO not computed")
    if excluded:
        add(WarningKind.EXCLUDED_RUNS, f"{excluded} runs not completed were excluded")
    return out


def _git_label(prov: Mapping[str, Any]) -> str | None:
    git = _section(prov, "git")
    sha = git.get("sha")
    if sha is None:
        return None
    return f"{sha}-dirty" if git.get("dirty") else sha


def costs_at(
    hourly: Micros | None, goodput: GoodputResult, allocation: CostAllocation
) -> CostAtSlo | None:
    """Cost at SLO at one hourly price; None without a price or a goodput."""
    return None if hourly is None else cost_from_goodput(hourly, goodput, allocation)


def _analyze_sweep(
    key: SweepKey,
    runs: Sequence[BenchRun],
    excluded: int,
    *,
    slo: Slo,
    allocation: CostAllocation,
    price_resolver: PriceResolver,
    confidence: float,
    cv_warn: float,
) -> ConfigResult:
    points, summaries, ordered = _points(runs, confidence, cv_warn)
    gpus = {s.gpus for s in summaries}
    if len(gpus) != 1:
        raise ValueError(f"{key}: runs disagree on GPUs per replica: {sorted(gpus)}")
    goodput = find_goodput(slo, [(p.load, p.aggregate) for p in points], key.load_mode)
    prov = ordered[0].provenance
    prices = price_resolver(prov)
    costs = {col: costs_at(prices.get(col), goodput, allocation) for col in PriceColumn}
    at = next((p for p in points if p.load == goodput.max_load), None)

    peak: LoadPoint | None = None
    for p in points:
        est = p.aggregate.get("throughput.output_tok_s")
        best = peak.aggregate.get("throughput.output_tok_s") if peak else None
        if est is not None and (best is None or est.mean > best.mean):
            peak = p

    model = _section(prov, "model")
    content = _section(prov, "workload").get("content")
    cell_keys = sorted({r.cell_key for r in ordered if r.cell_key})
    return ConfigResult(
        config_hash=key.config_hash,
        workload=key.workload,
        cell_key=cell_keys[0] if len(cell_keys) == 1 else None,
        label=label_from_provenance(prov),
        model_repo=model.get("repo"),
        model_revision=model.get("revision"),
        content=None if content is None else ContentKind(content),
        load_mode=key.load_mode,
        gpus=gpus.pop(),
        points=points,
        goodput=goodput,
        allocation=allocation.describe(),
        prices=prices,
        cost=costs[PriceColumn.ON_DEMAND],
        spot_cost=costs[PriceColumn.SPOT],
        committed_1y_cost=costs[PriceColumn.COMMITTED_1Y],
        as_run_cost=costs[PriceColumn.AS_RUN],
        ttft_p95_ms=at.aggregate.get("ttft_ms.p95") if at else None,
        tpot_p95_ms=at.aggregate.get("tpot_ms.p95") if at else None,
        peak_output_tok_s=peak.aggregate.get("throughput.output_tok_s") if peak else None,
        peak_load=peak.load if peak else None,
        warnings=_warnings(key, points, summaries, goodput, prices, excluded, cv_warn),
        experiment_ids=sorted({str(r.experiment_id) for r in ordered}),
        run_ids=[str(r.id) for r in ordered],
        provenance_digests=sorted({provenance_digest(r.provenance) for r in ordered}),
        git_shas=sorted({g for r in ordered if (g := _git_label(r.provenance))}),
        provenance=dict(prov),
    )


def analyze_runs(
    runs: Iterable[BenchRun],
    *,
    slo: Slo,
    allocation: CostAllocation,
    price_resolver: PriceResolver,
    confidence: float = 0.95,
    cv_warn: float = 0.10,
) -> list[ConfigResult]:
    """One ConfigResult per (config_hash, workload, load_mode), sorted by those keys."""
    completed, excluded = _group(runs)
    return [
        _analyze_sweep(
            key,
            completed[key],
            excluded.get(key, 0),
            slo=slo,
            allocation=allocation,
            price_resolver=price_resolver,
            confidence=confidence,
            cv_warn=cv_warn,
        )
        for key in sorted(completed)
    ]


_EPOCH = dt.datetime.min.replace(tzinfo=dt.UTC)


def _recency(row: BenchEvalRun | BenchGateDecision) -> tuple[dt.datetime, str]:
    return (row.created_at or _EPOCH, str(row.id))


def _latest_scores(config: str, eval_runs: Sequence[BenchEvalRun]) -> dict[str, BenchEvalRun]:
    by_task: dict[str, list[BenchEvalRun]] = defaultdict(list)
    for row in eval_runs:
        if row.config_hash == config:
            by_task[row.task].append(row)
    return {task: max(rows, key=_recency) for task, rows in by_task.items()}


def quality_for(
    config: str,
    eval_runs: Iterable[BenchEvalRun],
    gate_decisions: Iterable[BenchGateDecision],
) -> Quality | None:
    """Latest per-task scores of `config`, deltas vs its gate baseline, and the gate verdict.

    A config that is only ever a gate's baseline gets gate "baseline". None when the
    store has neither eval runs nor gate decisions for it.
    """
    evals = list(eval_runs)
    gates = list(gate_decisions)
    gate = max((g for g in gates if g.candidate_config_hash == config), key=_recency, default=None)
    is_baseline = any(g.baseline_config_hash == config for g in gates)
    scores = _latest_scores(config, evals)
    if not scores and gate is None and not is_baseline:
        return None

    baseline = gate.baseline_config_hash if gate is not None else None
    base_scores = _latest_scores(baseline, evals) if baseline is not None else {}
    tasks = []
    for task in sorted(scores):
        row = scores[task]
        base = base_scores.get(task)
        tasks.append(
            TaskQuality(
                task=task,
                task_version=row.task_version,
                n=row.n,
                score=row.score,
                ci_low=row.ci_low,
                ci_high=row.ci_high,
                baseline_score=None if base is None else base.score,
                delta=None if base is None else row.score - base.score,
            )
        )
    status: GateStatus | None = "baseline" if is_baseline else None
    divergence = None
    if gate is not None:
        status = cast(GateStatus, gate.decision)
        divergence = GateDivergence.from_details(gate.details)
    return Quality(gate=status, baseline_config_hash=baseline, tasks=tasks, divergence=divergence)


def with_quality(
    results: Iterable[ConfigResult],
    eval_runs: Iterable[BenchEvalRun],
    gate_decisions: Iterable[BenchGateDecision],
) -> list[ConfigResult]:
    evals, gates = list(eval_runs), list(gate_decisions)
    return [
        r.model_copy(update={"quality": quality_for(r.config_hash, evals, gates)}) for r in results
    ]


def cold_starts_by_config(
    cold_starts: Iterable[BenchColdStart], results: Iterable[ConfigResult]
) -> dict[str, ColdStartStat]:
    """Median cold-start time per config_hash.

    Rows recorded without a config hash are attributed only when their experiment
    benchmarked exactly one config.
    """
    configs_by_exp: dict[str, set[str]] = defaultdict(set)
    for r in results:
        for exp in r.experiment_ids:
            configs_by_exp[exp].add(r.config_hash)
    totals: dict[str, list[float]] = defaultdict(list)
    for row in cold_starts:
        if row.kind != ColdStartKind.COLD.value:
            continue
        if row.config_hash is not None:
            totals[row.config_hash].append(row.total_s)
            continue
        configs = configs_by_exp.get(str(row.experiment_id), set())
        if len(configs) == 1:
            totals[next(iter(configs))].append(row.total_s)
    return {h: ColdStartStat(median_s=median(v), n=len(v)) for h, v in sorted(totals.items())}
