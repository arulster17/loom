"""Experiment orchestration.

experiment row -> plan + cap check -> per host group: provision -> cold start ->
per cell (warm restart between configs): per workload, load points (fixed or
bisected on the SLO) x repetitions -> LoadJob -> provider.run_job -> Parquet +
summary + provenance -> quality evals -> teardown (always) -> completed.

Warmup is excluded from every summary and each load point is repeated; a single
repetition is never trusted (see `metrics.aggregate`).
"""

from __future__ import annotations

import json
import logging
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from loom_bench import __version__
from loom_bench.budget import (
    BudgetAbort,
    BudgetConfig,
    BudgetExceeded,
    BudgetGuard,
    BudgetStop,
    billable_spend,
    caps_for,
    is_billable,
)
from loom_bench.cost import cost_from_goodput
from loom_bench.experiment import (
    Cell,
    Experiment,
    LoadSpec,
    WorkloadEntry,
    derive_seed,
    expand,
    hardware_for,
    mock_config_from_launch,
    tokenizer_for,
)
from loom_bench.jobs import LoadJob, LoadJobResult
from loom_bench.metrics.aggregate import HEADLINE_METRICS, AggregateSummary, aggregate_runs
from loom_bench.metrics.gpu import parse_nvidia_smi, summarize_gpu
from loom_bench.metrics.prometheus import summarize_scrapes
from loom_bench.metrics.summary import RunSummary, flatten_metrics, summarize_run
from loom_bench.plan import Estimator, Plan, build_plan
from loom_bench.prices import PriceBook
from loom_bench.provenance import (
    DatasetInfo,
    EngineInfo,
    GitInfo,
    HardwareInfo,
    LoadgenInfo,
    LoadInfo,
    ModelInfo,
    Provenance,
    WorkloadInfo,
    build_provenance,
    canonical_json,
    config_hash,
    git_info,
)
from loom_bench.providers import make_provider, spot_interruption_errors
from loom_bench.providers.base import Endpoint, EngineLaunch, Host, HostRequest, Provider
from loom_bench.records import LoadMode, Market
from loom_bench.registry import REPO_ROOT, ModelSpec, Registry, read_yaml
from loom_bench.slo import bisect_next_load, find_goodput, slo_met
from loom_bench.store import repo
from loom_bench.store.db import session_scope
from loom_bench.store.models import BenchExperiment, ExperimentStatus, TerminatedBy
from loom_bench.store.parquet import read_requests, write_requests
from loom_bench.workloads import WorkloadProfile

log = logging.getLogger(__name__)

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_REFUSED = 3
EXIT_BUDGET_STOP = 4
EXIT_BUDGET_ABORT = 5

RUN_COMPLETED = "completed"
RUN_FAILED = "failed"
RUN_ABORTED = "aborted"
RUN_INTERRUPTED = "interrupted"


@dataclass
class RunnerContext:
    db_url: str | None
    out_dir: Path
    registry: Registry
    prices: PriceBook
    budget: BudgetConfig
    repo_dir: Path = REPO_ROOT
    provider: Provider | None = None  # default: make_provider(experiment.provider)


class PlanRefused(Exception):
    def __init__(self, plan: Plan, experiment_id: uuid.UUID | None = None) -> None:
        super().__init__("; ".join(plan.refusals))
        self.plan = plan
        self.experiment_id = experiment_id


class GoodputRow(BaseModel):
    cell: str
    config_hash: str
    workload: str
    load_mode: LoadMode
    points: list[tuple[float, bool]]
    max_load: float | None
    bracketed: bool
    trusted: bool
    output_tok_s: float | None
    hourly_micros: int
    output_per_mtok_micros: int | None
    total_per_mtok_micros: int | None


@dataclass
class Outcome:
    experiment_id: uuid.UUID
    status: ExperimentStatus
    reason: str | None
    plan: Plan
    run_ids: list[uuid.UUID]
    spent_micros: int
    goodput: list[GoodputRow]
    exit_code: int
    events: list[dict[str, Any]] = field(default_factory=list)


def _now() -> datetime:
    return datetime.now(UTC)


def plan_experiment(
    exp: Experiment, ctx: RunnerContext, *, exclude: uuid.UUID | None = None
) -> tuple[list[Cell], Plan]:
    cells = expand(exp, ctx.registry)
    with session_scope(ctx.db_url) as s:
        prior = billable_spend(s, exclude=exclude)
    caps = caps_for(exp.budget.max_spend, ctx.budget, prior)
    return cells, build_plan(exp, cells, prices=ctx.prices, caps=caps)


def _host_request(exp: Experiment, cell: Cell, experiment_id: uuid.UUID) -> HostRequest:
    hw = cell.hardware
    return HostRequest(
        cloud=hw.get("cloud"),
        region=hw.get("region"),
        instance_type=hw.get("instance_type"),
        market=Market(hw.get("market", Market.LOCAL.value)),
        gpus=cell.gpus,
        disk_gb=hw.get("disk_gb", 0),
        ttl_s=exp.budget.ttl_s,
        tags={
            "loom:experiment": str(experiment_id),
            "loom:experiment-name": exp.name,
            "loom:owner": "loom-bench",
        },
    )


def _extra_body(cell: Cell, profile: WorkloadProfile) -> dict[str, Any]:
    kwargs = cell.spec.engine.chat_template_kwargs
    if kwargs and profile.endpoint == "chat":
        return {"chat_template_kwargs": dict(kwargs)}
    return {}


class _Executor:
    """Runs the cells of one experiment; owns hosts until they are torn down."""

    def __init__(
        self,
        exp: Experiment,
        ctx: RunnerContext,
        *,
        experiment_id: uuid.UUID,
        git: GitInfo,
        guard: BudgetGuard,
        provider: Provider,
        plan: Plan,
        workloads: Sequence[tuple[WorkloadEntry, WorkloadProfile]],
        repetitions: Sequence[int],
    ) -> None:
        self.exp = exp
        self.ctx = ctx
        self.experiment_id = experiment_id
        self.git = git
        self.guard = guard
        self.provider = provider
        self.est = Estimator(exp)
        self.hourly = {h.key: h.hourly_micros for h in plan.hosts}
        self.workloads = workloads
        self.repetitions = repetitions
        self.run_dir = ctx.out_dir / str(experiment_id)
        self.live: dict[str, Host] = {}
        self.recorded: set[str] = set()
        self.done: dict[tuple[str, str, float, int], RunSummary] = {}
        self.run_ids: list[uuid.UUID] = []
        self.goodput: list[GoodputRow] = []
        self.events: list[dict[str, Any]] = []
        self.spot_errors = spot_interruption_errors()

    def event(self, kind: str, **data: Any) -> None:
        entry = {"at": _now().isoformat(), "kind": kind, **data}
        self.events.append(entry)
        with (self.run_dir / "events.jsonl").open("a", encoding="utf-8") as f:
            f.write(canonical_json(entry) + "\n")

    # --- hosts -----------------------------------------------------------------

    async def _provision(self, cell: Cell) -> Host:
        self.guard.check_next(
            self.est.cold_start_s(cell),
            extra_hourly_micros=self.hourly[cell.host_key],
            what=f"provisioning {cell.host_key}",
        )
        # Not cancellable: a half-created host must come back so it can be torn down.
        host = await self.provider.provision(_host_request(self.exp, cell, self.experiment_id))
        self.live[host.host_id] = host
        self.guard.add_host(host)
        if is_billable(host.provider) or host.provider == "mock":
            with session_scope(self.ctx.db_url) as s:
                repo.record_resource(
                    s,
                    provider=host.provider,
                    resource_type="instance",
                    resource_id=host.host_id,
                    region=host.request.region,
                    experiment_id=self.experiment_id,
                    tags=host.request.tags,
                    created_at=host.launched_at,
                    ttl_at=host.ttl_at,
                )
            self.recorded.add(host.host_id)
        self.event("provisioned", host=host.host_id, hourly_micros=host.hourly_micros)
        return host

    async def teardown(self, host: Host) -> None:
        try:
            await self.provider.teardown(host)
        except Exception:
            log.exception("teardown of %s failed; its TTL and the reaper remain", host.host_id)
            self.event("teardown_failed", host=host.host_id)
            raise
        finally:
            self.guard.remove_host(host, at=_now())
            self.live.pop(host.host_id, None)
        if host.host_id in self.recorded:
            with session_scope(self.ctx.db_url) as s:
                repo.mark_terminated(
                    s, provider=host.provider, resource_id=host.host_id, by=TerminatedBy.RUNNER
                )
        self.event("torn_down", host=host.host_id)

    async def teardown_all(self) -> None:
        for host in list(self.live.values()):
            try:
                await self.teardown(host)
            except Exception:
                continue

    # --- cells -----------------------------------------------------------------

    async def run(self, cells: Sequence[Cell]) -> None:
        groups: dict[str, list[Cell]] = {}
        for cell in cells:
            groups.setdefault(cell.host_key, []).append(cell)
        for group in groups.values():
            await self._run_group(group)

    async def _run_group(self, cells: list[Cell]) -> None:
        pending = list(cells)
        interrupted = False
        while pending:
            host = await self._provision(pending[0])
            try:
                prev: Cell | None = None
                while pending:
                    await self._run_cell(host, pending[0], prev)
                    prev = pending.pop(0)
            except self.spot_errors as e:
                self.event(
                    "spot_interruption", host=host.host_id, cell=pending[0].key, error=str(e)
                )
                if interrupted:
                    raise
                interrupted = True
                log.warning(
                    "spot interruption on %s; retrying %s on a new host",
                    host.host_id,
                    pending[0].key,
                )
            finally:
                if host.host_id in self.live:
                    await self.teardown(host)

    async def _run_cell(self, host: Host, cell: Cell, prev: Cell | None) -> None:
        warm = prev is not None
        start_s = self.est.warm_start_s(prev, cell) if warm else self.est.cold_start_s(cell)
        self.guard.check_next(start_s, what=f"starting {cell.key}")
        if warm:
            await self.guard.guarded(self.provider.stop_engine(host))
        endpoint = await self.guard.guarded(
            self.provider.start_engine(host, cell.launch, warm=warm)
        )
        stages = endpoint.start_stages
        with session_scope(self.ctx.db_url) as s:
            repo.record_cold_start(
                s,
                experiment_id=self.experiment_id,
                kind="warm" if warm else "cold",
                stages=stages,
                total_s=max(stages.values(), default=0.0),
                resource_id=host.host_id,
            )
        self.event("engine_started", host=host.host_id, cell=cell.key, warm=warm, stages=stages)
        for entry, profile in self.workloads:
            await self._run_workload(host, cell, endpoint, entry, profile)

    async def _run_workload(
        self,
        host: Host,
        cell: Cell,
        endpoint: Endpoint,
        entry: WorkloadEntry,
        profile: WorkloadProfile,
    ) -> None:
        load, slo = entry.load, self.exp.slo
        points: list[tuple[float, AggregateSummary]] = []
        history: list[tuple[float, bool]] = []

        async def point(value: float) -> None:
            agg = await self._run_point(host, cell, endpoint, entry, profile, value)
            met = agg is not None and slo is not None and slo_met(slo, agg).met
            history.append((value, met))
            if agg is not None:
                points.append((value, agg))

        if load.values is not None:
            for value in load.values:
                await point(value)
        else:
            search = load.search
            assert search is not None and slo is not None
            while len(history) < search.max_points:
                nxt = bisect_next_load(history, search.lo, search.hi, search.rel_tol)
                if nxt is None:
                    break
                if load.mode is LoadMode.CLOSED_LOOP:
                    nxt = float(max(round(nxt), 1))
                if any(v == nxt for v, _ in history):
                    break
                await point(nxt)

        if slo is None or not points:
            return
        goodput = find_goodput(slo, points, load.mode)
        cost = cost_from_goodput(host.hourly_micros, goodput, self.exp.cost_allocation.allocation())
        self.goodput.append(
            GoodputRow(
                cell=cell.key,
                config_hash=cell.config_hash,
                workload=entry.name,
                load_mode=load.mode,
                points=sorted(history),
                max_load=goodput.max_load,
                bracketed=goodput.bracketed,
                trusted=goodput.trusted,
                output_tok_s=goodput.output_tok_s.mean if goodput.output_tok_s else None,
                hourly_micros=host.hourly_micros,
                output_per_mtok_micros=cost.output_per_mtok.value if cost else None,
                total_per_mtok_micros=cost.total_per_mtok.value if cost else None,
            )
        )

    async def _run_point(
        self,
        host: Host,
        cell: Cell,
        endpoint: Endpoint,
        entry: WorkloadEntry,
        profile: WorkloadProfile,
        value: float,
    ) -> AggregateSummary | None:
        summaries = []
        for rep in self.repetitions:
            key = (cell.key, entry.name, value, rep)
            summary = self.done.get(key)
            if summary is None:
                summary = await self._run_once(host, cell, endpoint, entry, profile, value, rep)
            if summary is not None:
                self.done[key] = summary
                summaries.append(summary)
        return aggregate_runs(summaries) if summaries else None

    def _job(
        self,
        cell: Cell,
        endpoint: Endpoint,
        load: LoadSpec,
        profile: WorkloadProfile,
        value: float,
        seed: int,
    ) -> LoadJob:
        return LoadJob(
            run_id=str(uuid.uuid4()),
            base_url=endpoint.base_url,
            metrics_url=endpoint.metrics_url,
            engine=endpoint.engine,
            served_model=endpoint.served_model,
            loadgen=self.exp.loadgen,
            workload=profile.model_dump(mode="json"),
            tokenizer=cell.tokenizer,
            mode=load.mode,
            load_value=value,
            arrival=load.arrival_for(value) if load.mode is LoadMode.OPEN_LOOP else None,
            duration_s=load.duration_s,
            num_requests=load.num_requests,
            warmup_s=load.warmup_s or 0.0,
            warmup_requests=load.warmup_requests or 0,
            request_timeout_s=load.request_timeout_s,
            drain_timeout_s=load.drain_timeout_s,
            max_inflight=load.max_inflight,
            seed=seed,
            scrape_interval_s=load.scrape_interval_s,
            sample_gpu=self.exp.provider.kind == "aws_ec2",
            extra_body=_extra_body(cell, profile),
        )

    def provenance(
        self,
        host: Host,
        cell: Cell,
        endpoint: Endpoint,
        entry: WorkloadEntry,
        profile: WorkloadProfile,
        mode: LoadMode,
        value: float,
        seed: int,
        rep: int,
    ) -> Provenance:
        system = endpoint.system
        launch, hw = cell.launch, cell.hardware
        image = launch.image or None
        digest = system.get("image_digest") or (
            image.split("@", 1)[1] if image and "@" in image else None
        )
        engine = system.get("engine") or endpoint.engine
        version = system.get("engine_version") or (
            cell.spec.engine.version if engine == cell.spec.engine.name else None
        )
        # Real instance types fix the GPU; local endpoints report nothing we can trust.
        gpu_type = system.get("gpu_name") or (
            cell.spec.hardware.gpu if hw.get("instance_type") else None
        )
        dataset = DatasetInfo(**profile.dataset.model_dump()) if profile.dataset else DatasetInfo()
        return build_provenance(
            cell.config,
            git=self.git,
            loadgen=LoadgenInfo(
                name=self.exp.loadgen, version=__version__ if self.exp.loadgen == "native" else None
            ),
            engine=EngineInfo(
                name=engine,
                version=version,
                image=image,
                image_digest=digest,
                args={"argv": launch.args},
            ),
            cuda_version=system.get("cuda_version"),
            driver_version=system.get("driver_version"),
            model=ModelInfo(
                repo=cell.spec.hf.repo,
                revision=cell.spec.hf.revision,
                quantization=cell.spec.quantization,
            ),
            hardware=HardwareInfo(
                gpu_type=gpu_type,
                gpu_count=system.get("gpu_count") or cell.gpus,
                instance_type=hw.get("instance_type"),
            ),
            cloud=hw.get("cloud"),
            region=hw.get("region"),
            market=host.request.market,
            workload=WorkloadInfo(
                name=entry.name, profile_hash=config_hash(profile), content=profile.content
            ),
            dataset=dataset,
            load=LoadInfo(mode=mode, value=value, seed=seed),
            repetition=rep,
        )

    def _summarize(self, cell: Cell, endpoint: Endpoint, result: LoadJobResult) -> RunSummary:
        server = summarize_scrapes(endpoint.engine, result.scrapes) if result.scrapes else None
        gpu = (
            summarize_gpu(parse_nvidia_smi(result.nvidia_smi_csv))
            if result.nvidia_smi_csv
            else None
        )
        window = result.t_measure_end_s - result.t_measure_start_s
        return summarize_run(
            result.request_records(),
            window_s=window if window > 0 else None,
            gpus=cell.gpus,
            slo=self.exp.slo,
            server=server,
            gpu=gpu,
        )

    def _record(
        self,
        *,
        cell: Cell,
        entry: WorkloadEntry,
        load: LoadSpec,
        value: float,
        rep: int,
        prov: Provenance,
        status: str,
        started_at: datetime,
        summary: dict[str, Any] | None = None,
        result: LoadJobResult | None = None,
    ) -> uuid.UUID:
        with session_scope(self.ctx.db_url) as s:
            run = repo.record_run(
                s,
                experiment_id=self.experiment_id,
                config_hash=cell.config_hash,
                provenance=prov,
                status=status,
                summary=summary,
                cell_key=cell.key,
                workload=entry.name,
                load_mode=load.mode,
                load_value=value,
                repetition=rep,
                started_at=started_at,
                finished_at=_now(),
            )
            run_dir = self.run_dir / "runs" / str(run.id)
            run_dir.mkdir(parents=True, exist_ok=True)
            if result is not None:
                run.requests_uri = write_requests(
                    result.request_records(), run_dir / "requests.parquet"
                )
                s.flush()
            (run_dir / "provenance.json").write_text(prov.model_dump_json(indent=2))
        self.run_ids.append(run.id)
        return run.id

    async def _run_once(
        self,
        host: Host,
        cell: Cell,
        endpoint: Endpoint,
        entry: WorkloadEntry,
        profile: WorkloadProfile,
        value: float,
        rep: int,
    ) -> RunSummary | None:
        load = entry.load
        label = f"{cell.key} / {entry.name} @ {value:g} rep {rep}"
        self.guard.check_next(self.est.run_s(cell, load, profile, value), what=f"run {label}")
        seed = derive_seed(self.exp.seed, entry.name, value, rep)
        job = self._job(cell, endpoint, load, profile, value, seed)
        prov = self.provenance(host, cell, endpoint, entry, profile, load.mode, value, seed, rep)
        common = dict(cell=cell, entry=entry, load=load, value=value, rep=rep, prov=prov)
        started_at = _now()
        try:
            result = await self.guard.guarded(self.provider.run_job(host, job))
        except BudgetExceeded:
            self._record(**common, status=RUN_ABORTED, started_at=started_at)
            raise
        except self.spot_errors:
            self._record(**common, status=RUN_INTERRUPTED, started_at=started_at)
            raise
        except Exception as e:
            log.exception("run %s failed", label)
            self._record(
                **common,
                status=RUN_FAILED,
                started_at=started_at,
                summary={"error": f"{type(e).__name__}: {e}"},
            )
            return None
        try:
            summary = self._summarize(cell, endpoint, result)
        except ValueError as e:
            self._record(
                **common,
                status=RUN_FAILED,
                started_at=started_at,
                summary={"error": str(e), "client": result.meta},
                result=result,
            )
            return None
        doc = {
            **summary.model_dump(mode="json"),
            "client": {**result.meta, "client_saturated_count": result.client_saturated_count},
            "cost_basis": {
                "hourly_micros": host.hourly_micros,
                "market": host.request.market.value,
            },
        }
        self._record(
            **common, status=RUN_COMPLETED, started_at=started_at, summary=doc, result=result
        )
        log.info(
            "%s: p95 TTFT %s ms, %.1f out tok/s",
            label,
            _fmt(summary.ttft_ms.p95),
            summary.throughput.output_tok_s,
        )
        return summary


def _fmt(v: float | None) -> str:
    return "-" if v is None else f"{v:.1f}"


def _finish(
    ctx: RunnerContext,
    experiment_id: uuid.UUID,
    status: ExperimentStatus,
    reason: str | None = None,
) -> int:
    with session_scope(ctx.db_url) as s:
        exp = repo.update_experiment_status(s, experiment_id, status, abort_reason=reason)
        return exp.spent_micros


async def run_experiment(exp: Experiment, ctx: RunnerContext) -> Outcome:
    """Run `exp` end to end. Raises PlanRefused before anything is provisioned."""
    git = git_info(ctx.repo_dir)
    with session_scope(ctx.db_url) as s:
        row = repo.create_experiment(
            s,
            name=exp.name,
            spec=exp.model_dump(mode="json"),
            git=git,
            budget_micros=exp.budget.max_spend,
        )
        experiment_id = row.id
    cells, plan = plan_experiment(exp, ctx, exclude=experiment_id)
    if not plan.ok:
        _finish(
            ctx,
            experiment_id,
            ExperimentStatus.ABORTED,
            "refused by planner: " + "; ".join(plan.refusals),
        )
        raise PlanRefused(plan, experiment_id)
    workloads = [(w, w.resolve()) for w in exp.workloads]
    return await _execute(
        exp,
        ctx,
        experiment_id=experiment_id,
        git=git,
        cells=cells,
        plan=plan,
        workloads=workloads,
        repetitions=range(exp.repetitions),
    )


async def _execute(
    exp: Experiment,
    ctx: RunnerContext,
    *,
    experiment_id: uuid.UUID,
    git: GitInfo,
    cells: Sequence[Cell],
    plan: Plan,
    workloads: Sequence[tuple[WorkloadEntry, WorkloadProfile]],
    repetitions: Sequence[int],
) -> Outcome:
    run_dir = ctx.out_dir / str(experiment_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "spec.json").write_text(json.dumps(exp.model_dump(mode="json"), indent=2))
    with session_scope(ctx.db_url) as s:
        repo.update_experiment_status(s, experiment_id, ExperimentStatus.RUNNING)
    guard = BudgetGuard(
        db_url=ctx.db_url,
        experiment_id=experiment_id,
        cap_micros=plan.caps.effective,
        overall_cap_micros=plan.caps.overall,
        billable=is_billable(exp.provider.kind),
        interval_s=exp.budget.accrual_interval_s,
    )
    ex = _Executor(
        exp,
        ctx,
        experiment_id=experiment_id,
        git=git,
        guard=guard,
        provider=ctx.provider or make_provider(exp.provider),
        plan=plan,
        workloads=workloads,
        repetitions=repetitions,
    )
    status, reason, code = ExperimentStatus.COMPLETED, None, EXIT_OK
    interrupted: BaseException | None = None
    guard.start()
    try:
        await ex.run(cells)
    except BudgetStop as e:
        status, reason, code = ExperimentStatus.ABORTED, e.reason, EXIT_BUDGET_STOP
    except BudgetAbort as e:
        status, reason, code = ExperimentStatus.ABORTED, e.reason, EXIT_BUDGET_ABORT
    except Exception as e:
        log.exception("experiment %s failed", exp.name)
        status, reason, code = ExperimentStatus.FAILED, f"{type(e).__name__}: {e}", EXIT_FAILED
    except BaseException as e:  # Ctrl-C or cancellation: clean up, record, re-raise
        status, reason, code = (
            ExperimentStatus.FAILED,
            f"interrupted: {type(e).__name__}",
            EXIT_FAILED,
        )
        interrupted = e
    finally:
        await ex.teardown_all()
        await guard.stop()
    if guard.tripped.is_set() and status is ExperimentStatus.COMPLETED:
        status, reason, code = ExperimentStatus.ABORTED, guard.reason, EXIT_BUDGET_ABORT
    spent = _finish(ctx, experiment_id, status, reason)
    if interrupted is not None:
        raise interrupted
    (run_dir / "goodput.json").write_text(
        json.dumps([g.model_dump(mode="json") for g in ex.goodput], indent=2)
    )
    return Outcome(
        experiment_id=experiment_id,
        status=status,
        reason=reason,
        plan=plan,
        run_ids=ex.run_ids,
        spent_micros=spent,
        goodput=ex.goodput,
        exit_code=code,
        events=ex.events,
    )


# --- reproduce -------------------------------------------------------------------


class MetricComparison(BaseModel):
    metric: str
    original: float | None
    new: float | None
    rel_diff: float | None
    sibling_lo: float | None  # range across the original's repetitions of the same point
    sibling_hi: float | None
    within: bool


@dataclass
class ReproduceOutcome:
    original_run_id: uuid.UUID | None
    new_run_id: uuid.UUID | None
    outcome: Outcome
    warnings: list[str]
    comparisons: list[MetricComparison]

    @property
    def ok(self) -> bool:
        return (
            self.outcome.status is ExperimentStatus.COMPLETED
            and bool(self.comparisons)
            and all(c.within for c in self.comparisons)
        )


def compare_summaries(
    original: RunSummary,
    new: RunSummary,
    siblings: Sequence[RunSummary] = (),
    *,
    tolerance: float = 0.25,
    metrics: Sequence[str] = HEADLINE_METRICS,
) -> list[MetricComparison]:
    """Per-metric relative difference. A metric is within normal variance when it is
    within `tolerance` of the original or inside the range of the original's
    repetitions at the same load point."""
    a, b = flatten_metrics(original), flatten_metrics(new)
    sib = [flatten_metrics(s) for s in siblings]
    out = []
    for m in metrics:
        va, vb = a.get(m), b.get(m)
        values = [s[m] for s in sib if m in s]
        lo, hi = (min(values), max(values)) if values else (None, None)
        rel = None if va is None or vb is None or va == 0 else abs(vb - va) / abs(va)
        in_range = lo is not None and hi is not None and vb is not None and lo <= vb <= hi
        within = va == vb or (rel is not None and rel <= tolerance) or in_range
        out.append(
            MetricComparison(
                metric=m,
                original=va,
                new=vb,
                rel_diff=rel,
                sibling_lo=lo,
                sibling_hi=hi,
                within=within,
            )
        )
    return out


def cell_from_provenance(exp: Experiment, prov: Provenance, key: str) -> Cell:
    """The exact cell a run used, rebuilt from its stored resolved config."""
    cfg = prov.config
    spec = ModelSpec.model_validate(cfg["model"])
    launch = EngineLaunch.model_validate(cfg["launch"])
    mock = mock_config_from_launch(launch) if launch.engine == "mock" else None
    host_key, hardware = hardware_for(exp, spec)
    if hardware != cfg["hardware"]:
        raise ValueError(
            f"stored hardware {cfg['hardware']} does not match the experiment's provider "
            f"({hardware})"
        )
    cell = Cell(
        key=key,
        variant=key.split("[", 1)[0],
        knobs={},
        spec=spec,
        mock=mock,
        launch=launch,
        host_key=host_key,
        hardware=hardware,
        tokenizer=tokenizer_for(exp, spec),
        config=cfg,
        config_hash=config_hash(cfg),
    )
    if cell.config_hash != prov.config_hash:
        raise ValueError("stored config does not hash to the recorded config_hash")
    return cell


def _load_spec_doc(path: Path) -> dict[str, Any]:
    if path.suffix in (".yaml", ".yml"):
        return read_yaml(path)
    return json.loads(path.read_text())


@dataclass
class _Original:
    prov: Provenance
    spec_doc: dict[str, Any]
    run_id: uuid.UUID | None = None
    cell_key: str | None = None
    summary: RunSummary | None = None
    siblings: list[RunSummary] = field(default_factory=list)


def _original_from_file(path: Path, spec_path: Path | None) -> _Original:
    prov = Provenance.model_validate_json(path.read_text())
    orig = _Original(prov=prov, spec_doc=_load_spec_doc(spec_path or path.parents[2] / "spec.json"))
    parquet = path.parent / "requests.parquet"
    if parquet.is_file():
        orig.summary = summarize_run(read_requests(parquet), window_s=None, gpus=1)
    return orig


def _original_from_db(db_url: str | None, run_id: uuid.UUID, spec_path: Path | None) -> _Original:
    with session_scope(db_url) as s:
        run = repo.get_run(s, run_id)
        if run is None:
            raise LookupError(f"no run {run_id}")
        exp_row = s.get(BenchExperiment, run.experiment_id)
        assert exp_row is not None
        orig = _Original(
            prov=Provenance.model_validate(run.provenance),
            spec_doc=_load_spec_doc(spec_path) if spec_path else exp_row.spec,
            run_id=run_id,
            cell_key=run.cell_key,
        )
        if run.status == RUN_COMPLETED and run.summary:
            orig.summary = RunSummary.model_validate(run.summary)
        for sib in repo.list_runs(s, experiment_id=run.experiment_id, config_hash=run.config_hash):
            same_point = sib.workload == run.workload and sib.load_value == run.load_value
            if same_point and sib.status == RUN_COMPLETED and sib.summary:
                orig.siblings.append(RunSummary.model_validate(sib.summary))
    return orig


def _load_original(ref: str, db_url: str | None, spec_path: Path | None) -> _Original:
    path = Path(ref)
    if path.suffix == ".json" and path.is_file():
        return _original_from_file(path, spec_path)
    return _original_from_db(db_url, uuid.UUID(ref), spec_path)


async def reproduce(
    ref: str,
    ctx: RunnerContext,
    *,
    spec_path: Path | None = None,
    tolerance: float = 0.25,
) -> ReproduceOutcome:
    """Re-run one stored run from its provenance and compare it with the original.

    `ref` is a run id in the results DB or a run's `provenance.json`, whose
    experiment spec is read from `<results>/<experiment>/spec.json` unless
    `spec_path` is given.
    """
    orig = _load_original(ref, ctx.db_url, spec_path)
    prov = orig.prov
    exp = Experiment.model_validate(orig.spec_doc)

    warnings: list[str] = []
    current = git_info(ctx.repo_dir)
    if prov.git.sha != current.sha:
        warnings.append(f"git sha differs: original {prov.git.sha}, now {current.sha}")
    if current.dirty:
        warnings.append("working tree is dirty: uncommitted changes may change results")
    if prov.git.dirty:
        warnings.append("the original run was made from a dirty working tree")

    cell = cell_from_provenance(exp, prov, orig.cell_key or f"reproduce-{prov.config_hash[:12]}")
    entry = next((w for w in exp.workloads if w.name == prov.workload.name), None)
    if entry is None:
        raise LookupError(f"workload {prov.workload.name!r} is not in the experiment spec")
    profile = entry.resolve()
    if config_hash(profile) != prov.workload.profile_hash:
        warnings.append(
            f"workload {entry.name} resolves differently now (profile hash changed); "
            "requests will not match the original"
        )
    if prov.load.value is None or prov.repetition is None:
        raise ValueError("provenance lacks the load value or repetition")
    value, rep = prov.load.value, prov.repetition
    if derive_seed(exp.seed, entry.name, value, rep) != prov.load.seed:
        raise ValueError("the spec derives a different seed than the original run used")

    entry_one = entry.model_copy(
        update={"load": entry.load.model_copy(update={"values": [value], "search": None})}
    )
    exp_one = exp.model_copy(update={"workloads": [entry_one], "repetitions": 1, "quality": None})
    with session_scope(ctx.db_url) as s:
        prior = billable_spend(s)
    caps = caps_for(exp.budget.max_spend, ctx.budget, prior)
    plan = build_plan(exp_one, [cell], prices=ctx.prices, caps=caps)
    if not plan.ok:
        raise PlanRefused(plan)
    with session_scope(ctx.db_url) as s:
        experiment_id = repo.create_experiment(
            s,
            name=f"{exp.name}--reproduce",
            spec=orig.spec_doc,
            git=current,
            budget_micros=exp.budget.max_spend,
        ).id
    outcome = await _execute(
        exp_one,
        ctx,
        experiment_id=experiment_id,
        git=current,
        cells=[cell],
        plan=plan,
        workloads=[(entry_one, profile)],
        repetitions=[rep],
    )
    new_id = outcome.run_ids[0] if outcome.run_ids else None
    comparisons: list[MetricComparison] = []
    if new_id is not None and orig.summary is not None:
        with session_scope(ctx.db_url) as s:
            new_run = repo.get_run(s, new_id)
            new_doc = new_run.summary if new_run and new_run.status == RUN_COMPLETED else None
        if new_doc:
            comparisons = compare_summaries(
                orig.summary,
                RunSummary.model_validate(new_doc),
                orig.siblings,
                tolerance=tolerance,
            )
    return ReproduceOutcome(
        original_run_id=orig.run_id,
        new_run_id=new_id,
        outcome=outcome,
        warnings=warnings,
        comparisons=comparisons,
    )


# --- reaper ------------------------------------------------------------------------


class ReapedResource(BaseModel):
    provider: str
    resource_id: str
    experiment_id: uuid.UUID | None
    ttl_at: datetime
    action: str  # terminated | gone | would terminate


async def reap(
    db_url: str | None, *, now: datetime | None = None, dry_run: bool = False
) -> list[ReapedResource]:
    """Terminate DB-recorded resources whose TTL passed and mark them reaped.

    Mock servers die with the process that started them, so an expired mock
    resource not running in this process is already gone and is only marked.
    """
    from loom_bench.providers.mock import MockProvider

    now = now or _now()
    with session_scope(db_url) as s:
        expired = [
            (r.provider, r.resource_id, r.experiment_id, r.ttl_at)
            for r in repo.list_expired_resources(s, now)
        ]
    killed = set() if dry_run else set(await MockProvider().reap(now))
    out: list[ReapedResource] = []
    for provider, resource_id, experiment_id, ttl_at in expired:
        if dry_run:
            action = "would terminate"
        else:
            action = "terminated" if resource_id in killed else "gone"
            with session_scope(db_url) as s:
                repo.mark_terminated(
                    s, provider=provider, resource_id=resource_id, by=TerminatedBy.REAPER, at=now
                )
        out.append(
            ReapedResource(
                provider=provider,
                resource_id=resource_id,
                experiment_id=experiment_id,
                ttl_at=ttl_at,
                action=action,
            )
        )
    return out
