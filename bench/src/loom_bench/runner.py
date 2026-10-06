"""Experiment orchestration.

experiment row -> plan + cap check -> per host group: provision -> cold start ->
per cell (warm restart between configs): per workload, load points (fixed or
bisected on the SLO) x repetitions -> LoadJob -> provider.run_job -> Parquet +
summary + provenance -> EvalJob -> provider.run_eval (the baseline variant
captures the divergence reference; every other cell is scored against it and
gated against the baseline) -> teardown (always) -> completed.

Warmup is excluded from every summary and each load point is repeated; a single
repetition is never trusted (see `metrics.aggregate`).
"""

from __future__ import annotations

import asyncio
import functools
import json
import logging
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session

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
from loom_bench.cost import PriceColumn, ReplicaPrices
from loom_bench.experiment import (
    Cell,
    Experiment,
    LoadSpec,
    WorkloadEntry,
    derive_seed,
    expand,
    hardware_for,
    load_quality_suite,
    mock_config_from_launch,
    tokenizer_for,
)
from loom_bench.jobs import DivergenceMode, EvalJob, LoadJob, LoadJobResult
from loom_bench.metrics.aggregate import AggregateSummary, aggregate_runs
from loom_bench.metrics.gpu import parse_nvidia_smi, summarize_gpu
from loom_bench.metrics.prometheus import summarize_scrapes
from loom_bench.metrics.summary import RunSummary, summarize_run
from loom_bench.mock.config import MockConfig
from loom_bench.plan import EVAL_CONCURRENCY, Estimator, Plan, build_plan
from loom_bench.prices import PriceBook, UnverifiedPriceError
from loom_bench.provenance import (
    ContentKind,
    DatasetInfo,
    EngineInfo,
    GitInfo,
    HardwareInfo,
    LoadgenInfo,
    LoadInfo,
    ModelInfo,
    ParallelismInfo,
    Provenance,
    WorkloadInfo,
    build_provenance,
    canonical_json,
    config_hash,
    git_info,
    to_jsonable,
)
from loom_bench.providers import make_provider
from loom_bench.providers.base import (
    Endpoint,
    EngineLaunch,
    Host,
    HostLost,
    HostRequest,
    Provider,
    SpotInterrupted,
)
from loom_bench.quality.divergence import DivergenceResult, ReferenceLogprobs
from loom_bench.quality.gate import GateDecision
from loom_bench.quality.runner import (
    SuiteResult,
    TaskRun,
    gate_against_baseline,
    record_gate,
    record_suite_result,
    suite_result_of,
)
from loom_bench.quality.sanity import SanityResult
from loom_bench.quality.suite import Suite
from loom_bench.quality.tasks.base import ItemResult
from loom_bench.records import LoadMode, Market
from loom_bench.registry import REPO_ROOT, ModelSpec, Registry, read_yaml
from loom_bench.report.analyze import costs_at, default_price_resolver
from loom_bench.report.compare import Comparison, compare
from loom_bench.slo import bisect_next_load, find_goodput, slo_met
from loom_bench.stats import mean_ci
from loom_bench.store import repo
from loom_bench.store.db import session_scope
from loom_bench.store.models import (
    BenchEvalRun,
    BenchExperiment,
    BenchRun,
    ExperimentStatus,
    TerminatedBy,
)
from loom_bench.store.parquet import write_requests
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
RUN_INTERRUPTED = "interrupted"  # spot reclaim
RUN_HOST_LOST = "host_lost"

RESOURCE_TYPES = {"aws_ec2": "ec2_instance", "mock": "mock_server"}


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


class GoodputCost(BaseModel):
    hourly_micros: int | None
    output_per_mtok_micros: int | None
    total_per_mtok_micros: int | None


class GoodputRow(BaseModel):
    """One sweep's goodput, priced per price column exactly as `bench report` prices it
    (`default_price_resolver`); `price_error` says why a column has no price."""

    cell: str
    config_hash: str
    workload: str
    load_mode: LoadMode
    points: list[tuple[float, bool]]
    max_load: float | None
    bracketed: bool
    trusted: bool
    output_tok_s: float | None
    costs: dict[PriceColumn, GoodputCost]
    price_error: str | None


@dataclass
class _Baseline:
    cell: Cell
    result: SuiteResult
    reference: ReferenceLogprobs | None  # None when the suite has no divergence section


class GateRow(BaseModel):
    cell: str
    baseline: str
    decision: str  # pass | review | fail | inconclusive
    blocked: bool


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
    gates: list[GateRow] = field(default_factory=list)


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
        image=cell.launch.image or None,
        ttl_s=exp.budget.ttl_s,
        tags={
            "loom:experiment": str(experiment_id),
            "loom:experiment-name": exp.name,
        },
    )


def _extra_body(cell: Cell, profile: WorkloadProfile) -> dict[str, Any]:
    kwargs = cell.spec.engine.chat_template_kwargs
    if kwargs and profile.endpoint == "chat":
        return {"chat_template_kwargs": dict(kwargs)}
    return {}


def _engine_args(cell: Cell) -> dict[str, Any]:
    """Engine args as configured; for the mock, its settings that differ from defaults."""
    if cell.mock is None:
        return dict(cell.spec.engine.args)
    defaults = MockConfig().model_dump()
    return {k: v for k, v in cell.mock.model_dump().items() if k != "models" and v != defaults[k]}


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
        self.suite = exp.quality.load() if exp.quality is not None else None
        self.baselines: dict[str, _Baseline] = {}
        self.eval_hosts: set[str] = set()  # hosts whose client env has the eval harness
        self.gates: list[GateRow] = []

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
        if host.provider in RESOURCE_TYPES:  # local endpoints are not ours to reap
            with session_scope(self.ctx.db_url) as s:
                repo.record_resource(
                    s,
                    provider=host.provider,
                    resource_type=RESOURCE_TYPES[host.provider],
                    resource_id=host.host_id,
                    region=host.request.region,
                    experiment_id=self.experiment_id,
                    tags=host.info.get("tags", host.request.tags),
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
        if self.exp.quality is not None:  # baselines are evaluated before their candidates
            base = self.exp.quality.baseline_variant
            cells = sorted(cells, key=lambda c: c.variant != base)
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
            except SpotInterrupted as e:
                self.event(
                    "spot_interruption",
                    host=host.host_id,
                    cell=pending[0].key,
                    reason_code=e.reason_code,
                    seconds_since_launch=e.seconds_since_launch,
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
                config_hash=cell.config_hash,
            )
        self.event("engine_started", host=host.host_id, cell=cell.key, warm=warm, stages=stages)
        for entry, profile in self.workloads:
            await self._run_workload(host, cell, endpoint, entry, profile)
        if self.suite is not None:
            await self._run_quality(host, cell, endpoint, self.suite)

    def _eval_job(self, cell: Cell, endpoint: Endpoint, suite: Suite) -> EvalJob:
        quality = self.exp.quality
        assert quality is not None
        divergence: DivergenceMode | None = None
        reference = None
        if suite.divergence is not None:
            if cell.variant == quality.baseline_variant:
                divergence = "capture_and_floor"
            else:
                divergence, reference = "score", self._baseline(cell).reference
        return EvalJob(
            run_id=str(uuid.uuid4()),
            suite=suite,
            tasks=quality.task_names(suite),
            base_url=endpoint.base_url,
            served_model=endpoint.served_model,
            extra_body=suite.extra_body(),
            allow_code_exec=quality.allow_code_exec,
            tokenizer=cell.tokenizer,
            seed=suite.seed,
            concurrency=EVAL_CONCURRENCY,
            divergence=divergence,
            reference=reference,
        )

    def _baseline(self, cell: Cell) -> _Baseline:
        base = self.baselines.get(canonical_json(cell.knobs))
        if base is None:
            raise RuntimeError(f"{cell.key}: its baseline cell has no quality results to gate on")
        return base

    async def _run_quality(self, host: Host, cell: Cell, endpoint: Endpoint, suite: Suite) -> None:
        quality = self.exp.quality
        assert quality is not None
        is_baseline = cell.variant == quality.baseline_variant
        job = self._eval_job(cell, endpoint, suite)
        setup = 0.0 if host.host_id in self.eval_hosts else self.est.eval_setup_s()
        self.guard.check_next(self.est.eval_s(cell) + setup, what=f"quality suite on {cell.key}")
        out = await self.guard.guarded(self.provider.run_eval(host, job))
        self.eval_hosts.add(host.host_id)
        result = suite_result_of(out)
        evals_dir = self.run_dir / "evals" / cell.config_hash[:16]
        samples = write_samples(
            evals_dir / "samples.json",
            quality.suite,
            result,
            divergence=out.divergence,
            reference_config_hash=job.reference.config_hash if job.reference else None,
            self_divergence=out.reference.self_divergence if out.reference else None,
        )
        prov = build_provenance(cell.config, **self._serving_sections(host, cell, endpoint))
        with session_scope(self.ctx.db_url) as s:
            record_suite_result(
                s,
                experiment_id=self.experiment_id,
                config_hash=cell.config_hash,
                result=result,
                provenance=prov,
                samples_uri=str(samples),
            )
        scores = {name: run.estimate.mean for name, run in result.tasks.items()}
        seconds = {name: round(run.seconds, 1) for name, run in result.tasks.items()}
        self.event(
            "quality",
            cell=cell.key,
            suite=result.suite,
            eval_job=job.run_id,
            scores=scores,
            seconds=seconds,
        )
        if is_baseline:
            reference = None
            if out.reference is not None:
                reference = out.reference.model_copy(
                    update={"config_hash": cell.config_hash, "provenance": to_jsonable(prov)}
                )
                path = evals_dir / "reference.json"
                path.write_text(reference.model_dump_json())
                floor = reference.self_divergence
                self.event(
                    "reference_captured",
                    cell=cell.key,
                    config_hash=cell.config_hash,
                    prompts=len(reference.prompts),
                    path=str(path),
                    self_divergence=None if floor is None else floor.model_dump(mode="json"),
                )
            self.baselines[canonical_json(cell.knobs)] = _Baseline(cell, result, reference)
            return
        base = self._baseline(cell)
        decision = gate_against_baseline(
            base.result,
            result,
            suite,
            out.divergence,
            self_divergence=base.reference.self_divergence if base.reference else None,
        )
        with session_scope(self.ctx.db_url) as s:
            record_gate(
                s,
                experiment_id=self.experiment_id,
                baseline_config_hash=base.cell.config_hash,
                candidate_config_hash=cell.config_hash,
                decision=decision,
            )
        row = GateRow(
            cell=cell.key,
            baseline=base.cell.key,
            decision=decision.decision.value,
            blocked=decision.blocked,
        )
        self.gates.append(row)
        self.event("gate", **row.model_dump(), reasons=decision.reasons)

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
        prov = build_provenance(cell.config, **self._serving_sections(host, cell, endpoint))
        prices: ReplicaPrices | None = None
        try:
            prices = default_price_resolver(self.ctx.prices)(prov.model_dump(mode="json"))
            error = prices.missing
        except (KeyError, UnverifiedPriceError) as e:
            error = str(e)
        allocation = self.exp.cost_allocation.allocation()
        costs = {}
        for column in PriceColumn:
            hourly = prices.get(column) if prices else None
            cost = costs_at(hourly, goodput, allocation)
            costs[column] = GoodputCost(
                hourly_micros=hourly,
                output_per_mtok_micros=cost.output_per_mtok.value if cost else None,
                total_per_mtok_micros=cost.total_per_mtok.value if cost else None,
            )
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
                costs=costs,
                price_error=error,
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
        dataset = DatasetInfo(**profile.dataset.model_dump()) if profile.dataset else DatasetInfo()
        return build_provenance(
            cell.config,
            **self._serving_sections(host, cell, endpoint),
            loadgen=LoadgenInfo(
                name=self.exp.loadgen, version=__version__ if self.exp.loadgen == "native" else None
            ),
            workload=WorkloadInfo(
                name=entry.name,
                profile_hash=config_hash(profile),
                content=ContentKind(profile.content),
            ),
            dataset=dataset,
            load=LoadInfo(mode=mode, value=value, seed=seed),
            repetition=rep,
        )

    def _serving_sections(self, host: Host, cell: Cell, endpoint: Endpoint) -> dict[str, Any]:
        """Provenance sections describing what served: git, engine, model, hardware, and
        the as-run price with its basis (never the budget accrual rate)."""
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
        return dict(
            git=self.git,
            engine=EngineInfo(
                name=engine,
                version=version,
                image=image,
                image_digest=digest,
                args=_engine_args(cell),
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
            parallelism=ParallelismInfo(
                tp=cell.spec.parallelism.tp,
                pp=cell.spec.parallelism.pp,
                ep=cell.spec.parallelism.ep,
            ),
            cloud=hw.get("cloud"),
            region=hw.get("region"),
            market=host.request.market,
            hourly_micros=host.as_run_micros,
            price_basis=host.price_basis,
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
        run_id: uuid.UUID,
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
    ) -> None:
        run_dir = self.run_dir / "runs" / str(run_id)
        run_dir.mkdir(parents=True, exist_ok=True)
        requests_uri = (
            write_requests(result.request_records(), run_dir / "requests.parquet")
            if result is not None
            else None
        )
        (run_dir / "provenance.json").write_text(prov.model_dump_json(indent=2))
        with session_scope(self.ctx.db_url) as s:
            repo.record_run(
                s,
                run_id=run_id,
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
                requests_uri=requests_uri,
                started_at=started_at,
                finished_at=_now(),
            )
        self.run_ids.append(run_id)

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
        started_at = _now()
        record = functools.partial(
            self._record,
            run_id=uuid.UUID(job.run_id),
            cell=cell,
            entry=entry,
            load=load,
            value=value,
            rep=rep,
            prov=prov,
            started_at=started_at,
        )
        try:
            result = await self.guard.guarded(self.provider.run_job(host, job))
        except BudgetExceeded:
            record(status=RUN_ABORTED)
            raise
        except HostLost as e:
            status = RUN_INTERRUPTED if isinstance(e, SpotInterrupted) else RUN_HOST_LOST
            record(status=status, summary={"error": str(e)})
            raise
        except Exception as e:
            log.exception("run %s failed", label)
            record(status=RUN_FAILED, summary={"error": f"{type(e).__name__}: {e}"})
            return None
        try:
            summary = self._summarize(cell, endpoint, result)
        except ValueError as e:
            record(
                status=RUN_FAILED, summary={"error": str(e), "client": result.meta}, result=result
            )
            return None
        doc = {
            **summary.model_dump(mode="json"),
            "client": {
                **result.meta,
                "timeline": result.timeline,
                "client_saturated_count": result.client_saturated_count,
            },
        }
        record(status=RUN_COMPLETED, summary=doc, result=result)
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
    try:
        provider = ctx.provider or make_provider(exp.provider, prices=ctx.prices, work_dir=run_dir)
    except Exception as e:
        _finish(ctx, experiment_id, ExperimentStatus.FAILED, f"provider setup: {e}")
        raise
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
        provider=provider,
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
        gates=ex.gates,
    )


# --- quality -----------------------------------------------------------------------


def write_samples(
    path: Path,
    suite_ref: str,
    result: SuiteResult,
    *,
    divergence: DivergenceResult | None = None,
    reference_config_hash: str | None = None,
    self_divergence: DivergenceResult | None = None,
) -> Path:
    """Per-item scores of one suite run, so a gate can be re-decided later, with the
    candidate's divergence (from the reference captured on `reference_config_hash`) or
    the baseline's self-divergence."""
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = {
        "suite": result.suite,
        "suite_ref": suite_ref,
        "model": result.model,
        "started_at": result.started_at.isoformat(),
        "finished_at": result.finished_at.isoformat(),
        "sanity": result.sanity.model_dump(mode="json"),
        "divergence": None if divergence is None else divergence.model_dump(mode="json"),
        "divergence_reference": reference_config_hash,
        "self_divergence": (
            None if self_divergence is None else self_divergence.model_dump(mode="json")
        ),
        "tasks": {
            name: {
                "kind": run.kind,
                "version": run.version,
                "provenance": run.provenance,
                "seconds": run.seconds,
                "items": [i.model_dump(mode="json") for i in run.items],
            }
            for name, run in result.tasks.items()
        },
    }
    path.write_text(json.dumps(doc))
    return path


def read_samples(path: str | Path) -> tuple[str, SuiteResult]:
    """(suite reference, SuiteResult) from `write_samples` output."""
    doc = json.loads(Path(path).read_text())
    tasks = {}
    for name, t in doc["tasks"].items():
        items = [ItemResult.model_validate(i) for i in t["items"]]
        tasks[name] = TaskRun(
            name=name,
            kind=t["kind"],
            version=t["version"],
            items=items,
            estimate=mean_ci([i.score for i in items]),
            provenance=t["provenance"],
            seconds=t.get("seconds", 0.0),
        )
    result = SuiteResult(
        suite=doc["suite"],
        model=doc["model"],
        tasks=tasks,
        sanity=SanityResult.model_validate(doc["sanity"]),
        started_at=datetime.fromisoformat(doc["started_at"]),
        finished_at=datetime.fromisoformat(doc["finished_at"]),
    )
    return doc["suite_ref"], result


@dataclass
class SampleDivergences:
    divergence: DivergenceResult | None
    reference_config_hash: str | None  # the config whose reference `divergence` is from
    self_divergence: DivergenceResult | None


def read_sample_divergences(path: str | Path) -> SampleDivergences:
    """Divergences stored with `write_samples`; None where absent (older samples too)."""
    doc = json.loads(Path(path).read_text())
    div, floor = doc.get("divergence"), doc.get("self_divergence")
    return SampleDivergences(
        divergence=None if div is None else DivergenceResult.model_validate(div),
        reference_config_hash=doc.get("divergence_reference"),
        self_divergence=None if floor is None else DivergenceResult.model_validate(floor),
    )


@dataclass
class _EvalRef:
    experiment_id: uuid.UUID
    config_hash: str
    samples_uri: str


def _latest_eval(s: Session, ref: str) -> _EvalRef:
    """Latest recorded suite run for a config hash, or for an experiment's only config."""
    exp_id = _uuid(ref)
    column = BenchEvalRun.experiment_id if exp_id is not None else BenchEvalRun.config_hash
    rows = list(
        s.scalars(
            select(BenchEvalRun)
            .where(column == (exp_id if exp_id is not None else ref))
            .order_by(BenchEvalRun.created_at.desc())
        )
    )
    if not rows:
        raise LookupError(f"no eval runs for {ref}")
    configs = {r.config_hash for r in rows}
    if len(configs) > 1:
        raise LookupError(f"experiment {ref} evaluated {len(configs)} configs; pass a config hash")
    if rows[0].samples_uri is None:
        raise LookupError(f"eval runs for {ref} have no per-item samples")
    return _EvalRef(rows[0].experiment_id, rows[0].config_hash, rows[0].samples_uri)


def gate_stored(
    db_url: str | None, baseline: str, candidate: str, *, suite: str | None = None
) -> tuple[GateDecision, str, str]:
    """Re-decide the gate from stored per-item samples (and the stored divergence and
    noise floor, where present) and record it on the candidate's experiment. Returns
    (decision, baseline config hash, candidate config hash)."""
    with session_scope(db_url) as s:
        base, cand = _latest_eval(s, baseline), _latest_eval(s, candidate)
    _, base_result = read_samples(base.samples_uri)
    suite_ref, cand_result = read_samples(cand.samples_uri)
    policy = load_quality_suite(suite or suite_ref)
    floor = read_sample_divergences(base.samples_uri).self_divergence
    measured = read_sample_divergences(cand.samples_uri)
    # A candidate's divergence only applies against the baseline whose reference it used.
    divergence = measured.divergence if measured.reference_config_hash == base.config_hash else None
    decision = gate_against_baseline(
        base_result, cand_result, policy, divergence, self_divergence=floor
    )
    with session_scope(db_url) as s:
        record_gate(
            s,
            experiment_id=cand.experiment_id,
            baseline_config_hash=base.config_hash,
            candidate_config_hash=cand.config_hash,
            decision=decision,
        )
    return decision, base.config_hash, cand.config_hash


# --- reproduce -------------------------------------------------------------------


@dataclass
class ReproduceOutcome:
    original_run_id: uuid.UUID | None
    new_run_id: uuid.UUID | None
    outcome: Outcome
    warnings: list[str]
    comparison: Comparison | None  # original point's repetitions (A) vs the new run (B)

    @property
    def ok(self) -> bool:
        return (
            self.outcome.status is ExperimentStatus.COMPLETED
            and self.comparison is not None
            and self.comparison.within_normal_variance
        )


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
    point_runs: list[BenchRun] = field(default_factory=list)  # every rep of its load point


def _point_runs(s: Session, run: BenchRun) -> list[BenchRun]:
    return [
        r
        for r in repo.list_runs(s, experiment_id=run.experiment_id, config_hash=run.config_hash)
        if r.workload == run.workload and r.load_value == run.load_value
    ]


def _load_original(ref: str, db_url: str | None, spec_path: Path | None) -> _Original:
    """From a run id, or a `runs/<run id>/provenance.json` written by `bench run`."""
    path = Path(ref)
    from_file = path.suffix == ".json" and path.is_file()
    prov = Provenance.model_validate_json(path.read_text()) if from_file else None
    run_id = _uuid(path.parent.name) if from_file else uuid.UUID(ref)
    with session_scope(db_url) as s:
        run = repo.get_run(s, run_id) if run_id is not None else None
        if run is None and not from_file:
            raise LookupError(f"no run {ref}")
        if spec_path is not None:
            spec_doc = _load_spec_doc(spec_path)
        elif from_file:
            spec_doc = _load_spec_doc(path.parents[2] / "spec.json")
        else:
            exp_row = s.get(BenchExperiment, run.experiment_id)  # type: ignore[union-attr]
            spec_doc = exp_row.spec  # type: ignore[union-attr]
        if run is None:
            return _Original(prov=prov, spec_doc=spec_doc)  # type: ignore[arg-type]
        return _Original(
            prov=prov or Provenance.model_validate(run.provenance),
            spec_doc=spec_doc,
            run_id=run.id,
            cell_key=run.cell_key,
            point_runs=_point_runs(s, run),
        )


def _uuid(text: str) -> uuid.UUID | None:
    try:
        return uuid.UUID(text)
    except ValueError:
        return None


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
    `spec_path` is given. The comparison (`report.compare`) puts the original
    load point's repetitions against the new run.
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
    comparison = None
    if new_id is not None and orig.point_runs:
        with session_scope(ctx.db_url) as s:
            new_runs = repo.list_runs(s, experiment_id=experiment_id)
        if any(r.status == RUN_COMPLETED for r in new_runs) and any(
            r.status == RUN_COMPLETED for r in orig.point_runs
        ):
            comparison = compare(
                orig.point_runs,
                new_runs,
                rel_tol=tolerance,
                label_a="original",
                label_b="reproduction",
            )
    return ReproduceOutcome(
        original_run_id=orig.run_id,
        new_run_id=new_id,
        outcome=outcome,
        warnings=warnings,
        comparison=comparison,
    )


# --- reaper ------------------------------------------------------------------------


class ReapedResource(BaseModel):
    provider: str
    resource_id: str
    experiment_id: uuid.UUID | None
    ttl_at: datetime | None  # None: found by the AWS reaper, never recorded in this DB
    action: str  # terminated | gone | would terminate | skipped: ...


async def reap(
    db_url: str | None,
    *,
    now: datetime | None = None,
    dry_run: bool = False,
    ec2: Any = None,
) -> list[ReapedResource]:
    """Terminate resources whose TTL passed and mark the DB rows reaped.

    Mock servers die with the process that started them, so an expired mock
    resource not running in this process is already gone and is only marked.
    With an `ec2` client, `aws_reaper` also terminates every Loom-managed
    instance past its TTL tag, recorded here or not; an expired recorded
    instance AWS no longer lists ended itself at its TTL (`self-ttl`).
    """
    from loom_bench.providers.mock import MockProvider

    now = now or _now()
    with session_scope(db_url) as s:
        expired = [
            (r.provider, r.resource_id, r.experiment_id, r.ttl_at)
            for r in repo.list_expired_resources(s, now)
        ]
    killed = set() if dry_run else set(await MockProvider().reap(now))
    aws_ids: list[str] = []
    if ec2 is not None:
        from loom_bench.providers import aws_reaper

        aws_ids = await asyncio.to_thread(aws_reaper.reap, ec2, now, dry_run=dry_run)
    out: list[ReapedResource] = []
    for provider, resource_id, experiment_id, ttl_at in expired:
        by = TerminatedBy.REAPER
        if provider == "aws_ec2" and ec2 is None:
            action = "skipped: aws not configured"
        elif dry_run:
            action = "would terminate"
        elif provider == "aws_ec2":
            gone = resource_id not in aws_ids
            action, by = ("gone", TerminatedBy.SELF_TTL) if gone else ("terminated", by)
        else:
            action = "terminated" if resource_id in killed else "gone"
        if not dry_run and not action.startswith("skipped"):
            with session_scope(db_url) as s:
                repo.mark_terminated(s, provider=provider, resource_id=resource_id, by=by, at=now)
        out.append(
            ReapedResource(
                provider=provider,
                resource_id=resource_id,
                experiment_id=experiment_id,
                ttl_at=ttl_at,
                action=action,
            )
        )
    recorded = {r[1] for r in expired}
    out += [
        ReapedResource(
            provider="aws_ec2",
            resource_id=rid,
            experiment_id=None,
            ttl_at=None,
            action="would terminate" if dry_run else "terminated",
        )
        for rid in aws_ids
        if rid not in recorded
    ]
    return out
