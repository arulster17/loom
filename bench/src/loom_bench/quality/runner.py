"""Run a pinned suite against an endpoint, gate a candidate against a baseline,
and persist both."""

from __future__ import annotations

import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

from loom_bench.provenance import to_jsonable
from loom_bench.quality.client import EvalClient
from loom_bench.quality.divergence import DivergenceResult, load_prompts, measure_divergence
from loom_bench.quality.gate import GateDecision, GatePolicy, evaluate_gate
from loom_bench.quality.sanity import SanityResult, check_completions
from loom_bench.quality.suite import Suite
from loom_bench.quality.tasks import build_task
from loom_bench.quality.tasks.base import Completion, EvalContext, ItemResult
from loom_bench.stats import Estimate, mean_ci
from loom_bench.store.models import BenchEvalRun, BenchGateDecision
from loom_bench.store.repo import record_eval_run, record_gate_decision


@dataclass(slots=True)
class TaskRun:
    name: str
    kind: str
    version: str
    items: list[ItemResult]
    estimate: Estimate  # mean score with a Student-t CI over items
    provenance: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SuiteResult:
    suite: str
    model: str
    tasks: dict[str, TaskRun]
    sanity: SanityResult
    started_at: datetime
    finished_at: datetime

    def scores(self) -> dict[str, list[ItemResult]]:
        return {name: run.items for name, run in self.tasks.items()}


async def run_suite(
    suite: Suite,
    base_url: str,
    served_model: str,
    *,
    workdir: Path,
    api_key: str | None = None,
    concurrency: int = 16,
    timeout_s: float = 300.0,
    allow_code_exec: bool = False,
    only: Sequence[str] | None = None,
) -> SuiteResult:
    """Run every task of `suite` (or the `only` subset) against one endpoint.

    `base_url` includes the API prefix (``http://host:8000/v1``). Tasks run one
    after another, each with up to `concurrency` requests in flight.
    """
    selected = [t for t in suite.tasks if only is None or t.name in only]
    if only is not None and len(selected) != len(set(only)):
        raise ValueError(f"unknown tasks in {sorted(only)}")
    started = datetime.now(UTC)
    runs: dict[str, TaskRun] = {}
    completions: list[Completion] = []
    async with EvalClient(
        base_url,
        served_model,
        api_key=api_key,
        concurrency=concurrency,
        timeout_s=timeout_s,
        seed=suite.seed,
        extra_body=suite.extra_body(),
    ) as client:
        for spec in selected:
            task = build_task(spec.kind, spec.name, spec.params)
            ctx = EvalContext(
                client=client, workdir=workdir / spec.name, allow_code_exec=allow_code_exec
            )
            ctx.workdir.mkdir(parents=True, exist_ok=True)
            out = await task.run(ctx)
            if not out.items:
                raise RuntimeError(f"{spec.name}: task produced no items")
            version = str(out.provenance.get("lm_eval", {}).get("task_version") or task.version)
            runs[spec.name] = TaskRun(
                name=spec.name,
                kind=spec.kind,
                version=version,
                items=out.items,
                estimate=mean_ci([i.score for i in out.items]),
                provenance=out.provenance,
            )
            completions += out.completions
    return SuiteResult(
        suite=suite.suite,
        model=served_model,
        tasks=runs,
        sanity=check_completions(completions, suite.sanity_checks),
        started_at=started,
        finished_at=datetime.now(UTC),
    )


async def run_divergence(
    suite: Suite,
    reference_base_url: str,
    candidate_base_url: str,
    served_model: str,
    *,
    candidate_model: str | None = None,
    api_key: str | None = None,
    concurrency: int = 16,
    timeout_s: float = 300.0,
) -> DivergenceResult:
    """Teacher-forced divergence of the candidate from the reference on the pinned prompts."""
    spec = suite.divergence
    if spec is None:
        raise ValueError(f"suite {suite.suite} has no divergence section")
    prompts = load_prompts()[: spec.prompts]
    common: dict[str, Any] = {
        "api_key": api_key,
        "concurrency": concurrency,
        "timeout_s": timeout_s,
        "seed": suite.seed,
    }
    async with (
        EvalClient(reference_base_url, served_model, **common) as ref,
        EvalClient(candidate_base_url, candidate_model or served_model, **common) as cand,
    ):
        return await measure_divergence(
            ref,
            cand,
            prompts,
            top_k=spec.top_k,
            max_new_tokens=spec.max_new_tokens,
            confidence=suite.gate.confidence,
            n_boot=suite.gate.n_boot,
            seed=suite.seed,
        )


def gate_against_baseline(
    baseline: SuiteResult,
    candidate: SuiteResult,
    policy: Suite | GatePolicy,
    divergence: DivergenceResult | None = None,
) -> GateDecision:
    """Gate decision for `candidate` vs `baseline`; both must come from the same suite."""
    if baseline.suite != candidate.suite:
        raise ValueError(f"suites differ: {baseline.suite} vs {candidate.suite}")
    gate_policy = policy.policy() if isinstance(policy, Suite) else policy
    return evaluate_gate(
        baseline.scores(), candidate.scores(), divergence, candidate.sanity, gate_policy
    )


def record_suite_result(
    session: Session,
    *,
    experiment_id: uuid.UUID,
    config_hash: str,
    result: SuiteResult,
    provenance: Mapping[str, Any] | Any,
    samples_uri: str | None = None,
) -> list[BenchEvalRun]:
    """One `bench_eval_runs` row per task: mean score, t-interval and provenance."""
    base = to_jsonable(provenance)
    rows = []
    for run in result.tasks.values():
        est = run.estimate
        rows.append(
            record_eval_run(
                session,
                experiment_id=experiment_id,
                config_hash=config_hash,
                task=run.name,
                task_version=run.version,
                n=est.n,
                score=est.mean,
                ci_low=est.lo,
                ci_high=est.hi,
                provenance={
                    **base,
                    "eval": {"suite": result.suite, "kind": run.kind, **run.provenance},
                },
                samples_uri=samples_uri,
                created_at=result.finished_at,
            )
        )
    return rows


def record_gate(
    session: Session,
    *,
    experiment_id: uuid.UUID,
    baseline_config_hash: str,
    candidate_config_hash: str,
    decision: GateDecision,
) -> BenchGateDecision:
    return record_gate_decision(
        session,
        experiment_id=experiment_id,
        baseline_config_hash=baseline_config_hash,
        candidate_config_hash=candidate_config_hash,
        decision=decision.decision.value,
        details=decision.details(),
    )
