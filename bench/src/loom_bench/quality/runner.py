"""Run a pinned suite against an endpoint, gate a candidate against a baseline,
and persist both."""

from __future__ import annotations

import logging
import tempfile
import time
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

from loom_bench.jobs import EvalJob, EvalJobResult, EvalTaskResult
from loom_bench.provenance import to_jsonable
from loom_bench.quality.client import EvalClient
from loom_bench.quality.divergence import (
    DivergenceResult,
    ReferenceLogprobs,
    capture_reference,
    load_prompts,
    measure_divergence,
    score_against_reference,
)
from loom_bench.quality.gate import GateDecision, GatePolicy, evaluate_gate
from loom_bench.quality.sanity import SanityResult, check_completions
from loom_bench.quality.suite import Suite
from loom_bench.quality.tasks import build_task
from loom_bench.quality.tasks.base import (
    Completion,
    EvalContext,
    ItemResult,
    check_request_errors,
)
from loom_bench.stats import Estimate, mean_ci
from loom_bench.store.models import BenchEvalRun, BenchGateDecision
from loom_bench.store.repo import record_eval_run, record_gate_decision
from loom_bench.tokenize import verify_snapshot

log = logging.getLogger(__name__)


@dataclass(slots=True)
class TaskRun:
    name: str
    kind: str
    version: str
    items: list[ItemResult]
    estimate: Estimate  # mean score with a Student-t CI over items, clipped to [0, 1]
    provenance: dict[str, Any] = field(default_factory=dict)
    seconds: float = 0.0


def score_estimate(items: Sequence[ItemResult]) -> Estimate:
    """Mean item score with its Student-t CI. Scores are fractions in [0, 1], so the
    interval is clipped to that range ("t_clipped"): near a perfect score the plain
    interval reaches past 1 (vLLM tool_calling 0.967 [0.920, 1.013] in 565b8d3f)."""
    return mean_ci([i.score for i in items], lower=0.0, upper=1.0)


@dataclass(slots=True)
class SuiteResult:
    suite: str
    model: str
    tasks: dict[str, TaskRun]
    sanity: SanityResult
    started_at: datetime
    finished_at: datetime
    # Independent passes of the suite pooled into these items (`pool_replicates`): each
    # item's score is then its mean over the passes.
    replicates: int = 1

    def scores(self) -> dict[str, list[ItemResult]]:
        return {name: run.items for name, run in self.tasks.items()}


def pool_replicates(results: Sequence[SuiteResult]) -> SuiteResult:
    """One SuiteResult from R independent passes of the same suite on the same config.

    Each item's score becomes its mean over the passes, an unbiased estimate of the
    probability the config passes it. Greedy decoding is not run-to-run deterministic
    under batching (Qwen3-8B on vLLM: 32 of 541 IFEval items changed score across three
    runs), and a single pass carries that noise into every paired comparison; averaging
    R passes divides its variance by R, while item sampling, the part the gate
    generalises over, is unchanged. The per-pass scores stay in each item's
    `meta["replicate_scores"]`, so the noise can be measured from the samples.
    Sanity counts are summed over all passes.
    """
    if not results:
        raise ValueError("pool_replicates needs at least one result")
    if len(results) == 1:
        return results[0]
    first = results[0]
    for r in results[1:]:
        if (r.suite, r.model) != (first.suite, first.model):
            raise ValueError(
                f"replicates ran different suites or models: {first.suite}/{first.model} "
                f"vs {r.suite}/{r.model}"
            )
        if r.tasks.keys() != first.tasks.keys():
            raise ValueError(
                f"replicates ran different tasks: {sorted(first.tasks)} vs {sorted(r.tasks)}"
            )
    tasks: dict[str, TaskRun] = {}
    for name, run in first.tasks.items():
        runs = [r.tasks[name] for r in results]
        versions = {t.version for t in runs}
        if len(versions) != 1:
            raise ValueError(f"{name}: replicates ran versions {sorted(versions)}")
        by_id = [{i.item_id: i for i in t.items} for t in runs]
        if any(len(m) != len(t.items) for m, t in zip(by_id, runs, strict=True)):
            raise ValueError(f"{name}: duplicate item ids in a replicate")
        if any(m.keys() != by_id[0].keys() for m in by_id[1:]):
            raise ValueError(f"{name}: replicates scored different items")
        items = []
        for item in run.items:
            reps = [m[item.item_id] for m in by_id]
            if len({i.content_hash for i in reps}) != 1:
                raise ValueError(f"{name}: item {item.item_id!r} has different content")
            scores = [i.score for i in reps]
            items.append(
                item.model_copy(
                    update={
                        "score": sum(scores) / len(scores),
                        "meta": {**item.meta, "replicate_scores": scores},
                    }
                )
            )
        tasks[name] = TaskRun(
            name=name,
            kind=run.kind,
            version=run.version,
            items=items,
            estimate=score_estimate(items),
            provenance={
                **run.provenance,
                "replicates": len(runs),
                "replicate_means": [t.estimate.mean for t in runs],
            },
            seconds=sum(t.seconds for t in runs),
        )
    counts: dict[str, int] = {}
    for r in results:
        for k, v in r.sanity.counts.items():
            counts[k] = counts.get(k, 0) + v
    return SuiteResult(
        suite=first.suite,
        model=first.model,
        tasks=tasks,
        sanity=SanityResult(n=sum(r.sanity.n for r in results), counts=counts),
        started_at=min(r.started_at for r in results),
        finished_at=max(r.finished_at for r in results),
        replicates=sum(r.replicates for r in results),
    )


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
    extra_body: Mapping[str, Any] | None = None,
    seed: int | None = None,
    local_tokenizers: Mapping[str, str] | None = None,
) -> SuiteResult:
    """Run every task of `suite` (or the `only` subset) against one endpoint.

    `base_url` includes the API prefix (``http://host:8000/v1``). Tasks run one
    after another, each with up to `concurrency` requests in flight. Requests
    carry `extra_body` (default: the suite's) and `seed` (default: the suite's).
    `local_tokenizers` maps HF repos to snapshot directories tasks load them from.
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
        seed=suite.seed if seed is None else seed,
        extra_body=suite.extra_body() if extra_body is None else extra_body,
    ) as client:
        for spec in selected:
            task = build_task(spec.kind, spec.name, spec.params)
            ctx = EvalContext(
                client=client,
                workdir=workdir / spec.name,
                allow_code_exec=allow_code_exec,
                local_tokenizers=dict(local_tokenizers or {}),
            )
            ctx.workdir.mkdir(parents=True, exist_ok=True)
            t0 = time.monotonic()
            out = await task.run(ctx)
            seconds = time.monotonic() - t0
            if not out.items:
                raise RuntimeError(f"{spec.name}: task produced no items")
            check_request_errors(spec.name, out.items)
            version = out.version or task.version
            runs[spec.name] = TaskRun(
                name=spec.name,
                kind=spec.kind,
                version=version,
                items=out.items,
                estimate=score_estimate(out.items),
                provenance=out.provenance,
                seconds=seconds,
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
    prompts = spec.select(load_prompts())
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


async def execute_eval_job(job: EvalJob, workdir: Path | None = None) -> EvalJobResult:
    """Run an EvalJob where its endpoint is reachable (in-process, or on the GPU host via
    `bench quality job`). `workdir` keeps harness output; default: a temporary directory."""
    if workdir is None:
        with tempfile.TemporaryDirectory(prefix="loom-eval-") as tmp:
            return await execute_eval_job(job, Path(tmp))
    suite = job.suite
    local_tokenizers: dict[str, str] = {}
    tok = job.tokenizer
    if tok is not None and tok.local_dir is not None:
        assert tok.repo is not None and tok.revision is not None  # TokenizerSpec validates it
        local_tokenizers[tok.repo] = str(verify_snapshot(tok.local_dir, tok.repo, tok.revision))
    started = datetime.now(UTC)
    result = await run_suite(
        suite,
        job.base_url,
        job.served_model,
        workdir=workdir,
        concurrency=job.concurrency,
        timeout_s=job.request_timeout_s,
        allow_code_exec=job.allow_code_exec,
        only=job.tasks,
        extra_body=job.extra_body,
        seed=job.seed,
        local_tokenizers=local_tokenizers,
    )
    reference: ReferenceLogprobs | None = None
    divergence: DivergenceResult | None = None
    divergence_error: str | None = None
    if job.divergence is not None:
        try:
            reference, divergence = await _divergence(job)
        except Exception as e:  # the task scores above stand; only this half failed
            log.exception("divergence (%s) failed", job.divergence)
            divergence_error = f"{type(e).__name__}: {e}"
    return EvalJobResult(
        run_id=job.run_id,
        suite=result.suite,
        model=result.model,
        tasks={
            name: EvalTaskResult(
                kind=run.kind,
                version=run.version,
                items=run.items,
                provenance=run.provenance,
                seconds=run.seconds,
            )
            for name, run in result.tasks.items()
        },
        sanity=result.sanity,
        reference=reference,
        divergence=divergence,
        divergence_error=divergence_error,
        started_at=started.isoformat(),
        finished_at=datetime.now(UTC).isoformat(),
    )


async def _divergence(job: EvalJob) -> tuple[ReferenceLogprobs | None, DivergenceResult | None]:
    """The job's half of the divergence: (reference with its noise floor, None) for a
    capture, (None, divergence) for a score."""
    suite = job.suite
    spec = suite.divergence
    assert spec is not None  # EvalJob validates it
    stats: dict[str, Any] = {
        "confidence": suite.gate.confidence,
        "n_boot": suite.gate.n_boot,
        "seed": job.seed,
    }
    client_opts: dict[str, Any] = {"timeout_s": job.request_timeout_s, "seed": job.seed}
    async with EvalClient(
        job.base_url, job.served_model, concurrency=job.concurrency, **client_opts
    ) as client:
        if job.divergence == "score":
            assert job.reference is not None  # EvalJob validates it
            return None, await score_against_reference(client, job.reference, **stats)
        reference = await capture_reference(
            client,
            spec.select(load_prompts()),
            top_k=spec.top_k,
            max_new_tokens=spec.max_new_tokens,
        )
    if job.divergence == "capture_and_floor":
        async with EvalClient(
            job.base_url, job.served_model, concurrency=spec.floor_concurrency, **client_opts
        ) as client:
            floor = await score_against_reference(client, reference, **stats)
        reference = reference.model_copy(update={"self_divergence": floor})
    return reference, None


def suite_result_of(result: EvalJobResult) -> SuiteResult:
    """The SuiteResult an EvalJob's result describes (for the gate and the records)."""
    return SuiteResult(
        suite=result.suite,
        model=result.model,
        tasks={
            name: TaskRun(
                name=name,
                kind=t.kind,
                version=t.version,
                items=t.items,
                estimate=score_estimate(t.items),
                provenance=t.provenance,
                seconds=t.seconds,
            )
            for name, t in result.tasks.items()
        },
        sanity=result.sanity,
        started_at=datetime.fromisoformat(result.started_at),
        finished_at=datetime.fromisoformat(result.finished_at),
    )


def gate_against_baseline(
    baseline: SuiteResult,
    candidate: SuiteResult,
    policy: Suite | GatePolicy,
    divergence: DivergenceResult | None = None,
    *,
    self_divergence: DivergenceResult | None = None,
    divergence_error: str | None = None,
) -> GateDecision:
    """Gate decision for `candidate` vs `baseline`; both must come from the same suite.
    `self_divergence` is the baseline's noise floor (`ReferenceLogprobs.self_divergence`);
    `divergence_error` says why a divergence that should have been measured was not."""
    if baseline.suite != candidate.suite:
        raise ValueError(f"suites differ: {baseline.suite} vs {candidate.suite}")
    # A task's version covers its pinned data (e.g. json_schema "1+data.2"): results from
    # different versions measure different item sets and never pair.
    for name in baseline.tasks.keys() & candidate.tasks.keys():
        vb, vc = baseline.tasks[name].version, candidate.tasks[name].version
        if vb != vc:
            raise ValueError(
                f"{name}: baseline ran version {vb}, candidate {vc}; rerun the baseline"
            )
    gate_policy = policy.policy() if isinstance(policy, Suite) else policy
    return evaluate_gate(
        baseline.scores(),
        candidate.scores(),
        divergence,
        candidate.sanity,
        gate_policy,
        self_divergence=self_divergence,
        divergence_error=divergence_error,
        baseline_replicates=baseline.replicates,
        candidate_replicates=candidate.replicates,
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
