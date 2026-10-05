"""Regression gate: is a candidate config non-inferior to the baseline?

Per task, scores are paired by item id and delta = mean(candidate) -
mean(baseline), with a percentile-bootstrap CI over items
(`paired_bootstrap_delta`, items resampled jointly). The CI is then widened to
at least ±3/n around the point delta (rule of three): with n items a bootstrap
cannot see effects rarer than about 1/n, and when the two configs agree on
every item its interval collapses to a single point, which would "prove"
non-inferiority from a handful of items.

With threshold t (default 0.01 = one point absolute) and CI [lo, hi]:

- n < min_samples                       -> INCONCLUSIVE
- point delta < -t, or hi < -t          -> FAIL
- lo >= -t                              -> PASS (non-inferiority shown)
- otherwise                             -> INCONCLUSIVE (needs more samples)

A two-sided 95% CI makes the PASS rule a one-sided test at 2.5%.

Divergence and sanity checks FAIL the gate on their own: mean KL(ref || cand)
above `max_kl`, top-1 agreement below `min_top1`, or any sanity rate above its
limit. The overall decision is the worst one (FAIL > INCONCLUSIVE > PASS); the
change is blocked on FAIL, and on INCONCLUSIVE unless `inconclusive_blocks` is
off.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from enum import StrEnum
from typing import Annotated, Any, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from loom_bench.quality.divergence import DivergenceResult
from loom_bench.quality.sanity import SanityLimits, SanityResult
from loom_bench.quality.tasks.base import ItemResult
from loom_bench.stats import paired_bootstrap_delta

Threshold = Annotated[float, Field(ge=0.0, lt=1.0)]
MinSamples = Annotated[int, Field(ge=2)]
RESOLUTION_FACTOR = 3.0


class Verdict(StrEnum):
    PASS = "pass"
    FAIL = "fail"
    INCONCLUSIVE = "inconclusive"


_SEVERITY = {Verdict.PASS: 0, Verdict.INCONCLUSIVE: 1, Verdict.FAIL: 2}


def worst(verdicts: Sequence[Verdict]) -> Verdict:
    return max(verdicts, key=_SEVERITY.__getitem__, default=Verdict.PASS)


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class TaskPolicy(_Strict):
    threshold: Threshold | None = None
    min_samples: MinSamples | None = None


class GatePolicy(_Strict):
    threshold: Threshold = 0.01
    min_samples: MinSamples = 300
    confidence: Annotated[float, Field(gt=0.0, lt=1.0)] = 0.95
    n_boot: Annotated[int, Field(ge=100)] = 10_000
    seed: int = 0
    inconclusive_blocks: bool = True
    tasks: dict[str, TaskPolicy] = Field(default_factory=dict)
    max_kl: Annotated[float, Field(ge=0.0)] | None = 0.05
    min_top1: Annotated[float, Field(ge=0.0, le=1.0)] | None = 0.95
    sanity: SanityLimits = Field(default_factory=SanityLimits)

    def threshold_for(self, task: str) -> float:
        tp = self.tasks.get(task)
        return self.threshold if tp is None or tp.threshold is None else tp.threshold

    def min_samples_for(self, task: str) -> int:
        tp = self.tasks.get(task)
        return self.min_samples if tp is None or tp.min_samples is None else tp.min_samples


class TaskVerdict(_Strict):
    task: str
    verdict: Verdict
    n: int
    threshold: float
    min_samples: int
    baseline_mean: float
    candidate_mean: float
    delta: float
    ci_low: float | None
    ci_high: float | None
    bootstrap_low: float | None
    bootstrap_high: float | None
    improved: int  # items scoring higher on the candidate
    regressed: int  # items scoring lower on the candidate
    reason: str


class CheckVerdict(_Strict):
    verdict: Verdict
    reason: str
    result: dict[str, Any] | None = None


class GateDecision(_Strict):
    decision: Verdict
    blocked: bool
    tasks: list[TaskVerdict]
    divergence: CheckVerdict
    sanity: CheckVerdict
    policy: GatePolicy

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        parts = [t.verdict for t in self.tasks] + [self.divergence.verdict, self.sanity.verdict]
        if self.decision != worst(parts):
            raise ValueError("decision must be the worst of its parts")
        return self

    @property
    def reasons(self) -> list[str]:
        return [f"{t.task}: {t.reason}" for t in self.tasks] + [
            f"divergence: {self.divergence.reason}",
            f"sanity: {self.sanity.reason}",
        ]

    def details(self) -> dict[str, Any]:
        """JSON document for `record_gate_decision(details=...)`."""
        return self.model_dump(mode="json")

    def summary(self) -> str:
        head = f"gate {self.decision.value.upper()}" + (" (blocked)" if self.blocked else "")
        return "\n".join([head, *(f"  {r}" for r in self.reasons)])


def _pts(x: float) -> str:
    return f"{100 * x:+.2f} pts"


def _pair(
    task: str, baseline: Sequence[ItemResult], candidate: Sequence[ItemResult]
) -> tuple[list[float], list[float]]:
    base = {r.item_id: r for r in baseline}
    cand = {r.item_id: r for r in candidate}
    if len(base) != len(baseline) or len(cand) != len(candidate):
        raise ValueError(f"{task}: duplicate item ids")
    if base.keys() != cand.keys():
        only_b, only_c = len(base.keys() - cand.keys()), len(cand.keys() - base.keys())
        raise ValueError(
            f"{task}: baseline and candidate cover different items "
            f"({only_b} only in baseline, {only_c} only in candidate)"
        )
    ids = sorted(base)
    for i in ids:
        hb, hc = base[i].content_hash, cand[i].content_hash
        if hb is not None and hc is not None and hb != hc:
            raise ValueError(f"{task}: item {i!r} has different content in baseline and candidate")
    return [base[i].score for i in ids], [cand[i].score for i in ids]


def evaluate_task(
    task: str,
    baseline: Sequence[ItemResult],
    candidate: Sequence[ItemResult],
    policy: GatePolicy,
) -> TaskVerdict:
    b, c = _pair(task, baseline, candidate)
    n, t, min_n = len(b), policy.threshold_for(task), policy.min_samples_for(task)
    common: dict[str, Any] = {
        "task": task,
        "n": n,
        "threshold": t,
        "min_samples": min_n,
        "baseline_mean": sum(b) / n if n else 0.0,
        "candidate_mean": sum(c) / n if n else 0.0,
        "improved": sum(1 for x, y in zip(b, c, strict=True) if y > x),
        "regressed": sum(1 for x, y in zip(b, c, strict=True) if y < x),
    }
    if n < min_n:
        delta = common["candidate_mean"] - common["baseline_mean"]
        return TaskVerdict(
            verdict=Verdict.INCONCLUSIVE,
            delta=delta,
            ci_low=None,
            ci_high=None,
            bootstrap_low=None,
            bootstrap_high=None,
            reason=f"n={n} is below min_samples={min_n} (delta {_pts(delta)})",
            **common,
        )
    boot = paired_bootstrap_delta(
        c, b, confidence=policy.confidence, n_boot=policy.n_boot, seed=policy.seed
    )
    if boot.lo is None or boot.hi is None:
        raise ValueError(f"{task}: a paired bootstrap needs at least 2 items")
    floor = RESOLUTION_FACTOR / n
    lo, hi = min(boot.lo, boot.point - floor), max(boot.hi, boot.point + floor)
    ci = f"{_pts(boot.point)} [{_pts(lo)}, {_pts(hi)}], n={n}"
    if boot.point < -t:
        verdict, why = Verdict.FAIL, f"delta {ci} is a drop of more than {100 * t:.2f} pts"
    elif hi < -t:
        verdict, why = Verdict.FAIL, f"delta {ci}: whole CI below -{100 * t:.2f} pts"
    elif lo >= -t:
        verdict, why = Verdict.PASS, f"delta {ci}: non-inferior at {100 * t:.2f} pts"
    else:
        verdict, why = (
            Verdict.INCONCLUSIVE,
            f"delta {ci}: CI crosses -{100 * t:.2f} pts, more samples needed",
        )
    return TaskVerdict(
        verdict=verdict,
        delta=boot.point,
        ci_low=lo,
        ci_high=hi,
        bootstrap_low=boot.lo,
        bootstrap_high=boot.hi,
        reason=why,
        **common,
    )


def _divergence_verdict(div: DivergenceResult | None, policy: GatePolicy) -> CheckVerdict:
    if div is None:
        return CheckVerdict(verdict=Verdict.PASS, reason="not measured")
    problems = []
    if policy.max_kl is not None and div.kl.point > policy.max_kl:
        problems.append(f"mean KL {div.kl.point:.4f} nats exceeds {policy.max_kl:.4f}")
    if policy.min_top1 is not None and div.top1.point < policy.min_top1:
        problems.append(f"top-1 agreement {div.top1.point:.2%} is below {policy.min_top1:.2%}")
    summary = (
        f"KL {div.kl.point:.4f} nats, top-1 {div.top1.point:.2%} "
        f"({div.n_prompts} prompts, {div.n_positions} positions)"
    )
    return CheckVerdict(
        verdict=Verdict.FAIL if problems else Verdict.PASS,
        reason="; ".join(problems) if problems else summary,
        result=div.model_dump(mode="json"),
    )


def _sanity_verdict(sanity: SanityResult | None, policy: GatePolicy) -> CheckVerdict:
    if sanity is None:
        return CheckVerdict(verdict=Verdict.PASS, reason="not measured")
    problems = sanity.violations(policy.sanity)
    rates = ", ".join(f"{k} {v:.2%}" for k, v in sanity.rates.items())
    return CheckVerdict(
        verdict=Verdict.FAIL if problems else Verdict.PASS,
        reason="; ".join(problems) if problems else f"{sanity.n} outputs: {rates}",
        result={**sanity.model_dump(mode="json"), "rates": sanity.rates},
    )


def evaluate_gate(
    baseline: Mapping[str, Sequence[ItemResult]],
    candidate: Mapping[str, Sequence[ItemResult]],
    divergence: DivergenceResult | None,
    sanity: SanityResult | None,
    policy: GatePolicy | None = None,
) -> GateDecision:
    """Decide whether `candidate` may replace `baseline`; see the module docstring."""
    policy = policy or GatePolicy()
    if baseline.keys() != candidate.keys():
        raise ValueError(
            f"task sets differ: baseline {sorted(baseline)}, candidate {sorted(candidate)}"
        )
    tasks = [evaluate_task(name, baseline[name], candidate[name], policy) for name in baseline]
    div = _divergence_verdict(divergence, policy)
    san = _sanity_verdict(sanity, policy)
    decision = worst([t.verdict for t in tasks] + [div.verdict, san.verdict])
    blocked = decision is Verdict.FAIL or (
        decision is Verdict.INCONCLUSIVE and policy.inconclusive_blocks
    )
    return GateDecision(
        decision=decision, blocked=blocked, tasks=tasks, divergence=div, sanity=san, policy=policy
    )
