"""Regression gate: is a candidate config non-inferior to the baseline?

Per task, scores are paired by item id and delta = mean(candidate) -
mean(baseline), with a percentile-bootstrap CI over items
(`paired_bootstrap_delta`, items resampled jointly). A side run R > 1 times
(`quality.replicates`) contributes each item's mean score over its R passes
(`pool_replicates`): engine nondeterminism then adds var/R instead of its full
variance to every item's difference, and the bootstrap over items still carries
what remains of it. The CI is then widened to
at least ±3/n around the point delta (rule of three): with n items a bootstrap
cannot see effects rarer than about 1/n, and when the two configs agree on
every item its interval collapses to a single point, which would "prove"
non-inferiority from a handful of items.

With threshold t (default 0.01 = one point absolute) and CI [lo, hi]:

- n < min_samples                       -> INCONCLUSIVE
- point delta < -t, or hi < -t          -> FAIL
- lo >= -t                              -> PASS (non-inferiority shown)
- otherwise                             -> INCONCLUSIVE (needs more samples)

A two-sided 95% CI makes the PASS rule a one-sided test at 2.5%. The verdicts do not
look at zero, but the reasons do: a CI wholly below zero is a measurable drop and
is reported as one, whatever the verdict ("non-inferior, but measurably lower";
"a real drop, not shown to be within the margin"), so an INCONCLUSIVE never hides a
regression the data already shows.

`GATE_METHOD` names this procedure and is stored with every decision.

Logprob divergence is judged against the noise floor measured on the baseline
(`self_divergence`: the baseline scored against its own reference under
different batching). The limits are

    KL limit      = max(max_kl, noise_multiple x self-KL CI upper bound)
    disagreement  = max(1 - min_top1, noise_multiple x self-disagreement CI upper bound)

and, whatever the floor, a hard ceiling (`ceiling_kl`, `ceiling_top1`):

- divergence beyond the ceiling                              -> FAIL
- divergence beyond the limits, every task PASS               -> REVIEW
- divergence beyond the limits, any task FAIL / INCONCLUSIVE  -> FAIL
- otherwise                                                   -> PASS

Without a self-divergence the absolute `max_kl` / `min_top1` are the limits and
the reason says the check is uncalibrated. A divergence that should have been measured
but failed (its capture or scoring raised) is INCONCLUSIVE, with the error as reason;
the task verdicts are still decided and reported. A sanity rate above its limit FAILs.
The overall decision is the worst part (FAIL > INCONCLUSIVE > REVIEW > PASS);
the change is blocked on FAIL, on INCONCLUSIVE unless `inconclusive_blocks` is
off, and on REVIEW only when `review_blocks` is on.
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
Nats = Annotated[float, Field(ge=0.0)]
Share = Annotated[float, Field(ge=0.0, le=1.0)]
NoiseMultiple = Annotated[float, Field(ge=1.0)]
MinSamples = Annotated[int, Field(ge=2)]
RESOLUTION_FACTOR = 3.0
# v1: one pass per side. v2: per-item means over each side's replicates, and reasons
# that name a CI wholly below zero.
GATE_METHOD = "paired-bootstrap-over-items/replicate-means/v2"


class Verdict(StrEnum):
    PASS = "pass"
    REVIEW = "review"  # divergence beyond the calibrated limits while every task passes
    FAIL = "fail"
    INCONCLUSIVE = "inconclusive"


_SEVERITY = {Verdict.PASS: 0, Verdict.REVIEW: 1, Verdict.INCONCLUSIVE: 2, Verdict.FAIL: 3}


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
    # Divergence: absolute limits (the calibrated ones are never stricter), the multiple of
    # the baseline's measured self-divergence the limits widen to, and the hard ceiling.
    max_kl: Nats | None = 0.05
    min_top1: Share | None = 0.95
    noise_multiple: NoiseMultiple = 5.0
    ceiling_kl: Nats | None = 0.5
    ceiling_top1: Share | None = 0.80
    review_blocks: bool = False
    sanity: SanityLimits = Field(default_factory=SanityLimits)

    @model_validator(mode="after")
    def _ceiling_beyond_limits(self) -> Self:
        lim_kl, ceil_kl = self.max_kl, self.ceiling_kl
        if lim_kl is not None and ceil_kl is not None and ceil_kl <= lim_kl:
            raise ValueError("ceiling_kl must be above max_kl")
        lim_top1, ceil_top1 = self.min_top1, self.ceiling_top1
        if lim_top1 is not None and ceil_top1 is not None and ceil_top1 >= lim_top1:
            raise ValueError("ceiling_top1 must be below min_top1")
        return self

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


class DivergenceLimits(_Strict):
    """The divergence limits one decision applied, and where they came from."""

    calibrated: bool  # False: no self-divergence, the absolute limits were used
    noise_multiple: float
    self_kl: float | None  # CI upper bound of the baseline's self-KL
    self_top1: float | None  # CI lower bound of the baseline's self top-1 agreement
    max_kl: float | None
    min_top1: float | None
    ceiling_kl: float | None
    ceiling_top1: float | None


class DivergenceVerdict(CheckVerdict):
    limits: DivergenceLimits | None = None
    self_divergence: dict[str, Any] | None = None


class GateDecision(_Strict):
    decision: Verdict
    blocked: bool
    tasks: list[TaskVerdict]
    divergence: DivergenceVerdict
    sanity: CheckVerdict
    policy: GatePolicy
    method: str = GATE_METHOD
    baseline_replicates: int = 1
    candidate_replicates: int = 1

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
        head = f"gate {self.decision.value.upper()}"
        if self.blocked:
            head += " (blocked)"
        elif self.decision is Verdict.REVIEW:
            head += " (needs review, not blocking)"
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
    real_drop = hi < 0.0
    if boot.point < -t:
        verdict, why = Verdict.FAIL, f"delta {ci} is a drop of more than {100 * t:.2f} pts"
    elif hi < -t:
        verdict, why = Verdict.FAIL, f"delta {ci}: whole CI below -{100 * t:.2f} pts"
    elif lo >= -t:
        verdict, why = Verdict.PASS, f"delta {ci}: non-inferior at {100 * t:.2f} pts"
        if real_drop:
            why += ", but measurably lower (CI below 0)"
    elif real_drop:
        verdict, why = (
            Verdict.INCONCLUSIVE,
            f"delta {ci}: a real drop (CI below 0), not shown to be within "
            f"{100 * t:.2f} pts; more samples needed",
        )
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


def divergence_limits(
    policy: GatePolicy, self_divergence: DivergenceResult | None
) -> DivergenceLimits:
    """Limits for a candidate's divergence: the absolute ones, widened to
    `noise_multiple` x the baseline's self-divergence when it was measured."""
    common: dict[str, Any] = {
        "noise_multiple": policy.noise_multiple,
        "ceiling_kl": policy.ceiling_kl,
        "ceiling_top1": policy.ceiling_top1,
    }
    if self_divergence is None:
        return DivergenceLimits(
            calibrated=False,
            self_kl=None,
            self_top1=None,
            max_kl=policy.max_kl,
            min_top1=policy.min_top1,
            **common,
        )
    kl, top1 = self_divergence.kl, self_divergence.top1
    self_kl = kl.point if kl.hi is None else kl.hi
    self_top1 = top1.point if top1.lo is None else top1.lo
    m = policy.noise_multiple
    max_kl = None if policy.max_kl is None else max(policy.max_kl, m * self_kl)
    min_top1 = (
        None
        if policy.min_top1 is None
        else max(0.0, 1.0 - max(1.0 - policy.min_top1, m * (1.0 - self_top1)))
    )
    return DivergenceLimits(
        calibrated=True,
        self_kl=self_kl,
        self_top1=self_top1,
        max_kl=max_kl,
        min_top1=min_top1,
        **common,
    )


def limits_text(lim: DivergenceLimits) -> str:
    kl = "none" if lim.max_kl is None else f"{lim.max_kl:.4f} nats"
    top1 = "none" if lim.min_top1 is None else f"{lim.min_top1:.2%}"
    if not lim.calibrated:
        return f"uncalibrated (no self-divergence measured): absolute limits KL {kl}, top-1 {top1}"
    assert lim.self_kl is not None and lim.self_top1 is not None
    return (
        f"limits KL {kl}, top-1 {top1}: the looser of the absolute limits and "
        f"{lim.noise_multiple:g}x the baseline's noise floor "
        f"(self-KL <= {lim.self_kl:.4f} nats, self top-1 >= {lim.self_top1:.2%})"
    )


def _divergence_verdict(
    div: DivergenceResult | None,
    self_div: DivergenceResult | None,
    tasks: Sequence[TaskVerdict],
    policy: GatePolicy,
    error: str | None = None,
) -> DivergenceVerdict:
    if error is not None:
        return DivergenceVerdict(verdict=Verdict.INCONCLUSIVE, reason=f"not measured: {error}")
    if div is None:
        return DivergenceVerdict(verdict=Verdict.PASS, reason="not measured")
    lim = divergence_limits(policy, self_div)
    kl, top1 = div.kl.point, div.top1.point
    record: dict[str, Any] = {
        "result": div.model_dump(mode="json"),
        "limits": lim,
        "self_divergence": None if self_div is None else self_div.model_dump(mode="json"),
    }
    broken = []
    if lim.ceiling_kl is not None and kl > lim.ceiling_kl:
        broken.append(f"mean KL {kl:.4f} nats is above the hard ceiling {lim.ceiling_kl:.4f}")
    if lim.ceiling_top1 is not None and top1 < lim.ceiling_top1:
        broken.append(
            f"top-1 agreement {top1:.2%} is below the hard ceiling {lim.ceiling_top1:.2%}"
        )
    if broken:
        return DivergenceVerdict(verdict=Verdict.FAIL, reason="; ".join(broken), **record)
    over = []
    if lim.max_kl is not None and kl > lim.max_kl:
        over.append(f"mean KL {kl:.4f} nats exceeds {lim.max_kl:.4f}")
    if lim.min_top1 is not None and top1 < lim.min_top1:
        over.append(f"top-1 agreement {top1:.2%} is below {lim.min_top1:.2%}")
    if not over:
        measured = f"{div.n_prompts} prompts, {div.n_positions} positions"
        return DivergenceVerdict(
            verdict=Verdict.PASS,
            reason=f"KL {kl:.4f} nats, top-1 {top1:.2%} ({measured}); {limits_text(lim)}",
            **record,
        )
    undecided = [t.task for t in tasks if t.verdict is not Verdict.PASS]
    if undecided:
        return DivergenceVerdict(
            verdict=Verdict.FAIL,
            reason=f"{'; '.join(over)}, and {', '.join(undecided)} did not pass "
            f"non-inferiority; {limits_text(lim)}",
            **record,
        )
    return DivergenceVerdict(
        verdict=Verdict.REVIEW,
        reason=f"needs review: {'; '.join(over)} while every task passed non-inferiority; "
        f"{limits_text(lim)}",
        **record,
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
    *,
    self_divergence: DivergenceResult | None = None,
    divergence_error: str | None = None,
    baseline_replicates: int = 1,
    candidate_replicates: int = 1,
) -> GateDecision:
    """Decide whether `candidate` may replace `baseline`; see the module docstring.

    `self_divergence` is the baseline scored against its own divergence reference under
    different batching: the noise floor the divergence limits are calibrated on.
    `divergence_error` is why a divergence the suite asks for was not measured (its
    capture or scoring failed): the divergence check is then inconclusive, with that
    reason, and the task verdicts stand. `*_replicates` is how many passes each side's
    item scores average (`pool_replicates`); they are recorded with the decision.
    """
    policy = policy or GatePolicy()
    if baseline.keys() != candidate.keys():
        raise ValueError(
            f"task sets differ: baseline {sorted(baseline)}, candidate {sorted(candidate)}"
        )
    tasks = [evaluate_task(name, baseline[name], candidate[name], policy) for name in baseline]
    div = _divergence_verdict(divergence, self_divergence, tasks, policy, divergence_error)
    san = _sanity_verdict(sanity, policy)
    decision = worst([t.verdict for t in tasks] + [div.verdict, san.verdict])
    blocked = (
        decision is Verdict.FAIL
        or (decision is Verdict.INCONCLUSIVE and policy.inconclusive_blocks)
        or (decision is Verdict.REVIEW and policy.review_blocks)
    )
    return GateDecision(
        decision=decision,
        blocked=blocked,
        tasks=tasks,
        divergence=div,
        sanity=san,
        policy=policy,
        baseline_replicates=baseline_replicates,
        candidate_replicates=candidate_replicates,
    )
