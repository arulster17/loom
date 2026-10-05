"""Dry-run planner: hosts, per-step time estimates, estimated spend against the caps.

Cells with identical hardware share one host; after the first (cold) start the
host switches configs with warm restarts. Every timing assumption is in
`AWS_TIMING`, `LOCAL_TIMING`, `MOCK_*`, `ASSUMED_*` and `EVAL_*` below.
`bench run` refuses to start when `Plan.refusals` is non-empty.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from decimal import Decimal
from fractions import Fraction

from pydantic import ValidationError

from loom_bench.budget import Caps
from loom_bench.experiment import (
    AwsEc2ProviderSpec,
    Cell,
    Experiment,
    LoadSpec,
    MockProviderSpec,
    WorkloadEntry,
)
from loom_bench.money import Micros, cost_for_seconds, format_usd
from loom_bench.prices import HOURS_PER_MONTH, PriceBook
from loom_bench.quality.divergence import load_prompts
from loom_bench.quality.suite import SuiteTask
from loom_bench.records import LoadMode, Market
from loom_bench.workloads import WorkloadProfile


@dataclass(frozen=True)
class Timing:
    boot_s: float  # provision call -> instance running with Docker and the agent up
    image_pull_s: float  # pull and extract one engine image
    download_bytes_per_s: float  # model weights, HF hub -> host disk
    load_bytes_per_s: float  # model weights, host disk -> GPU memory
    engine_init_s: float  # CUDA graph capture, compilation, KV-cache allocation
    run_overhead_s: float  # per run, beyond duration and drain: job upload, result download
    teardown_s: float  # terminate call -> billing stops
    eval_setup_s: float  # once per host: eval harness installed into the client environment
    eval_job_s: float  # per eval job, beyond its requests: staging, container start, upload


# EC2 with a GPU DLAMI and Docker over SSM. Bandwidths are deliberately low: HF
# downloads often run well below the NIC rate and engines load safetensors from
# disk slower than the disk's peak. Weights are cached on the host between warm
# restarts, so only a different checkpoint is downloaded again.
AWS_TIMING = Timing(
    boot_s=180.0,
    image_pull_s=300.0,
    download_bytes_per_s=150e6,
    load_bytes_per_s=400e6,
    engine_init_s=240.0,
    run_overhead_s=30.0,
    teardown_s=90.0,
    eval_setup_s=300.0,  # loom-bench[lmeval]: lm-eval, torch and transformers wheels
    eval_job_s=60.0,
)
LOCAL_TIMING = Timing(
    boot_s=0.0,
    image_pull_s=0.0,
    download_bytes_per_s=math.inf,
    load_bytes_per_s=math.inf,
    engine_init_s=0.0,
    run_overhead_s=1.0,
    teardown_s=0.0,
    eval_setup_s=0.0,
    eval_job_s=1.0,
)
# The mock starts a uvicorn thread, then waits startup_delay_s * time_scale.
MOCK_SERVER_START_S = 0.5
MOCK_RUN_OVERHEAD_S = 0.3
MOCK_EVAL_JOB_S = 0.5

# Closed-loop runs bounded by request count (not duration) on a real engine assume
# these per-request costs: a modest prefill rate and decode at the 50 ms TPOT SLO.
ASSUMED_PREFILL_TOKENS_PER_S = 4000.0
ASSUMED_DECODE_S_PER_TOKEN = 0.05
# Shape used when a profile gives no length (datasets, traces without clamps).
DEFAULT_INPUT_TOKENS = 1024
DEFAULT_OUTPUT_TOKENS = 512

# Quality evals. An eval job takes, summed over its tasks,
#   items x EVAL_ITEM_S[kind] / concurrency
# plus EVAL_HARNESS_TASK_S per lm_eval task, the divergence prompts and the
# provider's per-job overhead. EVAL_ITEM_S is how long one item holds one of the
# task's concurrent request slots on a real engine decoding at the 50 ms TPOT SLO
# (ASSUMED_DECODE_S_PER_TOKEN) with every slot busy; concurrency is the lm_eval
# task's num_concurrent, else EVAL_CONCURRENCY. The mock scales item time by its
# time_scale.
EVAL_ITEM_S: dict[str, float] = {
    "lm_eval": 24.0,  # MMLU-Pro / GSM8K chain of thought, IFEval: ~400 tokens + few-shot prefill
    "code_exec": 16.0,  # ~300-token program, then the sandboxed tests
    "needle": 40.0,  # 8k-28k-token prompts: prefill-bound while every slot holds one
    "tool_calling": 6.0,  # one call, ~60 tokens
    "json_schema": 8.0,  # one object, ~100 tokens, under a grammar
    "toy_arithmetic": 2.0,  # one short sentence
}
EVAL_DIVERGENCE_PROMPT_S = 4.0  # 64-token greedy continuation, then one echo scoring request
EVAL_HARNESS_TASK_S = 60.0  # per lm_eval task: harness start-up and dataset download
EVAL_CONCURRENCY = 16  # EvalJob.concurrency the runner sets
LMEVAL_DEFAULT_CONCURRENCY = 16  # LmEvalParams.num_concurrent default


def profile_shape(p: WorkloadProfile) -> tuple[int, int]:
    """(input, output) tokens per request the estimate assumes for a profile."""
    inp = (
        getattr(p, "input_len", None)
        or getattr(p, "context_len", None)
        or getattr(p, "max_input_len", None)
        or DEFAULT_INPUT_TOKENS
    )
    out = getattr(p, "output_len", None) or getattr(p, "max_output_len", None)
    return int(inp), int(out or DEFAULT_OUTPUT_TOKENS)


def load_points(load: LoadSpec) -> int:
    """Load points a workload runs at most (a search may stop earlier)."""
    return len(load.values) if load.values is not None else load.search.max_points  # type: ignore[union-attr]


class Estimator:
    """Wall-time estimates for each step; shared by the planner and the budget checks."""

    def __init__(self, exp: Experiment) -> None:
        self.exp = exp
        self.suite = exp.quality.load() if exp.quality is not None else None
        p = exp.provider
        self.timing = (
            AWS_TIMING
            if isinstance(p, AwsEc2ProviderSpec)
            else LOCAL_TIMING
            if p.kind == "local"
            else None
        )

    def _mock_start_s(self, cell: Cell) -> float:
        assert cell.mock is not None
        return MOCK_SERVER_START_S + cell.mock.startup_delay_s * cell.mock.time_scale

    def cold_start_s(self, cell: Cell) -> float:
        if self.timing is None:
            return self._mock_start_s(cell)
        t, size = self.timing, cell.spec.hf.size_bytes
        return (
            t.boot_s
            + t.image_pull_s
            + size / t.download_bytes_per_s
            + size / t.load_bytes_per_s
            + t.engine_init_s
        )

    def warm_start_s(self, prev: Cell | None, cell: Cell) -> float:
        if self.timing is None:
            return self._mock_start_s(cell)
        t, size = self.timing, cell.spec.hf.size_bytes
        seconds = size / t.load_bytes_per_s + t.engine_init_s
        if prev is None or prev.launch.image != cell.launch.image:
            seconds += t.image_pull_s
        if prev is None or (prev.spec.hf.repo, prev.spec.hf.revision) != (
            cell.spec.hf.repo,
            cell.spec.hf.revision,
        ):
            seconds += size / t.download_bytes_per_s
        return seconds

    def _request_s(self, cell: Cell, profile: WorkloadProfile, concurrency: float) -> float:
        inp, out = profile_shape(profile)
        if cell.mock is not None:
            m = cell.mock
            ms = (
                m.prefill_ms_per_token * inp
                + (m.step_base_ms + m.decode_ms_per_seq * concurrency) * out
            )
            return ms * m.time_scale / 1000
        return inp / ASSUMED_PREFILL_TOKENS_PER_S + out * ASSUMED_DECODE_S_PER_TOKEN

    def run_overhead_s(self) -> float:
        return MOCK_RUN_OVERHEAD_S if self.timing is None else self.timing.run_overhead_s

    def run_s(self, cell: Cell, load: LoadSpec, profile: WorkloadProfile, value: float) -> float:
        if load.mode is LoadMode.OPEN_LOOP:
            # Worst case: an overloaded point waits out its whole drain timeout.
            assert load.duration_s is not None
            return load.duration_s + load.drain_timeout_s + self.run_overhead_s()
        if load.duration_s is not None:
            return load.duration_s + self.run_overhead_s()
        # Closed loop by count: rounds of `value` concurrent requests. Engine-side
        # queueing (e.g. max_num_seqs below the concurrency) is not modelled.
        assert load.num_requests is not None
        total = (load.warmup_requests or 0) + load.num_requests
        rounds = math.ceil(total / value)
        return rounds * self._request_s(cell, profile, value) + self.run_overhead_s()

    def max_load(self, load: LoadSpec) -> float:
        return max(load.values) if load.values is not None else load.search.hi  # type: ignore[union-attr]

    def workload_s(self, cell: Cell, entry: WorkloadEntry, profile: WorkloadProfile) -> float:
        """Upper bound for one workload's sweep: each point priced like the heaviest."""
        load = entry.load
        if load.values is not None:
            per_point = [self.run_s(cell, load, profile, v) for v in load.values]
        else:
            per_point = [self.run_s(cell, load, profile, self.max_load(load))] * load_points(load)
        return sum(per_point) * self.exp.repetitions

    def eval_tasks(self) -> list[SuiteTask]:
        if self.suite is None or self.exp.quality is None:
            return []
        return self.suite.select(self.exp.quality.subset)

    def uncounted_tasks(self) -> list[str]:
        """Eval tasks whose item count is unknown before they run (lm_eval without `items`)."""
        return [t.name for t in self.eval_tasks() if t.planned_items() is None]

    def eval_s(self, cell: Cell) -> float:
        """One cell's eval job: its suite tasks, its divergence half and the job overhead."""
        if self.suite is None:
            return 0.0
        busy = 0.0
        for t in self.eval_tasks():
            concurrency = (
                t.params.get("num_concurrent", LMEVAL_DEFAULT_CONCURRENCY)
                if t.kind == "lm_eval"
                else EVAL_CONCURRENCY
            )
            busy += (t.planned_items() or 0) * EVAL_ITEM_S[t.kind] / concurrency
        div = self.suite.divergence
        if div is not None:
            busy += (
                (div.prompts or len(load_prompts())) * EVAL_DIVERGENCE_PROMPT_S / EVAL_CONCURRENCY
            )
        if cell.mock is not None:
            return busy * cell.mock.time_scale + MOCK_EVAL_JOB_S
        assert self.timing is not None
        harness = EVAL_HARNESS_TASK_S * sum(t.kind == "lm_eval" for t in self.eval_tasks())
        return busy + harness + self.timing.eval_job_s

    def eval_setup_s(self) -> float:
        return 0.0 if self.suite is None or self.timing is None else self.timing.eval_setup_s

    def teardown_s(self) -> float:
        return 0.0 if self.timing is None else self.timing.teardown_s


@dataclass
class Step:
    kind: str  # cold_start | eval_setup | warm_start | workload | eval | teardown
    label: str
    seconds: float


@dataclass
class HostPlan:
    key: str
    provider: str
    instance_type: str | None
    market: str
    hourly_micros: Micros
    price_notes: list[str]
    ttl_s: int
    cells: list[str] = field(default_factory=list)
    steps: list[Step] = field(default_factory=list)

    @property
    def seconds(self) -> float:
        return sum(s.seconds for s in self.steps)

    @property
    def cost_micros(self) -> Micros:
        return cost_for_seconds(self.hourly_micros, self.seconds)

    @property
    def ttl_worst_micros(self) -> Micros:
        return cost_for_seconds(self.hourly_micros, self.ttl_s)


@dataclass
class Plan:
    experiment: str
    provider: str
    hosts: list[HostPlan]
    n_cells: int
    n_runs_max: int
    caps: Caps
    refusals: list[str]
    notes: list[str]

    @property
    def total_seconds(self) -> float:
        return sum(h.seconds for h in self.hosts)

    @property
    def total_micros(self) -> Micros:
        return sum(h.cost_micros for h in self.hosts)

    @property
    def ttl_worst_micros(self) -> Micros:
        return sum(h.ttl_worst_micros for h in self.hosts)

    @property
    def ok(self) -> bool:
        return not self.refusals


def host_price(exp: Experiment, cell: Cell, prices: PriceBook) -> tuple[Micros, str, list[str]]:
    """(hourly micros, market, notes) for the host a cell runs on."""
    p = exp.provider
    if isinstance(p, MockProviderSpec):
        if p.hourly_price is None:
            return 0, Market.LOCAL.value, ["unpriced mock: no hourly_price, so no cost at SLO"]
        return p.hourly_price, Market.LOCAL.value, ["simulated price (mock)"]
    if not isinstance(p, AwsEc2ProviderSpec):
        return 0, Market.LOCAL.value, ["not billed (local endpoint)"]
    hw = cell.hardware
    terms = aws_accrual_terms()
    notes: list[str] = []
    instance = prices.instance("aws", hw["region"], hw["instance_type"])
    if not instance.verified:
        notes.append(f"unverified instance price: {instance.note}")
    storage = prices.region("aws", hw["region"]).storage
    if storage is None:
        raise KeyError(f"no storage price for aws/{hw['region']}")
    if not storage.verified:
        notes.append(f"unverified storage price included: {storage.note}")
    quote = prices.instance_price(
        "aws", hw["region"], hw["instance_type"], Market(hw["market"]), allow_unverified=True
    )
    per_hour = quote.per_hour
    if quote.market is Market.SPOT:
        # The provider accrues spot at the live price x a safety multiplier; plan the same.
        per_hour = math.ceil(Fraction(per_hour) * Fraction(terms.spot_multiplier))
        notes.append(f"spot price x {terms.spot_multiplier} safety multiplier (as accrued)")
    if quote.spot_fallback:
        notes.append("no spot price recorded; estimated at on-demand")
    volume_gb = max(hw["disk_gb"], terms.root_volume_gb)
    ebs = math.ceil(Fraction(storage.per_gb_month * volume_gb, HOURS_PER_MONTH))
    return per_hour + ebs, quote.market.value, notes


@dataclass(frozen=True)
class AwsAccrualTerms:
    spot_multiplier: Decimal
    root_volume_gb: int
    max_ttl_s: int


def aws_accrual_terms() -> AwsAccrualTerms:
    """The aws_ec2 provider's accrual settings: configured ones, else its defaults."""
    from loom_bench.providers.aws_ec2 import AwsSettings, load_aws_settings

    try:
        s = load_aws_settings()
        return AwsAccrualTerms(s.spot_price_multiplier, s.root_volume_gb, s.max_ttl_s)
    except (ValidationError, FileNotFoundError):
        f = AwsSettings.model_fields
        return AwsAccrualTerms(
            f["spot_price_multiplier"].default,
            f["root_volume_gb"].default,
            f["max_ttl_s"].default,
        )


def _hardware_problems(exp: Experiment, cell: Cell, prices: PriceBook) -> list[str]:
    if not isinstance(exp.provider, AwsEc2ProviderSpec):
        return []
    hw = cell.hardware
    it = prices.instance("aws", hw["region"], hw["instance_type"])
    out = []
    if it.gpu != cell.spec.hardware.gpu:
        out.append(
            f"{cell.key}: {hw['instance_type']} has {it.gpu}, spec wants {cell.spec.hardware.gpu}"
        )
    if it.gpu_count < cell.gpus:
        out.append(f"{cell.key}: {hw['instance_type']} has {it.gpu_count} GPUs, needs {cell.gpus}")
    return out


def build_plan(
    exp: Experiment,
    cells: list[Cell],
    *,
    prices: PriceBook,
    caps: Caps,
) -> Plan:
    est = Estimator(exp)
    profiles = [(w, w.resolve()) for w in exp.workloads]
    ttl_s = exp.budget.ttl_s
    refusals = caps.violations()
    if isinstance(exp.provider, AwsEc2ProviderSpec):
        max_ttl = aws_accrual_terms().max_ttl_s
        if ttl_s > max_ttl:
            refusals.append(f"budget.ttl_minutes is above the AWS host limit of {max_ttl / 60:g}")
    if est.uncounted_tasks():
        refusals.append(
            f"quality tasks {est.uncounted_tasks()} have no item count to estimate their "
            "time from; set `items` on them in the suite"
        )
    notes: list[str] = []
    hosts: dict[str, HostPlan] = {}
    prev: dict[str, Cell] = {}
    n_runs = 0

    for cell in cells:
        refusals += _hardware_problems(exp, cell, prices)
        host = hosts.get(cell.host_key)
        if host is None:
            hourly, market, price_notes = host_price(exp, cell, prices)
            host = hosts[cell.host_key] = HostPlan(
                key=cell.host_key,
                provider=exp.provider.kind,
                instance_type=cell.hardware.get("instance_type"),
                market=market,
                hourly_micros=hourly,
                price_notes=price_notes,
                ttl_s=ttl_s,
            )
            host.steps.append(Step("cold_start", cell.key, est.cold_start_s(cell)))
            if est.eval_setup_s():
                host.steps.append(Step("eval_setup", host.key, est.eval_setup_s()))
        else:
            host.steps.append(
                Step("warm_start", cell.key, est.warm_start_s(prev[cell.host_key], cell))
            )
        prev[cell.host_key] = cell
        host.cells.append(cell.key)
        for entry, profile in profiles:
            label = (
                f"{cell.key} / {entry.name}: {load_points(entry.load)} pts x {exp.repetitions} reps"
            )
            host.steps.append(Step("workload", label, est.workload_s(cell, entry, profile)))
            n_runs += load_points(entry.load) * exp.repetitions
        if exp.quality is not None:
            q = exp.quality
            label = f"{cell.key} / {q.suite}" + (f" [{q.subset}]" if q.subset else "")
            host.steps.append(Step("eval", label, est.eval_s(cell)))

    for host in hosts.values():
        host.steps.append(Step("teardown", host.key, est.teardown_s()))
        notes += [f"{host.key}: {n}" for n in host.price_notes]
        if host.seconds > host.ttl_s:
            refusals.append(
                f"{host.key}: estimated {host.seconds / 60:.1f} min exceeds the host TTL of "
                f"{exp.budget.ttl_minutes:g} min (the host would self-terminate mid-run)"
            )

    plan = Plan(
        experiment=exp.name,
        provider=exp.provider.kind,
        hosts=list(hosts.values()),
        n_cells=len(cells),
        n_runs_max=n_runs,
        caps=caps,
        refusals=refusals,
        notes=notes,
    )
    cap = caps.effective
    if plan.total_micros > cap:
        refusals.append(
            f"estimated spend {format_usd(plan.total_micros, 2)} exceeds the effective cap "
            f"{format_usd(cap, 2)}"
        )
    if plan.ttl_worst_micros > cap:
        refusals.append(
            f"worst case if the runner dies (every host lives to its TTL) is "
            f"{format_usd(plan.ttl_worst_micros, 2)}, above the effective cap "
            f"{format_usd(cap, 2)}; lower budget.ttl_minutes"
        )
    if not exp.trusted:
        notes.append("single repetition: results are flagged untrusted")
    return plan
