"""Methodology and provenance footer: everything a reader needs to check or rerun a number."""

from __future__ import annotations

import datetime as dt
from collections.abc import Mapping, Sequence
from functools import cache
from typing import Any, cast

from pydantic import BaseModel, ConfigDict

from loom_bench.metrics.aggregate import ci_rule_summary
from loom_bench.money import Micros
from loom_bench.prices import PriceBook
from loom_bench.provenance import PriceBasis
from loom_bench.records import LoadMode, Market
from loom_bench.registry import Cloud
from loom_bench.report.analyze import ConfigResult, SweepKey
from loom_bench.report.format import describe_slo, load_mode_label, usd
from loom_bench.workloads.profiles import list_profiles, load_profile

ALLOCATION_TEXT = {
    "all_output": "all of the replica's hourly cost is charged to output tokens, so "
    "input tokens have no separate price ($/1M input is n/a, not $0); its $/1M output is "
    "the rank key, while the published $/1M input and output prices use the measured "
    "prefill-time split",
    "all_input": "all of the replica's hourly cost is charged to input tokens, so output "
    "tokens have no separate price ($/1M output is n/a, not $0)",
    "prefill_time": "input tokens are charged the share of the replica's time spent "
    "prefilling prompts, output tokens the rest (see the input/output split)",
}

# How the headline $/1M input and $/1M output are split; every leaderboard states it.
SPLIT_TEXT = (
    "Headline $/1M input and $/1M output use the prefill_time split, measured at the "
    "goodput point: input tokens pay for the share φ of the replica's time spent "
    "prefilling prompts, output tokens for the rest (decode steps, and any idle headroom "
    "the SLO needs). φ is the mean number of requests in their prefill phase, request "
    "rate × mean TTFT (Little's law), per repetition, combined like a throughput "
    "(geometric mean, log-t CI) and capped at 1. Input = φ × hourly price / input tok/s; "
    "output = (1 − φ) × hourly price / output tok/s; at the measured mix they bill "
    "exactly the replica's hourly price. TTFT includes queueing and the first decode step "
    "and concurrent prefills are each counted, so φ tends to overstate prefill time (input "
    "high, output low); idle headroom is charged to output (output high). The price CIs "
    "combine the φ and throughput CIs at the ends that push each price the same way, an "
    "outer bound. When φ is high (prefill-heavy traffic, e.g. long shared prompts or chat "
    "at high load) its CI can reach 1, the cap: the measurements then cannot rule out that "
    "prefill takes all of the replica's time, so the output price's low end is $0 and is "
    "marked as such. That $0 is a property of the split, not a measured free token; the "
    "point estimate and the high end are unaffected. Blended $/1M is the hourly price over "
    "all tokens at the workload's own input:output mix and needs no split. The ranking "
    "still uses the declared cost allocation, shown in its own column."
)


class PriceSource(BaseModel):
    hourly_micros: Micros | None  # on-demand: the price results are ranked by
    basis: str
    as_run: str
    urls: list[str]
    last_checked: dt.date | None
    note: str | None


class DatasetRef(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str | None
    source: str | None
    revision: str | None
    license: str | None
    license_url: str | None

    @property
    def text(self) -> str:
        parts = [self.name or "unnamed dataset"]
        if self.revision:
            parts.append(f"@ {self.revision}")
        parts.append(f"(license: {self.license or 'not recorded'})")
        return " ".join(parts)


@cache
def _profile_datasets() -> dict[tuple[str, str | None], DatasetRef]:
    """The dataset of every shipped workload profile, by (source, revision)."""
    out: dict[tuple[str, str | None], DatasetRef] = {}
    for name in list_profiles():
        try:
            profile = load_profile(name)
        except (OSError, ValueError):
            continue
        d = profile.dataset
        if d is not None:
            out[(d.source, d.revision)] = DatasetRef.model_validate(d.model_dump())
    return out


def public_dataset(recorded: DatasetRef) -> DatasetRef:
    """The dataset as published: its licence note as the workload profile states it now.

    A run's provenance keeps the note recorded at the time; an older one can carry text
    meant for review rather than for readers (chat-sharegpt's "Verify before
    publishing."). The same dataset (source and revision) takes the profile's current
    note; a dataset no shipped profile names is shown as recorded.
    """
    current = _profile_datasets().get((recorded.source or "", recorded.revision))
    if current is None:
        return recorded
    return recorded.model_copy(
        update={"license": current.license, "license_url": current.license_url}
    )


class ConfigProvenance(BaseModel):
    """Provenance of one result: a config on one workload and load mode."""

    name: str
    config_hash: str
    workload: str
    load_mode: LoadMode
    label: str
    engine: str
    image: str | None
    image_digest: str | None
    model: str
    gpu: str
    location: str
    cuda_version: str | None
    driver_version: str | None
    bench_version: str | None
    git_shas: list[str]
    price: PriceSource
    runs: int
    provenance_digests: list[str]
    reproduce: str

    @property
    def key(self) -> SweepKey:
        return SweepKey(self.config_hash, self.workload, self.load_mode)

    @property
    def title(self) -> str:
        return f"{self.name} · {self.workload}, {self.load_mode.value.replace('_', ' ')}"


class CiMethodRow(BaseModel):
    metrics: str  # metric family
    method: str


class Methodology(BaseModel):
    slos: list[str]
    allocations: list[str]
    load_modes: list[str]
    content_kinds: list[str]
    datasets: list[DatasetRef]
    repetitions: str
    ci_method: str
    ci_methods: list[CiMethodRow]
    price_book_last_checked: dt.date | None
    configs: list[ConfigProvenance]
    split: str = SPLIT_TEXT


def reproduce_command(run_id: str) -> str:
    return f"bench reproduce {run_id}"


def describe_allocation(allocation: str) -> str:
    if allocation in ALLOCATION_TEXT:
        return f"{allocation}: {ALLOCATION_TEXT[allocation]}"
    return (
        f"{allocation}: one output token costs as much as `output_input_ratio` input tokens; "
        "prices at these rates bill exactly the replica's hourly cost at the measured token mix"
    )


def _section(prov: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = prov.get(name)
    return value if isinstance(value, Mapping) else {}


def per_hour(m: Micros | None) -> str:
    return "n/a" if m is None else f"{usd(m)}/h"


# The storage volume each cloud's replica pays for, as its price book entry prices it.
STORAGE_NAMES = {"aws": "EBS gp3 volume", "runpod": "container disk"}


def storage_name(cloud: str | None) -> str:
    return STORAGE_NAMES.get(cloud or "", "storage volume")


def _storage_text(gb: int | None, cloud: str | None) -> str:
    return f" + {gb} GB {storage_name(cloud)}" if gb else ""


def as_run_text(result: ConfigResult) -> str:
    """How the as-run price recorded at launch was obtained."""
    doc = result.provenance.get("price_basis")
    if not isinstance(doc, Mapping):
        return "not recorded"
    b = PriceBasis.model_validate(doc)
    hourly = per_hour(result.prices.as_run)
    host = f"{b.market.value} host"
    storage = _storage_text(b.storage_gb, result.provenance.get("cloud"))
    if b.source == "observed_spot":
        at = b.observed_at.isoformat() if b.observed_at else "time not recorded"
        return (
            f"{hourly}, {host}: spot ${b.spot_price_usd}/h observed at launch in "
            f"{b.availability_zone or 'an unrecorded AZ'} ({at}){storage}"
        )
    if b.source == "prices_yaml":
        return f"{hourly}, {host}: price-book on-demand price{storage}"
    if b.source == "experiment":
        return f"{hourly}, {host}: declared by the experiment (provider.hourly_price)"
    if b.source == "observed_api":
        at = b.observed_at.isoformat() if b.observed_at else "time not recorded"
        cloud = result.provenance.get("cloud") or "the provider"
        return (
            f"{hourly}, {host}: {cloud} API price observed at launch in "
            f"{_zone_text(b.availability_zone)} ({at}){storage}"
        )
    return f"unknown, {host}: no spot price was observed for its availability zone at launch"


def _zone_text(zone: str | None) -> str:
    """An AZ or datacenter id; `location:<code>` is a country, so it is said as one."""
    if not zone:
        return "an unrecorded datacenter"
    if zone.startswith("location:"):
        return f"location {zone.removeprefix('location:')} (datacenter not reported)"
    return zone


def price_source(result: ConfigResult, price_book: PriceBook | None) -> PriceSource:
    """Where the on-demand (ranking), spot and committed prices of a result come from."""
    prov = result.provenance
    p = result.prices
    market = prov.get("market")
    as_run = as_run_text(result)
    if market is None or Market(market) is Market.LOCAL:
        basis = (
            f"{usd(p.on_demand)}/h declared by the experiment (provider.hourly_price); "
            "not a cloud price"
            if p.on_demand is not None
            else p.missing or "no price"
        )
        return PriceSource(
            hourly_micros=p.on_demand,
            basis=basis,
            as_run=as_run,
            urls=[],
            last_checked=None,
            note=None,
        )

    cloud, region = cast(Cloud, prov.get("cloud")), prov.get("region") or ""
    instance = _section(prov, "hardware").get("instance_type") or ""
    where = f"{cloud}/{region} {instance}"
    if p.on_demand is None:
        basis = f"{where}: {p.missing or 'no on-demand price'}"
    else:
        basis = (
            f"{where}{_storage_text(p.storage_gb, cloud)}, from the price book: on-demand "
            f"{per_hour(p.on_demand)}; spot {per_hour(p.spot)}; "
            f"committed 1y {per_hour(p.committed_1y)}"
        )
    if price_book is None:
        return PriceSource(
            hourly_micros=p.on_demand,
            basis=basis,
            as_run=as_run,
            urls=[],
            last_checked=None,
            note="price book not supplied to the report",
        )
    try:
        entry = price_book.instance(cloud, region, instance)
    except KeyError as e:
        return PriceSource(
            hourly_micros=p.on_demand,
            basis=basis,
            as_run=as_run,
            urls=[],
            last_checked=None,
            note=str(e),
        )
    urls = [str(u) for u in entry.sources]
    storage = price_book.region(cloud, region).storage
    if p.storage_gb and storage is not None:
        urls += [str(u) for u in storage.sources if str(u) not in urls]
    notes = []
    if entry.spot_per_hour is not None:
        notes.append("the spot price is an indicative average and moves hourly")
    if not entry.verified:
        notes.append(f"unverified price: {entry.note}")
    return PriceSource(
        hourly_micros=p.on_demand,
        basis=basis,
        as_run=as_run,
        urls=urls,
        last_checked=entry.last_checked,
        note="; ".join(notes) or None,
    )


def _config_provenance(result: ConfigResult, price_book: PriceBook | None) -> ConfigProvenance:
    prov = result.provenance
    engine = _section(prov, "engine")
    hardware = _section(prov, "hardware")
    location = " / ".join(str(p) for p in (prov.get("cloud"), prov.get("region")) if p) or "local"
    if hardware.get("instance_type"):
        location += f" · {hardware['instance_type']}"
    if prov.get("market"):
        location += f" · {prov['market']}"
    engine_text = " ".join(p for p in (engine.get("name"), engine.get("version")) if p)
    model_text = result.model_repo or "unknown model"
    if result.model_revision:
        model_text += f" @ {result.model_revision}"
    return ConfigProvenance(
        name=result.name,
        config_hash=result.config_hash,
        workload=result.workload,
        load_mode=result.load_mode,
        label=result.label.text,
        engine=engine_text or "unknown",
        image=engine.get("image"),
        image_digest=engine.get("image_digest"),
        model=model_text,
        gpu=f"{hardware.get('gpu_count') or '?'}×{hardware.get('gpu_type') or 'unknown GPU'}",
        location=location,
        cuda_version=prov.get("cuda_version"),
        driver_version=prov.get("driver_version"),
        bench_version=prov.get("bench_version"),
        git_shas=result.git_shas,
        price=price_source(result, price_book),
        runs=len(result.run_ids),
        provenance_digests=result.provenance_digests,
        reproduce=reproduce_command(result.reproduce_run_id()),
    )


def _distinct[T](values: Sequence[T]) -> list[T]:
    return list(dict.fromkeys(values))


def methodology(results: Sequence[ConfigResult], price_book: PriceBook | None) -> Methodology:
    reps = [p.aggregate.n_runs for r in results for p in r.points]
    confidences = _distinct([p.aggregate.confidence for r in results for p in r.points])
    datasets = _distinct(
        [
            public_dataset(DatasetRef.model_validate(dict(_section(r.provenance, "dataset"))))
            for r in results
            # Synthetic workloads record an empty dataset section: no dataset to name.
            if _section(r.provenance, "dataset").get("name")
            or _section(r.provenance, "dataset").get("source")
        ]
    )
    if reps:
        lo, hi = min(reps), max(reps)
        repetitions = f"{lo} per load point" if lo == hi else f"{lo}–{hi} per load point"
    else:
        repetitions = "none"
    conf = ", ".join(f"{c:.0%}" for c in confidences) or "95%"
    ci_method = (
        "Repetitions of a load point are combined per metric, with a two-sided "
        f"{conf} Student-t interval on the scale that fits the metric. Latencies, "
        "throughputs and rates are strictly positive and right-skewed: the value shown is "
        "the geometric mean and the interval is the t-interval of ln(value), exponentiated "
        "(geometric mean ×/÷ a factor), so both bounds are positive. Proportions such as "
        "error rate and SLO attainment use the arithmetic mean of the per-run proportions "
        "with the t-interval clipped to [0, 1]; it measures run-to-run variation, not "
        "binomial sampling, so it is [0, 0] when no repetition saw an error. Counts use the "
        "arithmetic t-interval clipped at 0, as does a positive metric when a repetition "
        "measured 0. A load meets the SLO only if the CI upper bound of every target is "
        "within it; goodput is the highest passing load below the first failing one. Cost "
        "CI bounds come from the goodput throughput CI (low cost from high throughput), so "
        "they are finite whenever the throughput lower bound is above zero. Results with a "
        "single repetition have no CI and are never trusted."
    )
    return Methodology(
        slos=_distinct([describe_slo(r.goodput.slo) for r in results]),
        allocations=_distinct([describe_allocation(r.allocation) for r in results]),
        load_modes=_distinct([load_mode_label(r.load_mode) for r in results]),
        content_kinds=_distinct([r.content.value if r.content else "unknown" for r in results]),
        datasets=datasets,
        repetitions=repetitions,
        ci_method=ci_method,
        ci_methods=[CiMethodRow(metrics=m, method=t) for m, t in ci_rule_summary()],
        price_book_last_checked=price_book.last_checked if price_book else None,
        configs=[_config_provenance(r, price_book) for r in results],
    )


def _code(value: str | None) -> str:
    return f"`{value}`" if value else "not recorded"


def methodology_markdown(m: Methodology) -> str:
    lines = ["## Methodology and provenance", ""]
    lines += [f"- **SLO:** {slo}" for slo in m.slos]
    lines += [f"- **Load generation:** {mode}" for mode in m.load_modes]
    lines.append(f"- **Content:** {', '.join(m.content_kinds)}")
    for d in m.datasets:
        source = f"; source: {d.source}" if d.source else ""
        lines.append(f"- **Dataset:** {d.text}{source}")
    lines.append(f"- **Repetitions:** {m.repetitions}")
    lines.append(f"- **Confidence intervals:** {m.ci_method}")
    lines += [f"  - {row.metrics}: {row.method}" for row in m.ci_methods]
    lines += [f"- **Cost allocation (ranking):** {a}" for a in m.allocations]
    lines.append(f"- **Input/output split (headline):** {m.split}")
    checked = m.price_book_last_checked.isoformat() if m.price_book_last_checked else "n/a"
    lines.append(f"- **Price book last checked:** {checked}")
    for c in m.configs:
        p = c.price
        price = p.basis
        if p.urls:
            price += f"; source: {', '.join(p.urls)}"
        if p.last_checked:
            price += f"; last checked {p.last_checked.isoformat()}"
        if p.note:
            price += f"; {p.note}"
        price += f". As run: {p.as_run}"
        shas = ", ".join(_code(s) for s in c.git_shas) or "not recorded"
        lines += [
            "",
            f"### {c.title} (`{c.config_hash}`)",
            "",
            f"- **Config:** {c.label}",
            f"- **Engine:** {c.engine}; image {_code(c.image)}; digest {_code(c.image_digest)}",
            f"- **Model:** {c.model}",
            f"- **Hardware:** {c.gpu} · {c.location}; CUDA {c.cuda_version or 'not recorded'}, "
            f"driver {c.driver_version or 'not recorded'}",
            f"- **Code:** commit {shas}; loom-bench {c.bench_version or 'not recorded'}",
            f"- **Price:** {price}",
            f"- **Runs:** {c.runs}; provenance digests: "
            + ", ".join(_code(d) for d in c.provenance_digests),
            f"- **Reproduce:** `{c.reproduce}`",
        ]
    return "\n".join(lines) + "\n"
