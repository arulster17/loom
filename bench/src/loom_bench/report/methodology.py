"""Methodology and provenance footer: everything a reader needs to check or rerun a number."""

from __future__ import annotations

import datetime as dt
from collections.abc import Mapping, Sequence
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

ALLOCATION_TEXT = {
    "all_output": "all of the replica's hourly cost is charged to output tokens, so "
    "input tokens have no separate price ($/1M input is n/a, not $0); $/1M output is the "
    "headline number",
    "all_input": "all of the replica's hourly cost is charged to input tokens, so output "
    "tokens have no separate price ($/1M output is n/a, not $0)",
}


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


def _storage_text(gb: int | None) -> str:
    return f" + {gb} GB block storage" if gb else ""


def as_run_text(result: ConfigResult) -> str:
    """How the as-run price recorded at launch was obtained."""
    doc = result.provenance.get("price_basis")
    if not isinstance(doc, Mapping):
        return "not recorded"
    b = PriceBasis.model_validate(doc)
    hourly = per_hour(result.prices.as_run)
    host = f"{b.market.value} host"
    if b.source == "observed_spot":
        at = b.observed_at.isoformat() if b.observed_at else "time not recorded"
        return (
            f"{hourly}, {host}: spot ${b.spot_price_usd}/h observed at launch in "
            f"{b.availability_zone or 'an unrecorded AZ'} ({at}){_storage_text(b.storage_gb)}"
        )
    if b.source == "prices_yaml":
        return f"{hourly}, {host}: price-book on-demand price{_storage_text(b.storage_gb)}"
    if b.source == "experiment":
        return f"{hourly}, {host}: declared by the experiment (provider.hourly_price)"
    return f"unknown, {host}: no spot price was observed for its availability zone at launch"


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
            f"{where}{_storage_text(p.storage_gb)}, from the price book: on-demand "
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
            DatasetRef.model_validate(dict(_section(r.provenance, "dataset")))
            for r in results
            if _section(r.provenance, "dataset")
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
    lines += [f"- **Cost allocation:** {a}" for a in m.allocations]
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
