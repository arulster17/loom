"""Methodology and provenance footer: everything a reader needs to check or rerun a number."""

from __future__ import annotations

import datetime as dt
from collections.abc import Mapping, Sequence
from typing import Any, cast

from pydantic import BaseModel, ConfigDict

from loom_bench.money import Micros
from loom_bench.prices import PriceBook
from loom_bench.records import Market
from loom_bench.registry import Cloud
from loom_bench.report.analyze import ConfigResult
from loom_bench.report.format import describe_slo, load_mode_label, usd

ALLOCATION_TEXT = {
    "all_output": "all of the replica's hourly cost is charged to output tokens "
    "(input tokens are free); $/1M output is the headline number",
    "all_input": "all of the replica's hourly cost is charged to input tokens "
    "(output tokens are free)",
}


class PriceSource(BaseModel):
    hourly_micros: Micros | None
    basis: str
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
    name: str
    config_hash: str
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


class Methodology(BaseModel):
    slos: list[str]
    allocations: list[str]
    load_modes: list[str]
    content_kinds: list[str]
    datasets: list[DatasetRef]
    repetitions: str
    ci_method: str
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


def price_source(result: ConfigResult, price_book: PriceBook | None) -> PriceSource:
    prov = result.provenance
    market = prov.get("market")
    hourly = result.hourly_micros
    if market is None or Market(market) is Market.LOCAL:
        basis = (
            "not billed (market local, no hourly_micros recorded)"
            if hourly is None
            else f"{usd(hourly)}/h recorded as hourly_micros in the provenance (not a cloud price)"
        )
        return PriceSource(hourly_micros=hourly, basis=basis, urls=[], last_checked=None, note=None)

    cloud, region = prov.get("cloud"), prov.get("region")
    instance = _section(prov, "hardware").get("instance_type")
    basis = f"{cloud}/{region} {instance} {market}: {usd(hourly)}/h"
    if price_book is None:
        return PriceSource(
            hourly_micros=hourly,
            basis=basis,
            urls=[],
            last_checked=None,
            note="price book not supplied to the report",
        )
    try:
        entry = price_book.instance(cast(Cloud, cloud), region or "", instance or "")
    except KeyError as e:
        return PriceSource(
            hourly_micros=hourly, basis=basis, urls=[], last_checked=None, note=str(e)
        )
    notes = []
    if Market(market) is Market.SPOT:
        notes.append(
            "no spot price recorded; the on-demand price was used"
            if entry.spot_per_hour is None
            else "spot prices are indicative and move hourly"
        )
    if not entry.verified:
        notes.append(f"unverified price: {entry.note}")
    return PriceSource(
        hourly_micros=hourly,
        basis=basis,
        urls=[str(u) for u in entry.sources],
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
        f"Each metric is the mean across repetitions with a two-sided Student-t {conf} "
        "confidence interval. A load meets the SLO only if the CI upper bound of every target "
        "is within it; goodput is the highest passing load below the first failing one. Cost "
        "CI bounds come from the goodput throughput CI (low cost from high throughput). "
        "Results with a single repetition have no CI and are never trusted."
    )
    return Methodology(
        slos=_distinct([describe_slo(r.goodput.slo) for r in results]),
        allocations=_distinct([describe_allocation(r.allocation) for r in results]),
        load_modes=_distinct([load_mode_label(r.load_mode) for r in results]),
        content_kinds=_distinct([r.content.value if r.content else "unknown" for r in results]),
        datasets=datasets,
        repetitions=repetitions,
        ci_method=ci_method,
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
        shas = ", ".join(_code(s) for s in c.git_shas) or "not recorded"
        lines += [
            "",
            f"### {c.name} (`{c.config_hash}`)",
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
