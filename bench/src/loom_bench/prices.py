"""Cloud prices (`bench/prices.yaml`) and competitor list prices (`bench/competitors.yaml`).

Both files are maintained by hand from public price pages; every entry carries its
source URL and the date it was checked. All money is integer micro-dollars.
"""

from __future__ import annotations

import datetime as dt
from fractions import Fraction
from pathlib import Path
from typing import Annotated, Literal, NamedTuple, Self

from pydantic import Field, HttpUrl, model_validator

from loom_bench.money import Micros, round_half_up
from loom_bench.records import Market
from loom_bench.registry import (
    REPO_ROOT,
    Cloud,
    MicrosField,
    NonEmptyStr,
    Quantization,
    StrictModel,
    read_yaml,
)

DEFAULT_PRICES_YAML = REPO_ROOT / "bench" / "prices.yaml"
DEFAULT_COMPETITORS_YAML = REPO_ROOT / "bench" / "competitors.yaml"

# AWS converts GB-month storage prices to hourly at 730 hours per month.
HOURS_PER_MONTH = 730


class UnverifiedPriceError(ValueError):
    """Raised when cost math would use a price marked `verified: false`."""


class Quote(NamedTuple):
    per_hour: Micros
    market: Market  # market actually priced
    spot_fallback: bool  # spot was asked for but has no price, so on-demand was used


class _Sourced(StrictModel):
    sources: Annotated[list[HttpUrl], Field(min_length=1)]
    last_checked: dt.date
    verified: bool = True
    note: NonEmptyStr | None = None

    @model_validator(mode="after")
    def _unverified_needs_note(self) -> Self:
        if not self.verified and self.note is None:
            raise ValueError("verified: false requires a note saying what to check")
        return self


class InstanceType(_Sourced):
    gpu: NonEmptyStr
    gpu_count: Annotated[int, Field(ge=1)]
    gpu_memory_gb: Annotated[int, Field(ge=1)]
    vcpus: Annotated[int, Field(ge=1)] | None = None
    memory_gib: Annotated[int, Field(ge=1)] | None = None
    local_nvme_gb: Annotated[int, Field(ge=0)] | None = None
    on_demand_per_hour: MicrosField
    spot_per_hour: MicrosField | None = None
    committed_1y_per_hour: MicrosField | None = None

    @model_validator(mode="after")
    def _verified_needs_specs(self) -> Self:
        if self.verified and None in (self.vcpus, self.memory_gib, self.local_nvme_gb):
            raise ValueError("verified entries need vcpus, memory_gib and local_nvme_gb")
        return self


class BlockStorage(_Sourced):
    kind: NonEmptyStr
    per_gb_month: MicrosField


class DataTransfer(_Sourced):
    ingress_per_gb: MicrosField
    egress_internet_per_gb: MicrosField


class Region(StrictModel):
    instances: Annotated[dict[str, InstanceType], Field(min_length=1)]
    storage: BlockStorage | None = None
    data_transfer: DataTransfer | None = None


class PriceBook(StrictModel):
    last_checked: dt.date
    clouds: dict[Cloud, dict[str, Region]]

    def region(self, cloud: Cloud, region: str) -> Region:
        try:
            return self.clouds[cloud][region]
        except KeyError:
            raise KeyError(f"no prices for {cloud}/{region}") from None

    def instance(self, cloud: Cloud, region: str, instance_type: str) -> InstanceType:
        try:
            return self.region(cloud, region).instances[instance_type]
        except KeyError:
            raise KeyError(f"no prices for {cloud}/{region}/{instance_type}") from None

    def instance_price(
        self,
        cloud: Cloud,
        region: str,
        instance_type: str,
        market: Market = Market.ON_DEMAND,
        *,
        allow_unverified: bool = False,
    ) -> Quote:
        """Hourly price of one instance. Spot without a recorded price falls back to
        on-demand and says so; a missing committed price is an error."""
        it = self.instance(cloud, region, instance_type)
        if not it.verified and not allow_unverified:
            raise UnverifiedPriceError(f"{cloud}/{region}/{instance_type}: {it.note}")
        if market is Market.SPOT:
            if it.spot_per_hour is None:
                return Quote(it.on_demand_per_hour, Market.ON_DEMAND, spot_fallback=True)
            return Quote(it.spot_per_hour, Market.SPOT, spot_fallback=False)
        if market is Market.COMMITTED_1Y:
            if it.committed_1y_per_hour is None:
                raise KeyError(f"no committed_1y price for {cloud}/{region}/{instance_type}")
            return Quote(it.committed_1y_per_hour, Market.COMMITTED_1Y, spot_fallback=False)
        return Quote(it.on_demand_per_hour, Market.ON_DEMAND, spot_fallback=False)

    def replica_hourly_cost(
        self,
        cloud: Cloud,
        region: str,
        instance_type: str,
        market: Market = Market.ON_DEMAND,
        storage_gb: int = 0,
        *,
        allow_unverified: bool = False,
    ) -> Quote:
        """Hourly cost of one single-node replica: the instance plus `storage_gb` of
        block storage amortised per hour. Rounded once, at the end."""
        if storage_gb < 0:
            raise ValueError("storage_gb must be >= 0")
        quote = self.instance_price(
            cloud, region, instance_type, market, allow_unverified=allow_unverified
        )
        if storage_gb == 0:
            return quote
        storage = self.region(cloud, region).storage
        if storage is None:
            raise KeyError(f"no storage price for {cloud}/{region}")
        if not storage.verified and not allow_unverified:
            raise UnverifiedPriceError(f"{cloud}/{region} storage: {storage.note}")
        storage_per_hour = Fraction(storage.per_gb_month * storage_gb, HOURS_PER_MONTH)
        return quote._replace(per_hour=round_half_up(quote.per_hour + storage_per_hour))


def load_prices(path: Path | str = DEFAULT_PRICES_YAML) -> PriceBook:
    return PriceBook.model_validate(read_yaml(Path(path)))


class CompetitorEntry(StrictModel):
    model_id: NonEmptyStr  # our registry id
    provider_model: NonEmptyStr | None = None  # provider's own name, when the page gives one
    input_per_mtok: MicrosField
    output_per_mtok: MicrosField
    quantization: Quantization | None = None  # only when the provider discloses it
    availability: Literal["listed", "unverified"] = "listed"
    source: HttpUrl
    notes: NonEmptyStr | None = None


class Provider(StrictModel):
    name: NonEmptyStr
    pricing_url: HttpUrl
    aggregator: bool = False  # resells other providers' endpoints
    last_checked: dt.date
    entries: list[CompetitorEntry]


class NotOffered(StrictModel):
    name: NonEmptyStr
    reason: NonEmptyStr
    url: HttpUrl | None = None
    last_checked: dt.date


class Competitors(StrictModel):
    last_checked: dt.date
    providers: list[Provider]
    not_offered: list[NotOffered] = Field(default_factory=list)

    @model_validator(mode="after")
    def _unique_names(self) -> Self:
        names = [p.name for p in self.providers] + [n.name for n in self.not_offered]
        dupes = sorted({n for n in names if names.count(n) > 1})
        if dupes:
            raise ValueError(f"provider listed more than once: {dupes}")
        return self

    def entries_for(self, model_id: str) -> list[tuple[Provider, CompetitorEntry]]:
        return [(p, e) for p in self.providers for e in p.entries if e.model_id == model_id]


def load_competitors(path: Path | str = DEFAULT_COMPETITORS_YAML) -> Competitors:
    return Competitors.model_validate(read_yaml(Path(path)))
