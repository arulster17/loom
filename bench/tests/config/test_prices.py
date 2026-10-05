import copy
from fractions import Fraction
from typing import Any

import pytest
from pydantic import ValidationError

from loom_bench.prices import (
    Competitors,
    Market,
    PriceBook,
    Quote,
    UnverifiedPriceError,
    load_competitors,
    load_prices,
)
from loom_bench.registry import load_registry

URL = "https://example.com/prices"


def instance(**overrides: Any) -> dict[str, Any]:
    return {
        "gpu": "L40S",
        "gpu_count": 1,
        "gpu_memory_gb": 48,
        "vcpus": 4,
        "memory_gib": 32,
        "local_nvme_gb": 250,
        "on_demand_per_hour": 1_000_000,
        "sources": [URL],
        "last_checked": "2026-10-04",
        **overrides,
    }


def book(
    storage_per_gb_month: int | None = None,
    storage_verified: bool = True,
    **instance_overrides: Any,
) -> PriceBook:
    region: dict[str, Any] = {"instances": {"x.large": instance(**instance_overrides)}}
    if storage_per_gb_month is not None:
        region["storage"] = {
            "kind": "ebs-gp3",
            "per_gb_month": storage_per_gb_month,
            "sources": [URL],
            "last_checked": "2026-10-04",
            **({} if storage_verified else {"verified": False, "note": "check it"}),
        }
    return PriceBook.model_validate(
        {"last_checked": "2026-10-04", "clouds": {"aws": {"r-1": region}}}
    )


def test_every_shipped_price_loads_and_is_consistent():
    """Properties of whatever bench/prices.yaml ships, so adding a price is config only."""
    prices = load_prices()
    entries = [
        (cloud, region, name, it)
        for cloud, regions in prices.clouds.items()
        for region, r in regions.items()
        for name, it in r.instances.items()
    ]
    assert entries
    for cloud, region, name, it in entries:
        where = f"{cloud}/{region}/{name}"
        assert it.last_checked <= prices.last_checked, where
        for discounted in (it.spot_per_hour, it.committed_1y_per_hour):
            assert discounted is None or discounted < it.on_demand_per_hour, where
        quote = prices.instance_price(cloud, region, name, allow_unverified=True)
        assert quote.per_hour == it.on_demand_per_hour
    # A cloud with a provider is priced with storage: every one of its regions needs it.
    for region in prices.clouds["aws"].values():
        assert region.storage is not None and region.storage.verified


def test_every_registry_instance_type_has_a_price():
    prices = load_prices()
    for model in load_registry().models:
        for cloud in model.clouds:
            instance_type = getattr(model.hardware.instance_types, cloud)
            regions = prices.clouds[cloud].values()
            matches = [r.instances[instance_type] for r in regions if instance_type in r.instances]
            assert matches, f"{model.id}: no price for {cloud}/{instance_type}"
            if cloud == "aws":  # reports refuse unverified prices
                assert all(m.verified for m in matches), f"{model.id}: {instance_type}"
            assert all(m.gpu == model.hardware.gpu for m in matches)
            assert all(m.gpu_count == model.hardware.gpus_per_replica for m in matches)


def test_instance_price_markets():
    prices = load_prices()
    od = prices.instance_price("aws", "us-east-1", "g6e.xlarge")
    assert od == Quote(1_861_000, Market.ON_DEMAND, spot_fallback=False)
    spot = prices.instance_price("aws", "us-east-1", "g6e.xlarge", Market.SPOT)
    assert spot == Quote(1_838_600, Market.SPOT, spot_fallback=False)


def test_spot_falls_back_to_on_demand_with_flag():
    quote = book().instance_price("aws", "r-1", "x.large", Market.SPOT)
    assert quote == Quote(1_000_000, Market.ON_DEMAND, spot_fallback=True)


def test_committed_without_price_is_an_error():
    with pytest.raises(KeyError, match="committed_1y"):
        book().instance_price("aws", "r-1", "x.large", Market.COMMITTED_1Y)
    quote = book(committed_1y_per_hour=600_000).instance_price(
        "aws", "r-1", "x.large", Market.COMMITTED_1Y
    )
    assert quote == Quote(600_000, Market.COMMITTED_1Y, spot_fallback=False)


def test_unknown_lookups_raise():
    prices = load_prices()
    with pytest.raises(KeyError, match="aws/eu-west-1"):
        prices.instance_price("aws", "eu-west-1", "g6e.xlarge")
    with pytest.raises(KeyError, match=r"g7\.xlarge"):
        prices.instance_price("aws", "us-east-1", "g7.xlarge")


def test_unverified_price_refused_unless_allowed():
    prices = load_prices()
    with pytest.raises(UnverifiedPriceError, match="g2-standard-8"):
        prices.instance_price("gcp", "us-central1", "g2-standard-8")
    quote = prices.instance_price("gcp", "us-central1", "g2-standard-8", allow_unverified=True)
    assert quote.per_hour == 853_600


def test_storage_amortised_per_hour_exact_to_the_micro():
    prices = load_prices()
    # $1.861/h + 100 GB x $0.08/GB-month / 730 h = 1_861_000 + 10_958.904... micros
    quote = prices.replica_hourly_cost(
        "aws", "us-east-1", "g6e.xlarge", Market.ON_DEMAND, 100, allow_unverified=True
    )
    assert quote == Quote(1_871_959, Market.ON_DEMAND, spot_fallback=False)
    # 730 micros/GB-month is exactly 1 micro/GB-hour.
    assert book(730).replica_hourly_cost("aws", "r-1", "x.large", storage_gb=7).per_hour == (
        1_000_007
    )


def test_storage_rounds_once_at_the_end():
    # Each GB costs half a micro per hour: 3 GB = 1.5 -> 2, not 3 x round(0.5).
    quote = book(365).replica_hourly_cost("aws", "r-1", "x.large", storage_gb=3)
    assert quote.per_hour == 1_000_002
    quote = book(365).replica_hourly_cost("aws", "r-1", "x.large", storage_gb=2)
    assert quote.per_hour == 1_000_001


def test_replica_cost_keeps_spot_fallback_flag():
    quote = book(730).replica_hourly_cost("aws", "r-1", "x.large", Market.SPOT, 1)
    assert quote == Quote(1_000_001, Market.ON_DEMAND, spot_fallback=True)


def test_storage_price_checks():
    prices = load_prices()
    assert prices.replica_hourly_cost("aws", "us-east-1", "g6e.xlarge").per_hour == 1_861_000
    # $0.08/GB-month x 200 GB / 730 h = 21,917.8 micros/h, added before the one rounding
    with_volume = prices.replica_hourly_cost("aws", "us-east-1", "g6e.xlarge", storage_gb=200)
    assert with_volume.per_hour == 1_882_918
    assert prices.with_storage("aws", "us-east-1", Fraction(1, 2), 200) == 21_918
    unverified = book(730, storage_verified=False)
    with pytest.raises(UnverifiedPriceError, match="storage"):
        unverified.replica_hourly_cost("aws", "r-1", "x.large", storage_gb=100)
    assert unverified.with_storage("aws", "r-1", 0, 730, allow_unverified=True) == 730
    with pytest.raises(KeyError, match="no storage price"):
        book().replica_hourly_cost("aws", "r-1", "x.large", storage_gb=1)
    with pytest.raises(ValueError, match="storage_gb"):
        book(730).replica_hourly_cost("aws", "r-1", "x.large", storage_gb=-1)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"on_demand_per_hour": 1.861}, "on_demand_per_hour"),
        ({"on_demand_per_hour": "1.861"}, "on_demand_per_hour"),
        ({"spot_per_hour": -1}, "spot_per_hour"),
        ({"vcpus": None}, "vcpus"),
        ({"verified": False}, "requires a note"),
        ({"sources": []}, "sources"),
        ({"sources": ["not a url"]}, "sources"),
        ({"price": 1}, "price"),
    ],
)
def test_rejects_bad_instance(overrides: dict[str, Any], message: str):
    with pytest.raises(ValidationError, match=message):
        book(**overrides)


def test_unverified_entry_may_omit_specs():
    b = book(vcpus=None, memory_gib=None, local_nvme_gb=None, verified=False, note="check it")
    assert b.instance("aws", "r-1", "x.large").vcpus is None


def test_real_competitors_load():
    comp = load_competitors()
    ids = {m.id for m in load_registry().models}
    assert {e.model_id for p in comp.providers for e in p.entries} <= ids

    by_name = {p.name: p for p in comp.providers}
    assert by_name["OpenRouter"].aggregator
    assert not any(p.aggregator for n, p in by_name.items() if n != "OpenRouter")
    assert all(e.availability == "unverified" for e in by_name["Fireworks AI"].entries)
    together = by_name["Together AI"].entries[0]
    assert (together.input_per_mtok, together.output_per_mtok) == (1_040_000, 1_040_000)
    assert by_name["DeepInfra"].entries[0].quantization == "fp8"
    assert {n.name for n in comp.not_offered} == {
        "Groq",
        "Cerebras",
        "Lambda",
        "Hyperbolic",
        "Nebius",
    }
    assert [p.name for p, _ in comp.entries_for("qwen3-8b")] == ["Fireworks AI", "OpenRouter"]


def test_competitors_reject_duplicates_and_floats():
    raw = load_competitors().model_dump(mode="json")
    dup = copy.deepcopy(raw)
    dup["not_offered"].append({**dup["not_offered"][0]})
    with pytest.raises(ValidationError, match="more than once"):
        Competitors.model_validate(dup)
    bad = copy.deepcopy(raw)
    bad["providers"][0]["entries"][0]["input_per_mtok"] = 1.04
    with pytest.raises(ValidationError, match="input_per_mtok"):
        Competitors.model_validate(bad)
