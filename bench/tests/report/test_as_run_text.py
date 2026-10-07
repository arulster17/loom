"""The methodology's "As run" text for each price basis a provider records.

RunPod records `observed_api` (the pod's API `costPerHr` at launch); the text used to fall
through to the spot-only "no spot price was observed" sentence and read "As run: unknown".
"""

from datetime import UTC, datetime
from types import SimpleNamespace

from loom_bench.provenance import Market, PriceBasis
from loom_bench.report.methodology import as_run_text

AT = datetime(2026, 10, 7, 5, 40, tzinfo=UTC)


def _result(basis: PriceBasis | None, as_run: int | None, cloud: str = "runpod"):
    prov = {"cloud": cloud, "price_basis": basis.model_dump(mode="json") if basis else None}
    return SimpleNamespace(provenance=prov, prices=SimpleNamespace(as_run=as_run))


def _observed(zone: str | None) -> PriceBasis:
    return PriceBasis(
        market=Market.ON_DEMAND,
        source="observed_api",
        observed_at=AT,
        availability_zone=zone,
        storage_gb=80,
    )


def test_an_api_observed_price_is_shown_with_its_datacenter_and_time():
    text = as_run_text(_result(_observed("US-MO-1"), 1_100_959))
    assert text == (
        "$1.1010/h, on_demand host: runpod API price observed at launch in US-MO-1 "
        "(2026-10-07T05:40:00+00:00) + 80 GB block storage"
    )


def test_a_location_is_not_presented_as_a_datacenter():
    text = as_run_text(_result(_observed("location:SE"), 1_100_959))
    assert "in location SE (datacenter not reported)" in text
    assert "unknown" not in text


def test_an_unrecorded_datacenter_is_said_so():
    assert "in an unrecorded datacenter" in as_run_text(_result(_observed(None), 1_100_959))


def test_an_unobserved_spot_price_is_still_unknown():
    basis = PriceBasis(market=Market.SPOT, source="unobserved", storage_gb=200)
    text = as_run_text(_result(basis, None, cloud="aws"))
    assert text.startswith("unknown, spot host: no spot price was observed")


def test_no_basis_is_not_recorded():
    assert as_run_text(_result(None, None)) == "not recorded"
