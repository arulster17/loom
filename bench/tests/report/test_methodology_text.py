"""Methodology text published from provenance: dataset licence notes, synthetic runs with
no dataset, and the storage each cloud's replica pays for."""

from loom_bench.cost import CostAllocation
from loom_bench.report.analyze import analyze_runs, default_price_resolver
from loom_bench.report.methodology import (
    DatasetRef,
    methodology,
    public_dataset,
    public_license,
    storage_name,
)
from loom_bench.workloads.profiles import load_profile

from .factories import SLO, make_runs

SHAREGPT = load_profile("chat-sharegpt").dataset


def test_internal_reminders_are_not_published_but_stay_in_the_profile():
    """The profile keeps its note (its hash is pinned); published text drops the reminder."""
    assert SHAREGPT is not None
    assert SHAREGPT.license.endswith("Verify before publishing.")
    recorded = DatasetRef.model_validate(SHAREGPT.model_dump())
    shown = public_dataset(recorded)
    assert "Verify before publishing" not in shown.text
    assert shown.license is not None
    assert shown.license.endswith("publishes only latency and cost.")
    assert "OpenAI's terms of use" in shown.license
    assert public_license("CC-BY-4.0") == "CC-BY-4.0"
    assert public_license(None) is None


def test_synthetic_runs_name_no_dataset(price_book):
    runs = make_runs("vllm-bf16")
    for run in runs:  # a synthetic workload records an empty dataset section
        run.provenance = {**run.provenance, "dataset": {"name": None, "source": None}}
    results = analyze_runs(
        runs,
        slo=SLO,
        allocation=CostAllocation.all_output(),
        price_resolver=default_price_resolver(price_book),
    )
    m = methodology(results, price_book)
    assert m.datasets == []


def test_each_clouds_storage_is_named():
    assert storage_name("aws") == "EBS gp3 volume"
    assert storage_name("runpod") == "container disk"
    assert storage_name(None) == "storage volume"
