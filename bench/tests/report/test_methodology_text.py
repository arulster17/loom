"""Methodology text published from provenance: dataset licence notes, synthetic runs with
no dataset, and the storage each cloud's replica pays for."""

from loom_bench.cost import CostAllocation
from loom_bench.report.analyze import analyze_runs, default_price_resolver
from loom_bench.report.methodology import DatasetRef, methodology, public_dataset, storage_name
from loom_bench.workloads.profiles import load_profile

from .factories import SLO, make_runs

SHAREGPT = load_profile("chat-sharegpt").dataset


def test_a_recorded_licence_note_is_published_as_the_profile_states_it_now():
    assert SHAREGPT is not None
    recorded = DatasetRef(
        name=SHAREGPT.name,
        source=SHAREGPT.source,
        revision=SHAREGPT.revision,
        license=SHAREGPT.license + " Verify before publishing.",
        license_url=SHAREGPT.license_url,
    )
    shown = public_dataset(recorded)
    assert shown.license == SHAREGPT.license
    assert "Verify before publishing" not in shown.text
    # another revision of the dataset is not the profile's: shown as recorded
    other = recorded.model_copy(update={"revision": "0" * 40})
    assert public_dataset(other) == other


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
