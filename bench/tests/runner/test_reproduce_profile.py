"""`bench reproduce` compares a workload's profile with the one a run recorded; a change to
the dataset's licence note alone does not change a request, so it is not a difference."""

from types import SimpleNamespace
from typing import cast

from loom_bench.provenance import DatasetInfo, Provenance, WorkloadInfo, config_hash
from loom_bench.runner import same_profile
from loom_bench.workloads.profiles import load_profile

OLD_NOTE = "HF card: apache-2.0. Verify before publishing."


def _recorded(profile, **dataset) -> Provenance:
    """The two sections of a run's provenance the comparison reads."""
    stub = SimpleNamespace(
        workload=WorkloadInfo(name="chat-sharegpt", profile_hash=config_hash(profile)),
        dataset=DatasetInfo(**dataset),
    )
    return cast(Provenance, stub)


def test_a_licence_only_change_is_the_same_workload():
    current = load_profile("chat-sharegpt")
    assert current.dataset is not None
    assert "Verify before publishing" not in current.dataset.license
    old = current.model_copy(
        update={"dataset": current.dataset.model_copy(update={"license": OLD_NOTE})}
    )
    prov = _recorded(old, **{**current.dataset.model_dump(), "license": OLD_NOTE})
    assert config_hash(current) != prov.workload.profile_hash
    assert same_profile(current, prov)


def test_any_other_change_is_still_a_difference():
    current = load_profile("chat-sharegpt")
    assert current.dataset is not None
    old = current.model_copy(
        update={
            "seed": current.seed + 1,
            "dataset": current.dataset.model_copy(update={"license": OLD_NOTE}),
        }
    )
    prov = _recorded(old, **{**current.dataset.model_dump(), "license": OLD_NOTE})
    assert not same_profile(current, prov)
    assert same_profile(current, _recorded(current))
