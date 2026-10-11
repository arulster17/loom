import pytest
from pydantic import ValidationError

from loom_bench.workloads import list_profiles, load_profile, parse_profile
from loom_bench.workloads.profiles import (
    ChatDatasetProfile,
    SharedPrefixProfile,
    SyntheticProfile,
    TraceProfile,
)


def test_every_shipped_profile_loads_and_validates():
    """Properties of whatever bench/workloads/ ships, so adding a profile is config only."""
    names = list_profiles()
    assert names
    for name in names:
        p = load_profile(name)
        assert p.name == name  # the file name is the profile name experiments use
        assert p.description and p.content in ("synthetic", "realistic")
        if p.content == "realistic" or hasattr(p, "path"):  # reads a dataset or trace file
            assert p.dataset is not None and p.dataset.source and p.dataset.license, name


def test_dataset_backed_profiles_carry_provenance_and_license():
    chat = load_profile("chat-sharegpt")
    assert isinstance(chat, ChatDatasetProfile) and chat.content == "realistic"
    assert "verify before publishing" in chat.dataset.license.lower()
    trace = load_profile("trace-azure-code")
    assert isinstance(trace, TraceProfile) and trace.dataset.license == "CC-BY-4.0"
    assert load_profile("fixed-1k-1k").dataset is None


def test_fixed_shapes():
    shapes = {
        "fixed-128-128": (128, 128),
        "fixed-1k-1k": (1024, 1024),
        "fixed-8k-1k": (8192, 1024),
    }
    for name, (i, o) in shapes.items():
        p = load_profile(name)
        assert isinstance(p, SyntheticProfile) and (p.input_len, p.output_len) == (i, o)
        assert p.ignore_eos
    long = load_profile("fixed-32k-1k")
    assert long.input_len + long.output_len <= 32768


def test_overrides_are_validated_and_deep_merged():
    p = load_profile("shared-prefix", {"prefix_share": 0.9})
    assert isinstance(p, SharedPrefixProfile) and p.prefix_share == 0.9
    assert load_profile("shared-prefix").prefix_share == 0.5
    with pytest.raises(ValidationError):
        load_profile("shared-prefix", {"prefix_share": 0.99})
    with pytest.raises(ValidationError):
        load_profile("fixed-1k-1k", {"no_such_field": 1})
    t = load_profile("trace-azure-code", {"path": "/x.csv", "dataset": {"revision": "abc123"}})
    assert t.path == "/x.csv"
    assert t.dataset.revision == "abc123" and t.dataset.license == "CC-BY-4.0"


def test_load_by_path_and_unknown_name(tmp_path):
    f = tmp_path / "mine.yaml"
    f.write_text(
        "name: mine\ndescription: d\nkind: synthetic\ncontent: synthetic\n"
        "input_len: 10\noutput_len: 5\n"
    )
    assert load_profile(f).input_len == 10
    assert load_profile(str(f), {"input_len": 20}).input_len == 20
    with pytest.raises(FileNotFoundError, match="fixed-1k-1k"):
        load_profile("does-not-exist")


def test_endpoint_constraints():
    base = {"name": "x", "description": "d", "content": "synthetic"}
    with pytest.raises(ValidationError):
        parse_profile({**base, "kind": "code_completion", "endpoint": "chat", "input_len": 8,
                       "output_len": 8})  # fmt: skip
    with pytest.raises(ValidationError):
        parse_profile({**base, "kind": "shared_prefix", "endpoint": "completions",
                       "input_len": 8, "output_len": 8, "prefix_share": 0.5})  # fmt: skip
    with pytest.raises(ValidationError, match="no room"):
        parse_profile({**base, "kind": "shared_prefix", "input_len": 10, "output_len": 8,
                       "prefix_share": 0.95, "range_ratio": 0.5})  # fmt: skip
