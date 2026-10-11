"""Every committed workload profile hashes to a known value.

A run records `config_hash(profile)` as `workload.profile_hash`, and stored runs,
`bench compare` and `bench reproduce` key on it. Any edit to a profile file, even a
comment-like change inside a hashed field such as a dataset licence note, moves the hash
and breaks the pairing with every stored run. Change a value here only together with a
deliberate, documented change of workload.

The first five values are the hashes recorded by stored runs (results store, 2026-10-11:
7a8237d0, 95cde129, 565b8d3f, 9f0853d7 and others); the rest are the profiles as
committed, not run on a GPU yet.
"""

import json
from pathlib import Path

import pytest

from loom_bench.provenance import config_hash
from loom_bench.workloads.profiles import list_profiles, load_profile

KNOWN = {
    # recorded by stored runs
    "chat-sharegpt": "334a484887e52c9dfe1c416f715a192d7b186c8eec10550b867f82fe4fb457b4",
    "code-completion": "6ebf848c79eccea09d317294ef0051599b65f456b858e3eeeb0a99a1acf2ab19",
    "fixed-128-128": "3943749cfa9d9b94d20214fa4e290eb0b65adb3d62218170517149198b8eed22",
    "fixed-1k-1k": "75c5407fe717f2ffcdc83872292afe7dc5bacb51b585f4ea0948f62c8a20500f",
    "shared-prefix": "270207787c4cea50a0ed6d798292773c497099e1402095b20d177748144878d0",
    # as committed
    "fixed-32k-1k": "a7b5d16713b06e3a789a9f7c091e660fdaa859069da36a776440be990b68ddf5",
    "fixed-8k-1k": "632edabb57f66b91c0c76a01a0adafd1b579e511f0724a0a7d749aa779e986d6",
    "long-context-needle": "ac455171c8511f6a7c523c233e4e1c034f0e00d68fc31073676601a093d11ad2",
    "long-generation": "f26ab19b2febd5efd2a0fe29002ba1702d248bb9cfc079fa1a6703076df4c70d",
    "trace-azure-code": "7feca782b47aa13bd634bf510d73de3b1162c77d3a03af0a7e2b16670250dfd6",
}

STORED_RUNS = (
    Path(__file__).resolve().parents[1] / "runpod" / "fixtures" / "reproduce_7a8237d0" / "runs.json"
)


def test_every_committed_profile_is_pinned():
    assert sorted(list_profiles()) == sorted(KNOWN)


@pytest.mark.parametrize("name", sorted(KNOWN))
def test_profile_hash_is_unchanged(name):
    assert config_hash(load_profile(name)) == KNOWN[name], (
        f"{name}'s profile hash moved: stored runs, compare and reproduce key on it"
    )


def test_hash_matches_the_stored_run_fixture():
    runs = json.loads(STORED_RUNS.read_text())
    recorded = {
        (r["provenance"]["workload"]["name"], r["provenance"]["workload"]["profile_hash"])
        for r in runs
    }
    assert recorded == {("chat-sharegpt", KNOWN["chat-sharegpt"])}
