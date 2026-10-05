import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta, timezone
from enum import Enum, IntEnum
from pathlib import Path

import pytest
from pydantic import BaseModel, ValidationError

from loom_bench import __version__
from loom_bench.provenance import (
    EngineInfo,
    GitInfo,
    Market,
    Provenance,
    build_provenance,
    canonical_json,
    config_hash,
    git_info,
    provenance_hash,
)
from loom_bench.records import LoadMode


class Color(Enum):
    RED = "red"


class Level(IntEnum):
    HIGH = 3


@dataclass
class Knobs:
    tp: int
    prefix_caching: bool


class KnobsModel(BaseModel):
    tp: int
    prefix_caching: bool


def test_canonical_json_is_sorted_and_compact():
    assert canonical_json({"b": 1, "a": {"d": [1, 2], "c": None}}) == (
        '{"a":{"c":null,"d":[1,2]},"b":1}'
    )
    assert config_hash({"b": 1, "a": 2}) == config_hash({"a": 2, "b": 1})


def test_canonical_json_pinned_encoding():
    obj = {
        "mode": LoadMode.OPEN_LOOP,
        "color": Color.RED,
        "level": Level.HIGH,
        Color.RED: 1,
        2: "int key",
        "path": Path("/data") / "share.jsonl",
        "at": datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone(timedelta(hours=2))),
        "tuple": (1, 2),
        "set": {"b", "a"},
        "unicode": "héllo",
    }
    assert canonical_json(obj) == (
        '{"2":"int key","at":"2026-01-02T01:04:05+00:00","color":"red","level":3,'
        '"mode":"open_loop","path":"/data/share.jsonl","red":1,"set":["a","b"],'
        '"tuple":[1,2],"unicode":"héllo"}'
    )


def test_floats_encode_shortest_round_trip():
    assert canonical_json({"x": 0.1 + 0.2, "y": 1e-7, "z": 2.0}) == (
        '{"x":0.30000000000000004,"y":1e-07,"z":2.0}'
    )
    assert config_hash({"x": 0.1}) == config_hash({"x": float("0.1")})
    assert config_hash({"x": 1}) != config_hash({"x": 1.0})
    with pytest.raises(ValueError):
        canonical_json({"x": float("nan")})


def test_models_dataclasses_and_dicts_hash_alike():
    plain = {"tp": 2, "prefix_caching": True}
    assert config_hash(Knobs(tp=2, prefix_caching=True)) == config_hash(plain)
    assert config_hash(KnobsModel(tp=2, prefix_caching=True)) == config_hash(plain)


def test_canonical_json_rejects_ambiguous_values():
    with pytest.raises(ValueError):
        canonical_json({"at": datetime(2026, 1, 1)})
    with pytest.raises(ValueError):
        canonical_json({1: "a", "1": "b"})
    with pytest.raises(TypeError):
        canonical_json({"x": object()})


def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=t",
            "-c",
            "user.email=t@example.com",
            "-c",
            "commit.gpgsign=false",
            *args,
        ],
        cwd=repo,
        check=True,
        capture_output=True,
    )


def test_git_info_in_temp_repo(tmp_path):
    _git(tmp_path, "init", "-b", "main")
    assert git_info(tmp_path) == GitInfo(sha=None, dirty=None, branch="main")

    (tmp_path / "a.txt").write_text("one")
    _git(tmp_path, "add", "a.txt")
    _git(tmp_path, "commit", "-m", "first")
    info = git_info(tmp_path)
    assert info.sha is not None and len(info.sha) == 40
    assert info.dirty is False and info.branch == "main"

    (tmp_path / "untracked.txt").write_text("x")
    assert git_info(tmp_path).dirty is False
    (tmp_path / "a.txt").write_text("two")
    assert git_info(tmp_path).dirty is True

    _git(tmp_path, "checkout", "--detach")
    assert git_info(tmp_path).branch is None


def test_git_info_outside_a_repo(tmp_path):
    assert git_info(tmp_path / "missing") == GitInfo()
    assert git_info(tmp_path) == GitInfo()


def _prov(**kw):
    return build_provenance(
        {"engine": "vllm", "args": {"max_num_seqs": 256}},
        created_at=datetime(2026, 10, 4, 12, 0, tzinfo=UTC),
        **kw,
    )


def test_build_provenance_leaves_unknowns_none():
    p = _prov(engine=EngineInfo(name="vllm", version="0.11.0"), market=Market.SPOT)
    dumped = p.model_dump(mode="json")
    assert dumped["config_hash"] == config_hash({"engine": "vllm", "args": {"max_num_seqs": 256}})
    assert dumped["engine"]["image_digest"] is None
    assert dumped["cuda_version"] is None and dumped["driver_version"] is None
    assert dumped["dataset"] == {
        "name": None,
        "source": None,
        "revision": None,
        "license": None,
        "license_url": None,
    }
    assert dumped["git"] == {"sha": None, "dirty": None, "branch": None}
    assert dumped["bench_version"] == __version__
    assert dumped["host"]["python"] is not None
    assert dumped["market"] == "spot"


def test_provenance_round_trip_and_hash_stability():
    p = _prov(repetition=2, load={"mode": "open_loop", "value": 4.5, "seed": 7})
    restored = Provenance.model_validate_json(p.model_dump_json())
    assert restored == p
    assert provenance_hash(restored) == provenance_hash(p)
    assert provenance_hash(_prov(repetition=3)) != provenance_hash(p)


def test_provenance_created_at_normalized_to_utc():
    local = datetime(2026, 10, 4, 14, 0, tzinfo=timezone(timedelta(hours=2)))
    p = build_provenance({}, created_at=local)
    assert p.created_at.utcoffset() == timedelta(0)
    assert p.created_at == local


def test_provenance_rejects_derived_and_unknown_fields():
    with pytest.raises(TypeError):
        _prov(config_hash="abc")
    with pytest.raises(ValidationError):
        _prov(gpu="L40S")
    with pytest.raises(ValidationError):
        build_provenance({}, created_at=datetime(2026, 1, 1))
