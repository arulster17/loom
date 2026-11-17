"""`config/serving.yaml` and its loader `loom_platform.serving.load_serving`
(docs/phase1-m2-spec.md §4, §13 item 1).

Every rejection test starts from a file that loads (`test_base_file_loads`) and changes
one thing, so a failure points at that one rule.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest
import yaml

from loom_bench.engines import render_launch
from loom_bench.experiment import expand, load_experiment
from loom_bench.registry import REPO_ROOT, load_registry

SERVING_ENV = "LOOM_SERVING_YAML"
MODELS_ENV = "LOOM_MODELS_YAML"
SMALL = "qwen3-8b"
BIG = "llama-3.3-70b-instruct-fp8"

# The §4 example, with provider `fake` (an empty egress allowlist is allowed there) and
# the 70B's engine env as benchmarked.
BASE: dict[str, Any] = {
    "version": 1,
    "provider": "fake",
    "spend": {"monthly_cap_usd": 100, "alert_fractions": [0.5, 0.8, 1.0], "max_price_ratio": 1.25},
    "placement": {
        "data_center_prefixes": ["US-"],
        "allowed_cuda_versions": ["13.0"],
        "container_disk_gb": {SMALL: 80, BIG: 120},
    },
    "network": {"gateway_egress_cidrs": [], "proxy_port": 8443},
    "lifecycle": {
        "reconcile_every_s": 15,
        "lease_ttl_s": 3600,
        "lease_renew_every_s": 300,
        "drain_max_s": 600,
        "server_cert_days": 7,
        "server_cert_renew_before_s": 172800,
        "max_replacements_per_hour": 3,
    },
    "health": {
        "pod_state_every_s": 30,
        "proxy_health_every_s": 10,
        "proxy_health_failures": 3,
        "unhealthy_grace_s": 120,
        "progress_stall_s": 60,
        "bootstrap_stall_s": 600,
        "bootstrap_timeout_s": 1800,
    },
    "capacity": {"backoff_s": [30, 60, 120, 240], "unavailable_retry_after_s": 300},
    "models": {
        SMALL: {
            "min_replicas": 0,
            "max_replicas": 1,
            "idle_stop_s": 1800,
            "cold_start_estimate_s": 240,
            "engine_env": {},
        },
        BIG: {
            "min_replicas": 0,
            "max_replicas": 1,
            "idle_stop_s": 1800,
            "cold_start_estimate_s": 450,
            "engine_env": {"NCCL_P2P_LEVEL": "NVL"},
        },
    },
}


@pytest.fixture(autouse=True)
def _default_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every test starts from the repo's own files unless it sets the variables."""
    monkeypatch.delenv(SERVING_ENV, raising=False)
    monkeypatch.delenv(MODELS_ENV, raising=False)


def write(tmp_path: Path, doc: dict[str, Any], name: str = "serving.yaml") -> Path:
    path = tmp_path / name
    path.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
    return path


def base() -> dict[str, Any]:
    return copy.deepcopy(BASE)


def load(path: Path | str | None = None) -> Any:
    from loom_platform.serving import load_serving

    return load_serving(path) if path is not None else load_serving()


def rejected(path: Path) -> str:
    """The error message of loading a file that must not load."""
    with pytest.raises(ValueError) as e:
        load(path)
    return str(e.value)


# ------------------------------------------------------------------ loads


def test_base_file_loads(tmp_path: Path) -> None:
    cfg = load(write(tmp_path, base()))
    assert cfg.version == 1
    assert cfg.provider == "fake"
    # Compared as numbers: float or Decimal both pass.
    assert float(cfg.spend.monthly_cap_usd) == 100
    assert [float(x) for x in cfg.spend.alert_fractions] == [0.5, 0.8, 1.0]
    assert float(cfg.spend.max_price_ratio) == 1.25
    assert list(cfg.placement.data_center_prefixes) == ["US-"]
    assert cfg.placement.container_disk_gb[BIG] == 120
    assert cfg.network.proxy_port == 8443
    assert cfg.lifecycle.lease_ttl_s == 3600
    assert cfg.lifecycle.max_replacements_per_hour == 3
    assert cfg.health.proxy_health_failures == 3
    assert cfg.health.bootstrap_timeout_s == 1800
    assert list(cfg.capacity.backoff_s) == [30, 60, 120, 240]
    assert cfg.capacity.unavailable_retry_after_s == 300
    assert set(cfg.models) == {SMALL, BIG}
    small = cfg.models[SMALL]
    assert (small.min_replicas, small.max_replicas) == (0, 1)
    assert (small.idle_stop_s, small.cold_start_estimate_s) == (1800, 240)
    assert dict(small.engine_env) == {}
    assert dict(cfg.models[BIG].engine_env) == {"NCCL_P2P_LEVEL": "NVL"}


def test_repo_serving_yaml_loads_with_the_decided_values() -> None:
    """The committed config/serving.yaml (the default path) is the compose default
    (§4): `provider: local` with an empty egress allowlist. It carries the decisions §4
    and §15 name: on demand, US only, the $100 placeholder cap, and the measured cold
    starts (contracts §10: 8B 240 s, 70B 450 s)."""
    cfg = load()
    assert cfg.version == 1
    assert cfg.provider == "local"
    assert list(cfg.network.gateway_egress_cidrs) == []
    assert set(cfg.models) == {SMALL, BIG}
    assert cfg.models[SMALL].cold_start_estimate_s == 240
    assert cfg.models[BIG].cold_start_estimate_s == 450
    for model in cfg.models.values():
        assert model.min_replicas == 0  # on demand (decision 2026-10-31)
        assert model.max_replicas >= 1
    assert list(cfg.placement.data_center_prefixes) == ["US-"]
    assert float(cfg.spend.monthly_cap_usd) == 100


def test_env_var_overrides_the_default_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    doc = base()
    doc["models"][SMALL]["cold_start_estimate_s"] = 123
    monkeypatch.setenv(SERVING_ENV, str(write(tmp_path, doc)))
    assert load().models[SMALL].cold_start_estimate_s == 123


def test_explicit_path_wins_over_the_env_var(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """§4: explicit `load_serving(path)` > `LOOM_SERVING_YAML` > the default path."""
    from_env = base()
    from_env["models"][BIG]["cold_start_estimate_s"] = 111
    monkeypatch.setenv(SERVING_ENV, str(write(tmp_path, from_env, "env-serving.yaml")))
    explicit = base()
    explicit["models"][BIG]["cold_start_estimate_s"] = 999
    path = write(tmp_path, explicit, "explicit-serving.yaml")
    assert load(path).models[BIG].cold_start_estimate_s == 999
    assert load(str(path)).models[BIG].cold_start_estimate_s == 999  # a str path too
    assert load().models[BIG].cold_start_estimate_s == 111


def test_file_without_a_servable_model_loads(tmp_path: Path) -> None:
    """A servable model missing from `models` is not a load error. (Its pool going
    `unavailable` with `not_configured` is the manager's, tested with it.)"""
    doc = base()
    del doc["models"][BIG]
    del doc["placement"]["container_disk_gb"][BIG]
    cfg = load(write(tmp_path, doc))
    assert set(cfg.models) == {SMALL}


@pytest.mark.parametrize("provider", ["fake", "local"])
def test_empty_egress_allowlist_is_fine_for_fake_and_local(tmp_path: Path, provider: str) -> None:
    doc = base()
    doc["provider"] = provider
    assert load(write(tmp_path, doc)).provider == provider


def test_runpod_with_an_egress_allowlist_loads(tmp_path: Path) -> None:
    doc = base()
    doc["provider"] = "runpod"
    doc["network"]["gateway_egress_cidrs"] = ["203.0.113.10/32"]
    cfg = load(write(tmp_path, doc))
    assert cfg.provider == "runpod"
    assert list(cfg.network.gateway_egress_cidrs) == ["203.0.113.10/32"]


# ------------------------------------------------------------------ rejects


def test_runpod_without_egress_allowlist_is_rejected(tmp_path: Path) -> None:
    doc = base()
    doc["provider"] = "runpod"
    assert "gateway_egress_cidrs" in rejected(write(tmp_path, doc))


def test_unknown_provider_is_rejected(tmp_path: Path) -> None:
    doc = base()
    doc["provider"] = "aws"
    assert "provider" in rejected(write(tmp_path, doc))


def test_min_replicas_above_max_is_rejected(tmp_path: Path) -> None:
    doc = base()
    doc["models"][SMALL]["min_replicas"] = 2
    doc["models"][SMALL]["max_replicas"] = 1
    assert "min_replicas" in rejected(write(tmp_path, doc))


@pytest.mark.parametrize("name", ["HF_TOKEN", "AWS_SECRET_ACCESS_KEY", "RUNPOD_API_KEY"])
def test_secret_names_in_engine_env_are_rejected(tmp_path: Path, name: str) -> None:
    doc = base()
    doc["models"][SMALL]["engine_env"] = {name: "x"}
    assert "engine_env" in rejected(write(tmp_path, doc))


def test_unservable_model_is_rejected(tmp_path: Path) -> None:
    """qwen3-8b-fp8 is in the registry (with a runpod instance type) but has no price."""
    doc = base()
    doc["models"]["qwen3-8b-fp8"] = copy.deepcopy(doc["models"][SMALL])
    assert "qwen3-8b-fp8" in rejected(write(tmp_path, doc))


def test_model_missing_from_the_registry_is_rejected(tmp_path: Path) -> None:
    doc = base()
    doc["models"]["no-such-model"] = copy.deepcopy(doc["models"][SMALL])
    assert "no-such-model" in rejected(write(tmp_path, doc))


def test_model_without_a_runpod_instance_type_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A registry (LOOM_MODELS_YAML) where qwen3-8b is servable but has no `runpod`
    entry in `hardware.instance_types`."""
    registry = yaml.safe_load((REPO_ROOT / "config" / "models.yaml").read_text(encoding="utf-8"))
    (row,) = [m for m in registry["models"] if m["id"] == SMALL]
    row["hardware"]["instance_types"] = {"aws": "g6e.xlarge"}
    row["clouds"] = ["aws"]
    models_yaml = write(tmp_path, registry, "models.yaml")
    assert load_registry(models_yaml).get(SMALL).hardware.instance_types.runpod is None
    monkeypatch.setenv(MODELS_ENV, str(models_yaml))
    msg = rejected(write(tmp_path, base()))
    assert SMALL in msg and "runpod" in msg


@pytest.mark.parametrize(
    ("where", "key"),
    [
        ((), "unexpected_top_level"),
        (("spend",), "monthly_cap_eur"),
        (("lifecycle",), "lease_ttl_minutes"),
        (("health",), "proxy_health_every_ms"),
        (("models", SMALL), "warm_spares"),
    ],
)
def test_unknown_keys_are_rejected(tmp_path: Path, where: tuple[str, ...], key: str) -> None:
    """`extra="forbid"` everywhere: a misspelt key is an error, not a silent default."""
    doc = base()
    section = doc
    for part in where:
        section = section[part]
    section[key] = 1
    assert key in rejected(write(tmp_path, doc))


# ------------------------------------------------------------------ the benchmark pin


def _cell_env(spec_file: str, model_id: str, variant: str) -> dict[str, str]:
    """The engine environment the benchmark launched `model_id` with, beyond what the
    registry row alone renders (`render_launch(row).env`)."""
    registry = load_registry()
    exp = load_experiment(REPO_ROOT / "bench" / "experiments" / spec_file)
    cells = [c for c in expand(exp, registry) if c.spec.id == model_id and c.variant == variant]
    assert cells, (spec_file, model_id, variant)
    envs = {tuple(sorted(c.launch.env.items())) for c in cells}
    assert len(envs) == 1, envs
    return dict(cells[0].launch.env)


@pytest.mark.parametrize(
    ("model_id", "spec_file", "variant"),
    [
        # 9f0853d7: the 70B FP8 row's price.
        (BIG, "llama-3.3-70b-h100-tp2-runpod.yaml", "vllm-tp2-fp8"),
        # 7a8237d0: qwen3-8b's price, the bf16 cell.
        (SMALL, "qwen3-8b-config-sweep-runpod.yaml", "bf16"),
    ],
)
def test_engine_env_equals_the_benchmarked_launch(
    model_id: str, spec_file: str, variant: str
) -> None:
    """The served launch stays the benchmarked one: the registry row's rendered env plus
    serving.yaml's `engine_env` is exactly the env the pricing run's cell launched with."""
    cfg = load()
    benchmarked = _cell_env(spec_file, model_id, variant)
    row_env = render_launch(load_registry().get(model_id)).env
    assert {**row_env, **dict(cfg.models[model_id].engine_env)} == benchmarked
    if model_id == BIG:
        # Not a trivial {} == {}: the 70B ran with its NCCL peer-to-peer setting.
        assert benchmarked.get("NCCL_P2P_LEVEL") == "NVL"
        assert dict(cfg.models[BIG].engine_env) == {"NCCL_P2P_LEVEL": "NVL"}
    else:
        assert dict(cfg.models[SMALL].engine_env) == {}
