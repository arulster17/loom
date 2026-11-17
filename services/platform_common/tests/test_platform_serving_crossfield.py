"""Cross-field rules of `config/serving.yaml` (docs/phase1-m2-spec.md §4, "Also (decided
after the wave 1 review ...)"):

- `lifecycle.lease_renew_every_s < lifecycle.lease_ttl_s`;
- `lifecycle.server_cert_renew_before_s < lifecycle.server_cert_days x 86400`;
- `health.bootstrap_stall_s <= health.bootstrap_timeout_s`;
- `spend.alert_fractions` unique.

(`idle_stop_s: 0` meaning "never idle-stopped" is the manager's behaviour, tested with it.)

As in test_platform_serving.py, every rejection starts from a file that loads and changes
one thing, and the error must name the field(s) of the rule it broke.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest
import yaml

SERVING_ENV = "LOOM_SERVING_YAML"
MODELS_ENV = "LOOM_MODELS_YAML"
SMALL = "qwen3-8b"
BIG = "llama-3.3-70b-instruct-fp8"
DAY_S = 86400

# The §4 example with provider `fake` (same as test_platform_serving.BASE; copied because
# test modules are not importable from one another here).
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


def with_lifecycle(**values: int) -> dict[str, Any]:
    doc = base()
    doc["lifecycle"].update(values)
    return doc


def with_health(**values: int) -> dict[str, Any]:
    doc = base()
    doc["health"].update(values)
    return doc


# ------------------------------------------------------------------ still loads


def test_base_file_loads(tmp_path: Path) -> None:
    """The starting point of every rejection below satisfies all four rules."""
    cfg = load(write(tmp_path, base()))
    assert cfg.lifecycle.lease_renew_every_s < cfg.lifecycle.lease_ttl_s
    assert cfg.lifecycle.server_cert_renew_before_s < cfg.lifecycle.server_cert_days * DAY_S
    assert cfg.health.bootstrap_stall_s <= cfg.health.bootstrap_timeout_s
    fractions = [float(x) for x in cfg.spend.alert_fractions]
    assert len(set(fractions)) == len(fractions)


def test_repo_serving_yaml_still_loads() -> None:
    """The committed config/serving.yaml (the default path) passes the new rules."""
    cfg = load()
    assert cfg.lifecycle.lease_renew_every_s < cfg.lifecycle.lease_ttl_s
    assert cfg.lifecycle.server_cert_renew_before_s < cfg.lifecycle.server_cert_days * DAY_S
    assert cfg.health.bootstrap_stall_s <= cfg.health.bootstrap_timeout_s
    fractions = [float(x) for x in cfg.spend.alert_fractions]
    assert len(set(fractions)) == len(fractions)


# ------------------------------------------------------------------ lease renewal


def test_lease_renew_just_below_ttl_loads(tmp_path: Path) -> None:
    cfg = load(write(tmp_path, with_lifecycle(lease_ttl_s=3600, lease_renew_every_s=3599)))
    assert (cfg.lifecycle.lease_ttl_s, cfg.lifecycle.lease_renew_every_s) == (3600, 3599)


@pytest.mark.parametrize(
    ("ttl", "renew"),
    [
        (3600, 3600),  # equal: the lease could lapse before it is renewed
        (3600, 3601),
        (300, 3600),
    ],
)
def test_lease_renew_not_below_ttl_is_rejected(tmp_path: Path, ttl: int, renew: int) -> None:
    msg = rejected(write(tmp_path, with_lifecycle(lease_ttl_s=ttl, lease_renew_every_s=renew)))
    assert "lease_renew_every_s" in msg
    assert "lease_ttl_s" in msg


# ------------------------------------------------------------------ server cert renewal


@pytest.mark.parametrize("days", [1, 7])
def test_cert_renew_just_below_lifetime_loads(tmp_path: Path, days: int) -> None:
    doc = with_lifecycle(server_cert_days=days, server_cert_renew_before_s=days * DAY_S - 1)
    cfg = load(write(tmp_path, doc))
    assert cfg.lifecycle.server_cert_renew_before_s == days * DAY_S - 1


@pytest.mark.parametrize(
    ("days", "renew_before"),
    [
        (7, 7 * DAY_S),  # equal: re-issued the moment it is issued
        (7, 7 * DAY_S + 1),
        (1, 172800),  # the committed 2-day margin with a 1-day cert
    ],
)
def test_cert_renew_not_below_lifetime_is_rejected(
    tmp_path: Path, days: int, renew_before: int
) -> None:
    doc = with_lifecycle(server_cert_days=days, server_cert_renew_before_s=renew_before)
    msg = rejected(write(tmp_path, doc))
    assert "server_cert_renew_before_s" in msg
    assert "server_cert_days" in msg


# ------------------------------------------------------------------ bootstrap stall


@pytest.mark.parametrize(
    ("stall", "timeout"),
    [
        (1800, 1800),  # equal is allowed (<=)
        (1799, 1800),
    ],
)
def test_bootstrap_stall_up_to_timeout_loads(tmp_path: Path, stall: int, timeout: int) -> None:
    doc = with_health(bootstrap_stall_s=stall, bootstrap_timeout_s=timeout)
    cfg = load(write(tmp_path, doc))
    assert (cfg.health.bootstrap_stall_s, cfg.health.bootstrap_timeout_s) == (stall, timeout)


@pytest.mark.parametrize(
    ("stall", "timeout"),
    [
        (1801, 1800),
        (600, 300),
    ],
)
def test_bootstrap_stall_above_timeout_is_rejected(
    tmp_path: Path, stall: int, timeout: int
) -> None:
    doc = with_health(bootstrap_stall_s=stall, bootstrap_timeout_s=timeout)
    msg = rejected(write(tmp_path, doc))
    assert "bootstrap_stall_s" in msg
    assert "bootstrap_timeout_s" in msg


# ------------------------------------------------------------------ alert fractions


@pytest.mark.parametrize(
    "fractions",
    [
        [1.0],
        [0.5, 0.8, 1.0],
        [0.25, 0.5, 0.75, 0.9, 1.0],
    ],
)
def test_distinct_alert_fractions_load(tmp_path: Path, fractions: list[float]) -> None:
    doc = base()
    doc["spend"]["alert_fractions"] = fractions
    cfg = load(write(tmp_path, doc))
    assert [float(x) for x in cfg.spend.alert_fractions] == fractions


@pytest.mark.parametrize(
    "fractions",
    [
        [0.5, 0.5, 1.0],  # adjacent duplicate
        [1.0, 1.0],  # all the same
        [0.8, 0.5, 0.8],  # duplicate not next to each other
        [0.5, 1, 1.0],  # equal as numbers though written differently
    ],
)
def test_duplicate_alert_fractions_are_rejected(tmp_path: Path, fractions: list[float]) -> None:
    doc = base()
    doc["spend"]["alert_fractions"] = fractions
    assert "alert_fractions" in rejected(write(tmp_path, doc))
