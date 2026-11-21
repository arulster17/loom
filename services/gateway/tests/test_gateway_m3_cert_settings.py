"""`LOOM_GATEWAY_CERT_CHECK_S` (docs/phase1-m3-spec.md §7, §6.13): how often the gateway
checks its client-certificate file and `LOOM_POD_CA_CERT`; a gateway setting, default
60, a positive number (the reload tests set 1, `test_gateway_m3_cert_reload.py`).

Read from the environment as every gateway setting is (`GatewaySettings`, prefix
`LOOM_`, so the field is `gateway_cert_check_s`). No servers needed.
"""

from __future__ import annotations

import pydantic
import pytest

from loom_gateway.settings import GatewaySettings

VAR = "LOOM_GATEWAY_CERT_CHECK_S"


def test_cert_check_interval_defaults_to_60_s(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(VAR, raising=False)
    assert GatewaySettings().gateway_cert_check_s == 60


def test_cert_check_interval_is_read_from_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(VAR, "1")
    assert GatewaySettings().gateway_cert_check_s == 1


@pytest.mark.parametrize("value", ["0", "-1"])
def test_cert_check_interval_must_be_positive(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setenv(VAR, value)
    with pytest.raises(pydantic.ValidationError):
        GatewaySettings()


# fu-flake-proof: no-op line so flake-check runs this file (draft PR, never merged).
