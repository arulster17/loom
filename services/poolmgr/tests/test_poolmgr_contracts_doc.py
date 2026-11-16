"""docs/phase1-contracts.md documents the pool manager (docs/phase1-m2-spec.md §12): §2
its `LOOM_POOLMGR_*` settings (every one the code reads), `LOOM_SERVING_YAML`, and the
`poolmgr` service with its Prometheus port 9102; the Redis keys it owns (the admin queue
`loom:poolmgr:admin` and the reset times `loom:poolmgr:reset:<model>`, §11).

The settings are taken from `PoolmgrSettings` itself, so a setting added later must be
documented too. A doc table may abbreviate a sibling, as it does elsewhere
(`LOOM_GATEWAY_CLIENT_CERT` / `_KEY`): `LOOM_POOLMGR_CLIENT_KEY` counts as documented
when one line names a sibling in full (`LOOM_POOLMGR_CLIENT_CERT`) and the rest as
`` `_KEY` `` (the trailing `_`-components after the shared prefix, at least the last one).
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from loom_poolmgr.manager import ADMIN_QUEUE, RESET_KEY
from loom_poolmgr.settings import DEFAULT_METRICS_PORT, PoolmgrSettings

ROOT = Path(__file__).resolve().parents[3]
DOC = ROOT / "docs" / "phase1-contracts.md"
WORD = r"\b"


def doc() -> str:
    return DOC.read_text(encoding="utf-8")


def section(number: int) -> str:
    """The text of `## <number>. ...` up to the next `## `."""
    m = re.search(rf"^## {number}\. .*?(?=^## |\Z)", doc(), re.MULTILINE | re.DOTALL)
    assert m, f"no section {number} in {DOC.name}"
    return m.group(0)


def poolmgr_vars() -> list[str]:
    return sorted(
        f"LOOM_{name.upper()}"
        for name in PoolmgrSettings.model_fields
        if name.startswith("poolmgr_")
    )


def test_the_settings_list_is_not_empty() -> None:
    assert "LOOM_POOLMGR_STATE_DIR" in poolmgr_vars()


def abbreviated_in(var: str, text: str) -> bool:
    """`LOOM_POOLMGR_CLIENT_CERT` / `_KEY` documents `LOOM_POOLMGR_CLIENT_KEY`: a line with
    a full sibling sharing a prefix of `var`, and the rest of `var` in backticks."""
    parts = var.split("_")
    for k in range(2, len(parts)):  # keep at least LOOM_POOLMGR
        prefix = "_".join(parts[:k]) + "_"
        rest = "_" + "_".join(parts[k:])
        sibling = re.compile(WORD + re.escape(prefix) + r"[A-Z0-9_]+" + WORD)
        for line in text.splitlines():
            if f"`{rest}`" in line and sibling.search(line):
                return True
    return False


def test_the_abbreviation_rule() -> None:
    line = "| `LOOM_POOLMGR_CLIENT_CERT` / `_KEY` | none | pool manager |"
    assert abbreviated_in("LOOM_POOLMGR_CLIENT_KEY", line)
    assert not abbreviated_in("LOOM_POOLMGR_SSH_KEY", "| `LOOM_POOLMGR_CLIENT_CERT` |")
    assert not abbreviated_in("LOOM_POOLMGR_CA_KEY", "| `LOOM_GATEWAY_CA_CERT` / `_KEY` |")


@pytest.mark.parametrize("var", poolmgr_vars())
def test_every_poolmgr_setting_is_documented(var: str) -> None:
    text = doc()
    if re.search(WORD + var + WORD, text):
        return
    assert abbreviated_in(var, text), f"{var} is not in {DOC.name}"


def test_serving_yaml_variable_is_documented() -> None:
    assert "LOOM_SERVING_YAML" in section(2)


def test_the_poolmgr_service_and_its_metrics_port_are_in_the_services_table() -> None:
    assert DEFAULT_METRICS_PORT == 9102
    rows = [line for line in section(2).splitlines() if re.match(r"^\|\s*`?poolmgr\b", line)]
    assert rows, "no poolmgr row in §2's services table"
    assert any("9102" in row for row in rows), rows


def test_the_admin_queue_key_is_documented() -> None:
    assert ADMIN_QUEUE == "loom:poolmgr:admin"
    assert ADMIN_QUEUE in doc()


def test_the_reset_key_is_documented() -> None:
    key = RESET_KEY.format(model="<model>")
    assert key == "loom:poolmgr:reset:<model>"
    assert key in doc()
