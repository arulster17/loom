"""The M1 acceptance driver (`loom-m1-acceptance`, docs/phase1-m1-acceptance.md).

The helpers are tested here on every run. The acceptance itself needs the whole compose
stack up, so it is marked `acceptance` and deselected by default:

    LOOM_ADMIN_EMAILS=admin@m1.test docker compose up -d --build
    uv run pytest -m acceptance -s services/devstack/tests/test_devstack_acceptance.py

It also skips when the web app or the gateway is not reachable. Ports and the compose
project come from the same variables as the CLI (`LOOM_WEB_PORT`, ..., `COMPOSE_PROJECT_NAME`).
"""

from __future__ import annotations

import contextlib

import httpx
import pytest

from loom_devstack import acceptance as acc

NEXT_FORM = (
    '<form class="space-y-3" action="" encType="multipart/form-data" method="POST">'
    '<input type="hidden" name="$ACTION_REF_1"/>'
    '<input type="hidden" name="$ACTION_1:0" value="{&quot;id&quot;:&quot;60ab&quot;,'
    '&quot;bound&quot;:&quot;$@1&quot;}"/>'
    '<input type="hidden" name="$ACTION_1:1" value="[&quot;$undefined&quot;]"/>'
    '<input type="hidden" name="$ACTION_KEY" value="k91f"/>'
    '<input type="hidden" name="org_id" value="o-1"/>'
    '<input type="checkbox" name="enabled"/>'
    '<input type="checkbox" name="kept" checked=""/>'
    '<input id="email" type="email" required="" name="email"/>'
    '<button type="submit" name="op" value="approve">Approve</button>'
    '<button type="submit">Plain</button>'
    "</form><form><input name=other></form>"
)


def test_parse_forms_reads_next_progressive_enhancement_fields():
    forms = acc.parse_forms(NEXT_FORM)
    assert len(forms) == 2
    f = forms[0]
    assert f.value("$ACTION_1:0") == '{"id":"60ab","bound":"$@1"}'
    assert f.value("org_id") == "o-1"
    assert f.value("enabled") is None  # unchecked checkboxes are not submitted
    assert f.value("kept") == ""
    assert f.has("email", "org_id") and not f.has("other")
    assert f.buttons == [("op", "approve")]
    assert forms[1].has("other")


def test_visible_text_and_page_errors():
    page = (
        '<p class="mt-2 text-xs">$5.00<!-- --> added, <!-- -->$0.000118<!-- --> used.</p>'
        # A red button before the message must not be swept into it.
        '<button class="btn text-red-700">Revoke</button>'
        '<p class="text-sm text-red-700">Key &quot;x&quot; <b>not</b> found.</p>'
    )
    assert "$5.00 added, $0.000118 used." in acc.visible(page)
    assert acc.page_errors(page) == ['Key "x" not found.']


@pytest.mark.parametrize(
    ("micros", "exact", "cents"),
    [
        (5_000_000, "$5.00", "$5.00"),
        (118, "$0.000118", "$0.00"),
        (120, "$0.00012", "$0.00"),
        (77_000, "$0.077", "$0.08"),
        (4_999_882, "$4.999882", "$5.00"),
        (1_234_567_000, "$1,234.567", "$1,234.57"),
        (-1_234_567, "-$1.234567", "-$1.23"),
    ],
)
def test_money_formats_match_the_web_app(micros, exact, cents):
    # apps/web/src/lib/money.ts: formatUsdExact and formatUsd (2 decimals, half away).
    assert acc.usd_exact(micros) == exact
    assert acc.usd_cents(micros) == cents


def test_sse_events_stop_at_done():
    lines = iter(
        [
            ": comment",
            'data: {"a": 1}',
            "",
            'data: {"error": {"code": "upstream_lost"}}',
            "data: [DONE]",
            'data: {"after": true}',
        ]
    )
    assert list(acc.sse_events(lines)) == [
        {"a": 1},
        {"error": {"code": "upstream_lost"}},
        "[DONE]",
    ]


def test_stack_defaults_follow_the_compose_port_variables(monkeypatch):
    for var in ("LOOM_ACCEPT_WEB_URL", "LOOM_ACCEPT_GATEWAY_URL", "LOOM_ACCEPT_DATABASE_URL"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("LOOM_WEB_PORT", "13000")
    monkeypatch.setenv("LOOM_GATEWAY_PORT", "18080")
    monkeypatch.setenv("LOOM_PG_PORT", "25432")
    monkeypatch.setenv("COMPOSE_PROJECT_NAME", "loom-x")
    s = acc.stack_from_env()
    assert s.web == "http://localhost:13000"
    assert s.gateway == "http://localhost:18080/v1"
    assert s.gateway_root == "http://localhost:18080"
    assert s.database_url == "postgresql://loom:loom@localhost:25432/loom_platform"
    assert s.project == "loom-x"


@pytest.mark.acceptance
@pytest.mark.timeout(1200)
def test_m1_acceptance_on_the_compose_stack():
    stack = acc.stack_from_env()
    for url in (f"{stack.web}/signin", f"{stack.gateway_root}/health"):
        try:
            httpx.get(url, timeout=3, trust_env=False)
        except httpx.HTTPError as e:
            pytest.skip(f"compose stack not reachable ({url}): {e}")
    report = acc.Report()
    with contextlib.suppress(acc.AcceptanceError):  # recorded in the report
        acc.run(stack, acc.ITEMS, report)
    failed = [f"{i.item}: {'; '.join(i.notes)}" for i in report.items if i.status != "PASS"]
    assert report.items and not failed, failed
