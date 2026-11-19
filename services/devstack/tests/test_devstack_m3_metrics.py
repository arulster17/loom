"""Metric names agree across the alarms, the CloudWatch agent and the services (M3 spec
§5.3, §6.9, A9).

For every Prometheus series an alarm of §5.3 reads:
- `infra/aws/platform/alarms.tf` has an alarm on it in namespace `Loom/Platform`;
- the CloudWatch agent's allowlist (`deploy/cloudwatch-agent.json`, its
  `metric_selectors`) lets it through;
- the owning service exports it, with the alarm's labels: each service's registry is
  scraped in-process (gateway `GatewayMetrics`, pool manager `PoolmgrMetrics`, billing's
  module registry), and every counter an alarm reads with fixed label values is
  initialised to 0 for each value at process start (here: once the service's modules are
  imported and its metrics object exists).

Counter math (§5.3) is the implementer's choice, pinned here only for consistency: either
every counter alarm sums deltas (`statistic`/`stat = "Sum"`, no metric math) or every one
uses `RATE()`/`DIFF()` metric math on cumulative values; if the agent config says which it
sends (a key or value naming `delta` or `cumulative`), the alarms agree with it.

Conversely, every metric an alarm reads in `Loom/Platform` is one of §5.3's series (the
namespace holds only those). An alarm on a §5.3 series names the series in its resource
block (`metric_name`, possibly inside `metric_query { metric { ... } }`); its namespace may
be a literal or a simple `local.<name>` / `var.<name>` (resolved from the directory's
`.tf` files).
"""

from __future__ import annotations

import importlib
import json
import re
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[3]
ALARMS = ROOT / "infra" / "aws" / "platform" / "alarms.tf"
AGENT = ROOT / "deploy" / "cloudwatch-agent.json"

STRIPE_TYPES = (
    "checkout.session.completed",
    "checkout.session.async_payment_succeeded",
    "checkout.session.async_payment_failed",
    "charge.refunded",
    "charge.dispute.created",
)

# name -> (owner, kind, label names the alarm needs, label values initialised at start)
SERIES: dict[str, tuple[str, str, set[str], dict[str, tuple[str, ...]]]] = {
    "loom_poolmgr_creates_total": ("poolmgr", "counter", {"model", "result"}, {}),
    "loom_poolmgr_pool_reason": ("poolmgr", "gauge", {"model", "reason"}, {}),
    "loom_poolmgr_ca_expiry_seconds": ("poolmgr", "gauge", set(), {}),
    "loom_gateway_settle_spool_depth": ("gateway", "gauge", set(), {}),
    "loom_gateway_settle_failures_total": ("gateway", "counter", set(), {}),
    "loom_gateway_client_cert_expiry_seconds": ("gateway", "gauge", set(), {}),
    "loom_billing_stripe_unapplied_stale_total": (
        "billing",
        "counter",
        {"type"},
        {"type": STRIPE_TYPES},
    ),
    "loom_billing_stripe_unapplied_stale_events": ("billing", "gauge", set(), {}),
    "loom_billing_stripe_webhooks_total": (
        "billing",
        "counter",
        {"result"},
        {"result": ("signature_mismatch", "stale_timestamp")},
    ),
    "loom_billing_stripe_webhook_errors_total": (
        "billing",
        "counter",
        {"status"},
        {"status": ("503", "500")},
    ),
    "loom_billing_stripe_handler_failures_total": (
        "billing",
        "counter",
        {"path"},
        {"path": ("webhook", "retry", "pending_clawback")},
    ),
    "loom_billing_stripe_retry_last_success_seconds": ("billing", "gauge", set(), {}),
}
COUNTERS = sorted(n for n, (_, kind, _, _) in SERIES.items() if kind == "counter")


# ---------------------------------------------------------------- alarms.tf


PLATFORM = ALARMS.parent
NAMESPACE = "Loom/Platform"
EXPORTED = {*SERIES, "up"}


def strip_comments(text: str) -> str:
    """HCL without `#`, `//` and `/* */` comments; quoted strings are kept whole."""
    out, i, n = [], 0, len(text)
    while i < n:
        c = text[i]
        if c == '"':
            j = i + 1
            while j < n and text[j] != '"':
                j += 2 if text[j] == "\\" else 1
            out.append(text[i : j + 1])
            i = j + 1
        elif c == "#" or text.startswith("//", i):
            j = text.find("\n", i)
            i = n if j < 0 else j
        elif text.startswith("/*", i):
            j = text.find("*/", i + 2)
            i = n if j < 0 else j + 2
        else:
            out.append(c)
            i += 1
    return "".join(out)


def body_at(text: str, start: int) -> str:
    """The text from `start` (just after a `{`) to its matching `}`."""
    depth, i = 1, start
    while depth and i < len(text):
        depth += {"{": 1, "}": -1}.get(text[i], 0)
        i += 1
    return text[start : i - 1]


def labelled_blocks(text: str, header: str) -> dict[str, str]:
    return {
        m.group(1): body_at(text, m.end()) for m in re.finditer(header + r'\s+"([^"]+)"\s*\{', text)
    }


def alarm_blocks() -> dict[str, str]:
    assert ALARMS.is_file(), "infra/aws/platform/alarms.tf does not exist"
    text = strip_comments(ALARMS.read_text())
    blocks = labelled_blocks(text, r'\bresource\s+"aws_cloudwatch_metric_alarm"')
    assert blocks, "alarms.tf defines no aws_cloudwatch_metric_alarm"
    return blocks


def simple_values() -> dict[str, str]:
    """`local.<name>` and `var.<name>` -> their plain string value, from every `.tf` of
    infra/aws/platform (so a namespace written once as a local resolves)."""
    values: dict[str, str] = {}
    for tf in sorted(PLATFORM.glob("*.tf")):
        text = strip_comments(tf.read_text())
        for m in re.finditer(r"\blocals\s*\{", text):
            for a in re.finditer(r'^\s*(\w+)\s*=\s*"([^"$]*)"\s*$', body_at(text, m.end()), re.M):
                values.setdefault(f"local.{a.group(1)}", a.group(2))
        for name, body in labelled_blocks(text, r"\bvariable").items():
            d = re.search(r'^\s*default\s*=\s*"([^"$]*)"', body, re.M)
            if d:
                values.setdefault(f"var.{name}", d.group(1))
    return values


def namespaces(body: str, values: dict[str, str]) -> list[str | None]:
    """Each `namespace = ...` in an alarm block, resolved; None when it is not a literal or
    a simple local/variable."""
    found: list[str | None] = []
    for m in re.finditer(r'\bnamespace\s*=\s*(?:"([^"]*)"|([\w.]+))', body):
        found.append(m.group(1) if m.group(1) is not None else values.get(m.group(2)))
    return found


def alarms_reading(metric: str) -> dict[str, str]:
    pat = re.compile(rf'\bmetric_name\s*=\s*"{re.escape(metric)}"')
    return {k: b for k, b in alarm_blocks().items() if pat.search(b)}


@pytest.mark.parametrize("metric", [*sorted(SERIES), "up"])
def test_an_alarm_reads_it_in_the_loom_platform_namespace(metric: str) -> None:
    found = alarms_reading(metric)
    assert found, f"no alarm in alarms.tf reads {metric}"
    values = simple_values()
    for name, body in found.items():
        ns = namespaces(body, values)
        resolved = [n for n in ns if n is not None]
        assert ns, f"alarm {name} on {metric} names no namespace"
        assert all(n == NAMESPACE for n in resolved), (
            f"alarm {name} on {metric} is in {resolved}, not {NAMESPACE}"
        )
        if not resolved:
            assert NAMESPACE in values.values() or f'"{NAMESPACE}"' in ALARMS.read_text(), (
                f"alarm {name} on {metric}: its namespace does not resolve to {NAMESPACE}"
            )


def test_every_loom_platform_metric_an_alarm_reads_is_a_section_5_3_series() -> None:
    """The reverse direction: no alarm in Loom/Platform reads a name the services do not
    export (a typo, or a series §5.3 does not list)."""
    values = simple_values()
    seen, unknown = 0, {}
    for name, body in alarm_blocks().items():
        if NAMESPACE not in namespaces(body, values):
            continue
        for m in re.finditer(r'\bmetric_name\s*=\s*"([^"]+)"', body):
            seen += 1
            if m.group(1) not in EXPORTED:
                unknown[name] = m.group(1)
    assert seen, f"no alarm in alarms.tf reads a metric in {NAMESPACE}"
    assert not unknown, f"alarms read {NAMESPACE} metrics no service exports: {unknown}"


def counter_form(name: str, body: str) -> str:
    if re.search(r"\b(RATE|DIFF)\s*\(", body):
        return "cumulative"
    if re.search(r'\b(statistic|stat)\s*=\s*"Sum"', body):
        return "delta"
    raise AssertionError(
        f'{name}: a counter alarm that neither sums deltas (statistic/stat = "Sum") nor '
        "uses RATE()/DIFF() metric math on cumulative values"
    )


def agent_states() -> str | None:
    """What the agent config says it sends counters as, if it says anything."""
    assert AGENT.is_file(), "deploy/cloudwatch-agent.json does not exist"
    words: list[str] = []

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            for k, v in node.items():
                words.append(str(k))
                walk(v)
        elif isinstance(node, list):
            for x in node:
                walk(x)
        elif isinstance(node, str):
            words.append(node)

    walk(json.loads(AGENT.read_text()))
    said = " ".join(words).lower()
    if "delta" in said:
        return "delta"
    if "cumulative" in said:
        return "cumulative"
    return None


@pytest.mark.parametrize("metric", COUNTERS)
def test_counter_alarms_use_delta_or_cumulative_math(metric: str) -> None:
    found = alarms_reading(metric)
    assert found, f"no alarm in alarms.tf reads {metric}"
    for name, body in found.items():
        counter_form(name, body)


def test_counter_math_is_one_choice_and_matches_the_agent_config() -> None:
    forms = {
        f"{name} ({metric})": counter_form(name, body)
        for metric in COUNTERS
        for name, body in alarms_reading(metric).items()
    }
    assert forms, "no alarm reads a counter of §5.3"
    kinds = set(forms.values())
    assert len(kinds) == 1, f"counter alarms mix delta and cumulative math: {forms}"
    stated = agent_states()
    assert stated is None or stated in kinds, (
        f"the agent config sends counters as {stated}, the alarms assume {kinds.pop()}"
    )


# ---------------------------------------------------------------- the agent's allowlist


def selectors() -> list[str]:
    assert AGENT.is_file(), "deploy/cloudwatch-agent.json does not exist"
    doc = json.loads(AGENT.read_text())
    found: list[str] = []

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            for k, v in node.items():
                if k.endswith("metric_selectors") and isinstance(v, list):
                    found.extend(str(x) for x in v)
                walk(v)
        elif isinstance(node, list):
            for x in node:
                walk(x)

    walk(doc)
    assert found, "cloudwatch-agent.json has no metric_selectors (the allowlist)"
    return found


@pytest.mark.parametrize("metric", [*sorted(SERIES), "up"])
def test_the_agent_allowlist_lets_it_through(metric: str) -> None:
    assert any(re.fullmatch(s, metric) or re.match(s, metric) for s in selectors()), (
        f"{metric} matches no metric_selectors entry"
    )


def test_the_allowlist_is_narrow() -> None:
    """Only §5.3's series (custom-metric cost, ~15 metrics): no catch-all selector."""
    for s in selectors():
        for probe in ("loom_gateway_requests_total", "loom_billing_usage_lag", "go_goroutines"):
            assert not re.match(s, probe), f"selector {s!r} also sends {probe}"


# ---------------------------------------------------------------- the services export it


def scrape(owner: str) -> tuple[str, Any]:
    """(the text exposition, the registry) of `owner` in-process."""
    from prometheus_client import generate_latest

    if owner == "gateway":
        from loom_gateway.metrics import GatewayMetrics

        reg = GatewayMetrics().registry
    elif owner == "poolmgr":
        from loom_poolmgr.metrics import PoolmgrMetrics

        reg = PoolmgrMetrics().registry
    else:
        from prometheus_client import REGISTRY

        for mod in ("metrics", "api", "webhooks", "worker"):
            importlib.import_module(f"loom_billing.{mod}")
        reg = REGISTRY
    return generate_latest(reg).decode(), reg


def label_names(text: str, reg: Any, metric: str) -> set[str]:
    """The label names of `metric`'s samples in the exposition. A labelled series with no
    child yet (its label values come from config or later events, such as a pool's model)
    has no sample, and only then are the names read from the collector (private API)."""
    names: set[str] = set()
    sampled = False
    for m in re.finditer(rf"^{re.escape(metric)}(?:\{{([^}}]*)\}})? ", text, re.M):
        sampled = True
        names |= set(re.findall(r'(\w+)="', m.group(1) or ""))
    if sampled:
        return names
    for collector in getattr(reg, "_collector_to_names", {}):
        cnames = reg._collector_to_names[collector]
        if metric in cnames or metric.removesuffix("_total") in cnames:
            return set(getattr(collector, "_labelnames", ()))
    raise AssertionError(f"{metric}: no sample and no collector in the registry")


@pytest.mark.parametrize("metric", sorted(SERIES))
def test_the_owning_service_exports_it_with_its_labels(metric: str) -> None:
    owner, kind, labels, initialised = SERIES[metric]
    text, reg = scrape(owner)
    assert re.search(rf"^# TYPE {re.escape(metric)} {kind}$", text, re.M), (
        f"{owner} does not export {metric} as a {kind}"
    )
    found = label_names(text, reg, metric)
    assert labels <= found, f"{metric}: labels {found}, the alarm needs {labels}"
    for label, values in initialised.items():
        for value in values:
            sample = rf'^{re.escape(metric)}\{{[^}}]*{label}="{re.escape(value)}"[^}}]*\}} '
            assert re.search(sample, text, re.M), (
                f'{metric}{{{label}="{value}"}} is not initialised at process start'
            )
    if not labels:
        assert re.search(rf"^{re.escape(metric)} ", text, re.M), f"{metric} has no sample"
