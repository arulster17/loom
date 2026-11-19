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

Counter math (§5.3): the CloudWatch agent sends a Prometheus counter as the delta
between two scrapes (its documented type conversion for Prometheus metrics), so a
counter alarm sums the deltas (`statistic = "Sum"`) and uses no `RATE()`/`DIFF()`
metric math, which would apply to cumulative values.
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


def alarm_blocks() -> dict[str, str]:
    assert ALARMS.is_file(), "infra/aws/platform/alarms.tf does not exist"
    text = ALARMS.read_text()
    blocks = {}
    for m in re.finditer(r'resource\s+"aws_cloudwatch_metric_alarm"\s+"([^"]+)"\s*\{', text):
        depth, i = 1, m.end()
        while depth and i < len(text):
            depth += {"{": 1, "}": -1}.get(text[i], 0)
            i += 1
        blocks[m.group(1)] = text[m.end() : i - 1]
    assert blocks, "alarms.tf defines no aws_cloudwatch_metric_alarm"
    return blocks


def alarms_reading(metric: str) -> dict[str, str]:
    pat = re.compile(rf'metric_name\s*=\s*"{re.escape(metric)}"')
    return {k: b for k, b in alarm_blocks().items() if pat.search(b)}


@pytest.mark.parametrize("metric", [*sorted(SERIES), "up"])
def test_an_alarm_reads_it_in_the_loom_platform_namespace(metric: str) -> None:
    found = alarms_reading(metric)
    assert found, f"no alarm in alarms.tf reads {metric}"
    for name, body in found.items():
        assert '"Loom/Platform"' in body, f"alarm {name} on {metric} is not in Loom/Platform"


@pytest.mark.parametrize("metric", COUNTERS)
def test_counter_alarms_sum_the_agents_deltas(metric: str) -> None:
    for name, body in alarms_reading(metric).items():
        assert re.search(r'statistic\s*=\s*"Sum"', body), f"{name}: counter not summed"
        assert not re.search(r"\b(RATE|DIFF)\s*\(", body), (
            f"{name}: RATE/DIFF on a counter the agent already sends as deltas"
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


def label_names(reg: Any, metric: str) -> set[str]:
    for collector in getattr(reg, "_collector_to_names", {}):
        names = reg._collector_to_names[collector]
        if metric in names or metric.removesuffix("_total") in names:
            return set(getattr(collector, "_labelnames", ()))
    raise AssertionError(f"{metric}: no collector in the registry")


@pytest.mark.parametrize("metric", sorted(SERIES))
def test_the_owning_service_exports_it_with_its_labels(metric: str) -> None:
    owner, kind, labels, initialised = SERIES[metric]
    text, reg = scrape(owner)
    assert re.search(rf"^# TYPE {re.escape(metric)} {kind}$", text, re.M), (
        f"{owner} does not export {metric} as a {kind}"
    )
    assert labels <= label_names(reg, metric), f"{metric}: labels {label_names(reg, metric)}"
    for label, values in initialised.items():
        for value in values:
            sample = rf'^{re.escape(metric)}\{{[^}}]*{label}="{re.escape(value)}"[^}}]*\}} '
            assert re.search(sample, text, re.M), (
                f'{metric}{{{label}="{value}"}} is not initialised at process start'
            )
    if not labels:
        assert re.search(rf"^{re.escape(metric)} ", text, re.M), f"{metric} has no sample"
