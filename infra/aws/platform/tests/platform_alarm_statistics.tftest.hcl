# A8 (docs/phase1-m3-spec.md §13, §5.3 "Statistic" column and the "Statistics" paragraph,
# user decision of 2026-11-19): each alarm reads its metric with the statistic §5.3 gives.
#
#   Maximum: an alarm on a value above a threshold ("> X for N min", "= 1 for N minutes",
#            "> X"): P2b, H2 (both), G1, S1b, D1 (both), R1 CPUUtilization; and S5's `m`
#            inside its metric math.
#   Minimum: a health alarm or one on a value below a threshold: P3, H1 (all three), G3,
#            R1 FreeStorageSpace.
#
# Pinned elsewhere and not repeated here: P1a/P1b Sum (platform_alarms), H3 Minimum
# (platform_healthcheck), the counter alarms' Sum-or-RATE/DIFF rule (platform_strength).
#
# Addresses as in platform_alarms. An alarm may read its metric directly (statistic) or
# through metric_query (metric[0].stat, as platform_alarms allows for Loom/Platform
# alarms); either way the set of statistics it reads must be exactly the one §5.3 names,
# so a missing statistic (e.g. extended_statistic only) or a second, different one fails.

mock_provider "aws" {
  source          = "./tests/mocks/aws"
  override_during = plan
}

mock_provider "aws" {
  alias           = "us_east_1"
  source          = "./tests/mocks/aws_us_east_1"
  override_during = plan
}

variables {
  domain          = "example.test"
  brand_name      = "Example Brand"
  ops_email       = "ops@example.test"
  budget_usd      = 75
  releases_bucket = "loom-test-releases"
}

# ---- Maximum: alarms on a value above a threshold -----------------------------------

run "a8_statistic_above_threshold_alarms_read_maximum" {
  command = plan

  assert {
    condition     = toset(compact(concat([aws_cloudwatch_metric_alarm.p2b.statistic], [for q in aws_cloudwatch_metric_alarm.p2b.metric_query : try(q.metric[0].stat, "")]))) == toset(["Maximum"])
    error_message = "P2b (model held at the cap, = 1 for 5 consecutive minutes) reads its gauge's Maximum (§5.3 Statistic); got statistic ${jsonencode(aws_cloudwatch_metric_alarm.p2b.statistic)}, metric_query stats ${jsonencode([for q in aws_cloudwatch_metric_alarm.p2b.metric_query : try(q.metric[0].stat, null)])}."
  }

  assert {
    condition     = toset(compact(concat([aws_cloudwatch_metric_alarm.h2_system.statistic], [for q in aws_cloudwatch_metric_alarm.h2_system.metric_query : try(q.metric[0].stat, "")]))) == toset(["Maximum"])
    error_message = "H2 (StatusCheckFailed_System, the recover action) reads Maximum (§5.3 Statistic); got statistic ${jsonencode(aws_cloudwatch_metric_alarm.h2_system.statistic)}, metric_query stats ${jsonencode([for q in aws_cloudwatch_metric_alarm.h2_system.metric_query : try(q.metric[0].stat, null)])}."
  }

  assert {
    condition     = toset(compact(concat([aws_cloudwatch_metric_alarm.h2_instance.statistic], [for q in aws_cloudwatch_metric_alarm.h2_instance.metric_query : try(q.metric[0].stat, "")]))) == toset(["Maximum"])
    error_message = "H2 (StatusCheckFailed_Instance, notify) reads Maximum (§5.3 Statistic); got statistic ${jsonencode(aws_cloudwatch_metric_alarm.h2_instance.statistic)}, metric_query stats ${jsonencode([for q in aws_cloudwatch_metric_alarm.h2_instance.metric_query : try(q.metric[0].stat, null)])}."
  }

  assert {
    condition     = toset(compact(concat([aws_cloudwatch_metric_alarm.g1.statistic], [for q in aws_cloudwatch_metric_alarm.g1.metric_query : try(q.metric[0].stat, "")]))) == toset(["Maximum"])
    error_message = "G1 (settle spool depth > 0 for 5 min) reads the gauge's Maximum (§5.3 Statistic: Minimum would let a spool that drains and refills hide); got statistic ${jsonencode(aws_cloudwatch_metric_alarm.g1.statistic)}, metric_query stats ${jsonencode([for q in aws_cloudwatch_metric_alarm.g1.metric_query : try(q.metric[0].stat, null)])}."
  }

  assert {
    condition     = toset(compact(concat([aws_cloudwatch_metric_alarm.s1b.statistic], [for q in aws_cloudwatch_metric_alarm.s1b.metric_query : try(q.metric[0].stat, "")]))) == toset(["Maximum"])
    error_message = "S1b (stuck events outstanding > 0 for 24 h) reads the gauge's Maximum (§5.3 Statistic); got statistic ${jsonencode(aws_cloudwatch_metric_alarm.s1b.statistic)}, metric_query stats ${jsonencode([for q in aws_cloudwatch_metric_alarm.s1b.metric_query : try(q.metric[0].stat, null)])}."
  }

  assert {
    condition     = toset(compact(concat([aws_cloudwatch_metric_alarm.d1_root.statistic], [for q in aws_cloudwatch_metric_alarm.d1_root.metric_query : try(q.metric[0].stat, "")]))) == toset(["Maximum"])
    error_message = "D1 (disk_used_percent > 80% on /) reads Maximum (§5.3 Statistic); got statistic ${jsonencode(aws_cloudwatch_metric_alarm.d1_root.statistic)}, metric_query stats ${jsonencode([for q in aws_cloudwatch_metric_alarm.d1_root.metric_query : try(q.metric[0].stat, null)])}."
  }

  assert {
    condition     = toset(compact(concat([aws_cloudwatch_metric_alarm.d1_data.statistic], [for q in aws_cloudwatch_metric_alarm.d1_data.metric_query : try(q.metric[0].stat, "")]))) == toset(["Maximum"])
    error_message = "D1 (disk_used_percent > 80% on /var/lib/loom) reads Maximum (§5.3 Statistic); got statistic ${jsonencode(aws_cloudwatch_metric_alarm.d1_data.statistic)}, metric_query stats ${jsonencode([for q in aws_cloudwatch_metric_alarm.d1_data.metric_query : try(q.metric[0].stat, null)])}."
  }

  assert {
    condition     = toset(compact(concat([aws_cloudwatch_metric_alarm.r1_cpu.statistic], [for q in aws_cloudwatch_metric_alarm.r1_cpu.metric_query : try(q.metric[0].stat, "")]))) == toset(["Maximum"])
    error_message = "R1 (RDS CPUUtilization > 80% for 15 min) reads Maximum (§5.3 Statistic: Maximum (CPU)); got statistic ${jsonencode(aws_cloudwatch_metric_alarm.r1_cpu.statistic)}, metric_query stats ${jsonencode([for q in aws_cloudwatch_metric_alarm.r1_cpu.metric_query : try(q.metric[0].stat, null)])}."
  }
}

# ---- Minimum: health alarms and alarms on a value below a threshold -----------------

run "a8_statistic_health_and_below_threshold_alarms_read_minimum" {
  command = plan

  assert {
    condition     = toset(compact(concat([aws_cloudwatch_metric_alarm.p3.statistic], [for q in aws_cloudwatch_metric_alarm.p3.metric_query : try(q.metric[0].stat, "")]))) == toset(["Minimum"])
    error_message = "P3 (pod CA expiry < 30 days) reads the gauge's Minimum (§5.3 Statistic); got statistic ${jsonencode(aws_cloudwatch_metric_alarm.p3.statistic)}, metric_query stats ${jsonencode([for q in aws_cloudwatch_metric_alarm.p3.metric_query : try(q.metric[0].stat, null)])}."
  }

  assert {
    condition     = toset(compact(concat([aws_cloudwatch_metric_alarm.h1_gateway.statistic], [for q in aws_cloudwatch_metric_alarm.h1_gateway.metric_query : try(q.metric[0].stat, "")]))) == toset(["Minimum"])
    error_message = "H1 (up{job=\"gateway\"} < 1 for 5 min) reads Minimum (§5.3 Statistic: one failed scrape in a period counts); got statistic ${jsonencode(aws_cloudwatch_metric_alarm.h1_gateway.statistic)}, metric_query stats ${jsonencode([for q in aws_cloudwatch_metric_alarm.h1_gateway.metric_query : try(q.metric[0].stat, null)])}."
  }

  assert {
    condition     = toset(compact(concat([aws_cloudwatch_metric_alarm.h1_billing.statistic], [for q in aws_cloudwatch_metric_alarm.h1_billing.metric_query : try(q.metric[0].stat, "")]))) == toset(["Minimum"])
    error_message = "H1 (up{job=\"billing\"} < 1 for 5 min) reads Minimum (§5.3 Statistic: one failed scrape in a period counts); got statistic ${jsonencode(aws_cloudwatch_metric_alarm.h1_billing.statistic)}, metric_query stats ${jsonencode([for q in aws_cloudwatch_metric_alarm.h1_billing.metric_query : try(q.metric[0].stat, null)])}."
  }

  assert {
    condition     = toset(compact(concat([aws_cloudwatch_metric_alarm.h1_poolmgr.statistic], [for q in aws_cloudwatch_metric_alarm.h1_poolmgr.metric_query : try(q.metric[0].stat, "")]))) == toset(["Minimum"])
    error_message = "H1 (up{job=\"poolmgr\"} < 1 for 5 min) reads Minimum (§5.3 Statistic: one failed scrape in a period counts); got statistic ${jsonencode(aws_cloudwatch_metric_alarm.h1_poolmgr.statistic)}, metric_query stats ${jsonencode([for q in aws_cloudwatch_metric_alarm.h1_poolmgr.metric_query : try(q.metric[0].stat, null)])}."
  }

  assert {
    condition     = toset(compact(concat([aws_cloudwatch_metric_alarm.g3.statistic], [for q in aws_cloudwatch_metric_alarm.g3.metric_query : try(q.metric[0].stat, "")]))) == toset(["Minimum"])
    error_message = "G3 (gateway client certificate expiry < 172,800 s) reads the gauge's Minimum (§5.3 Statistic); got statistic ${jsonencode(aws_cloudwatch_metric_alarm.g3.statistic)}, metric_query stats ${jsonencode([for q in aws_cloudwatch_metric_alarm.g3.metric_query : try(q.metric[0].stat, null)])}."
  }

  assert {
    condition     = toset(compact(concat([aws_cloudwatch_metric_alarm.r1_storage.statistic], [for q in aws_cloudwatch_metric_alarm.r1_storage.metric_query : try(q.metric[0].stat, "")]))) == toset(["Minimum"])
    error_message = "R1 (RDS FreeStorageSpace < 2 GB) reads Minimum (§5.3 Statistic: Minimum (storage)); got statistic ${jsonencode(aws_cloudwatch_metric_alarm.r1_storage.statistic)}, metric_query stats ${jsonencode([for q in aws_cloudwatch_metric_alarm.r1_storage.metric_query : try(q.metric[0].stat, null)])}."
  }
}

# ---- S5: the Maximum of m inside the metric math ------------------------------------

run "a8_statistic_s5_reads_maximum_of_m" {
  command = plan

  # Every query that reads the retry job's last-success gauge reads its Maximum (there is
  # at least one), and the alarm has no top-level statistic of its own (metric math).
  assert {
    condition = (
      aws_cloudwatch_metric_alarm.s5.statistic == null
      && toset([
        for q in aws_cloudwatch_metric_alarm.s5.metric_query : try(q.metric[0].stat, "")
        if try(q.metric[0].metric_name, "") == "loom_billing_stripe_retry_last_success_seconds"
      ]) == toset(["Maximum"])
    )
    error_message = "S5 (EPOCH(m) - m > 300) reads m, loom_billing_stripe_retry_last_success_seconds, as its Maximum inside the metric math, and has no top-level statistic (§5.3 Statistic: Maximum of m); got statistic ${jsonencode(aws_cloudwatch_metric_alarm.s5.statistic)}, metric_query ${jsonencode(aws_cloudwatch_metric_alarm.s5.metric_query)}."
  }
}
