# A8 (docs/phase1-m3-spec.md §13, §5.3): each §5.3 alarm exists with its metric,
# threshold, period and missing-data treatment; the P1 alarms' actions_enabled follows
# serving_reaper_enabled; every alarm notifies the SNS topic loom-platform-alerts (email
# to ops_email) on ALARM and OK. H3's own details are in platform_healthcheck.
#
# Addresses this file expects (the spec names the alarms, not their addresses): one
# aws_cloudwatch_metric_alarm per row of §5.3, named by the row's id in lower case, and one
# per part where a row has several:
#   p1a p1b p2a p2b p3 h1_gateway h1_billing h1_poolmgr h2_system h2_instance h3
#   g1 g2 g3 s1a s1b s2a s2b s2c s4 s5 d1_root d1_data r1_storage r1_cpu
#   aws_sns_topic.alerts, aws_sns_topic_subscription.ops_email
#   aws_lambda_function.serving_reaper, aws_instance.platform, aws_db_instance.platform
#
# Counter math (cumulative vs delta, §5.3, pinned by A9's pytest) is the implementer's
# choice: a Loom/Platform alarm may read its metric directly (metric_name, period) or
# through metric_query (metric math); these tests find the metric, dimensions and period
# in either place. Windows: "> 0 over 5 min" / ">= 1 in 15 min" / "over 1 h" mean any
# datapoint in the window (period x evaluation_periods = window, 1 datapoint to alarm);
# "for 5 min" / "for 24 h" / "for 15 min" mean every datapoint in the window.

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
  domain     = "example.test"
  brand_name = "Example Brand"
  ops_email  = "ops@example.test"
  budget_usd = 75
}

run "topic_and_every_alarm_notifies_it" {
  command = plan

  assert {
    condition     = aws_sns_topic.alerts.name == "loom-platform-alerts"
    error_message = "The alerts topic is loom-platform-alerts."
  }

  assert {
    condition = (
      aws_sns_topic_subscription.ops_email.protocol == "email"
      && aws_sns_topic_subscription.ops_email.endpoint == "ops@example.test"
      && aws_sns_topic_subscription.ops_email.topic_arn == aws_sns_topic.alerts.arn
    )
    error_message = "loom-platform-alerts has an email subscription to var.ops_email."
  }

  assert {
    condition = alltrue([
      for id, a in {
        p1a         = aws_cloudwatch_metric_alarm.p1a
        p1b         = aws_cloudwatch_metric_alarm.p1b
        p2a         = aws_cloudwatch_metric_alarm.p2a
        p2b         = aws_cloudwatch_metric_alarm.p2b
        p3          = aws_cloudwatch_metric_alarm.p3
        h1_gateway  = aws_cloudwatch_metric_alarm.h1_gateway
        h1_billing  = aws_cloudwatch_metric_alarm.h1_billing
        h1_poolmgr  = aws_cloudwatch_metric_alarm.h1_poolmgr
        h2_system   = aws_cloudwatch_metric_alarm.h2_system
        h2_instance = aws_cloudwatch_metric_alarm.h2_instance
        h3          = aws_cloudwatch_metric_alarm.h3
        g1          = aws_cloudwatch_metric_alarm.g1
        g2          = aws_cloudwatch_metric_alarm.g2
        g3          = aws_cloudwatch_metric_alarm.g3
        s1a         = aws_cloudwatch_metric_alarm.s1a
        s1b         = aws_cloudwatch_metric_alarm.s1b
        s2a         = aws_cloudwatch_metric_alarm.s2a
        s2b         = aws_cloudwatch_metric_alarm.s2b
        s2c         = aws_cloudwatch_metric_alarm.s2c
        s4          = aws_cloudwatch_metric_alarm.s4
        s5          = aws_cloudwatch_metric_alarm.s5
        d1_root     = aws_cloudwatch_metric_alarm.d1_root
        d1_data     = aws_cloudwatch_metric_alarm.d1_data
        r1_storage  = aws_cloudwatch_metric_alarm.r1_storage
        r1_cpu      = aws_cloudwatch_metric_alarm.r1_cpu
      } :
      contains(coalesce(a.alarm_actions, []), aws_sns_topic.alerts.arn) && contains(coalesce(a.ok_actions, []), aws_sns_topic.alerts.arn)
    ])
    error_message = "Every §5.3 alarm has loom-platform-alerts in both alarm_actions and ok_actions."
  }

  assert {
    condition = alltrue([
      for a in [
        aws_cloudwatch_metric_alarm.p2a, aws_cloudwatch_metric_alarm.p2b, aws_cloudwatch_metric_alarm.p3,
        aws_cloudwatch_metric_alarm.h1_gateway, aws_cloudwatch_metric_alarm.h1_billing, aws_cloudwatch_metric_alarm.h1_poolmgr,
        aws_cloudwatch_metric_alarm.h2_system, aws_cloudwatch_metric_alarm.h2_instance, aws_cloudwatch_metric_alarm.h3,
        aws_cloudwatch_metric_alarm.g1, aws_cloudwatch_metric_alarm.g2, aws_cloudwatch_metric_alarm.g3,
        aws_cloudwatch_metric_alarm.s1a, aws_cloudwatch_metric_alarm.s1b, aws_cloudwatch_metric_alarm.s2a,
        aws_cloudwatch_metric_alarm.s2b, aws_cloudwatch_metric_alarm.s2c, aws_cloudwatch_metric_alarm.s4,
        aws_cloudwatch_metric_alarm.s5, aws_cloudwatch_metric_alarm.d1_root, aws_cloudwatch_metric_alarm.d1_data,
        aws_cloudwatch_metric_alarm.r1_storage, aws_cloudwatch_metric_alarm.r1_cpu,
      ] : a.actions_enabled != false
    ])
    error_message = "Only the P1 alarms follow serving_reaper_enabled; every other alarm's actions are enabled."
  }

  assert {
    condition     = aws_cloudwatch_metric_alarm.p1a.actions_enabled == false && aws_cloudwatch_metric_alarm.p1b.actions_enabled == false
    error_message = "With serving_reaper_enabled = false (the default) the P1 alarms' actions are disabled."
  }
}

run "p1_actions_follow_serving_reaper_enabled" {
  command = plan

  variables {
    serving_reaper_enabled = true
  }

  assert {
    condition     = aws_cloudwatch_metric_alarm.p1a.actions_enabled == true && aws_cloudwatch_metric_alarm.p1b.actions_enabled == true
    error_message = "With serving_reaper_enabled = true the P1 alarms' actions are enabled."
  }
}

run "p1_serving_reaper" {
  command = plan

  # P1a: AWS/Lambda Errors (FunctionName = serving reaper), Sum >= 1 over 1 h, not breaching.
  assert {
    condition = (
      aws_cloudwatch_metric_alarm.p1a.namespace == "AWS/Lambda"
      && aws_cloudwatch_metric_alarm.p1a.metric_name == "Errors"
      && aws_cloudwatch_metric_alarm.p1a.statistic == "Sum"
      && contains([aws_lambda_function.serving_reaper.function_name, aws_lambda_function.serving_reaper.id], lookup(coalesce(aws_cloudwatch_metric_alarm.p1a.dimensions, {}), "FunctionName", ""))
    )
    error_message = "P1a reads AWS/Lambda Errors, Sum, for FunctionName = the serving reaper."
  }

  assert {
    condition     = aws_cloudwatch_metric_alarm.p1a.comparison_operator == "GreaterThanOrEqualToThreshold" && aws_cloudwatch_metric_alarm.p1a.threshold == 1
    error_message = "P1a alarms at >= 1."
  }

  assert {
    condition = (
      aws_cloudwatch_metric_alarm.p1a.period * aws_cloudwatch_metric_alarm.p1a.evaluation_periods == 3600
      && coalesce(aws_cloudwatch_metric_alarm.p1a.datapoints_to_alarm, aws_cloudwatch_metric_alarm.p1a.evaluation_periods) == 1
      && aws_cloudwatch_metric_alarm.p1a.treat_missing_data == "notBreaching"
    )
    error_message = "P1a: over 1 h, missing data not breaching."
  }

  # P1b: AWS/Lambda Invocations, Sum < 1 over 2 consecutive 15-min periods, breaching.
  assert {
    condition = (
      aws_cloudwatch_metric_alarm.p1b.namespace == "AWS/Lambda"
      && aws_cloudwatch_metric_alarm.p1b.metric_name == "Invocations"
      && aws_cloudwatch_metric_alarm.p1b.statistic == "Sum"
      && contains([aws_lambda_function.serving_reaper.function_name, aws_lambda_function.serving_reaper.id], lookup(coalesce(aws_cloudwatch_metric_alarm.p1b.dimensions, {}), "FunctionName", ""))
    )
    error_message = "P1b reads AWS/Lambda Invocations, Sum, for FunctionName = the serving reaper."
  }

  assert {
    condition     = aws_cloudwatch_metric_alarm.p1b.comparison_operator == "LessThanThreshold" && aws_cloudwatch_metric_alarm.p1b.threshold == 1
    error_message = "P1b alarms at < 1."
  }

  assert {
    condition = (
      aws_cloudwatch_metric_alarm.p1b.period == 900
      && aws_cloudwatch_metric_alarm.p1b.evaluation_periods == 2
      && coalesce(aws_cloudwatch_metric_alarm.p1b.datapoints_to_alarm, 2) == 2
      && aws_cloudwatch_metric_alarm.p1b.treat_missing_data == "breaching"
    )
    error_message = "P1b: 2 consecutive 15-min periods, missing data breaching."
  }
}

run "p2_p3_pool_manager" {
  command = plan

  # P2a: loom_poolmgr_creates_total{result="spend_cap"} > 0 over 5 min, not breaching.
  assert {
    condition = (
      contains(compact(concat([aws_cloudwatch_metric_alarm.p2a.metric_name], [for q in aws_cloudwatch_metric_alarm.p2a.metric_query : try(q.metric[0].metric_name, "")])), "loom_poolmgr_creates_total")
      && toset(compact(concat([aws_cloudwatch_metric_alarm.p2a.namespace], [for q in aws_cloudwatch_metric_alarm.p2a.metric_query : try(q.metric[0].namespace, "")]))) == toset(["Loom/Platform"])
      && anytrue([for d in concat([aws_cloudwatch_metric_alarm.p2a.dimensions], [for q in aws_cloudwatch_metric_alarm.p2a.metric_query : try(q.metric[0].dimensions, null)]) : lookup(coalesce(d, {}), "result", "") == "spend_cap"])
    )
    error_message = "P2a reads Loom/Platform loom_poolmgr_creates_total with result = spend_cap."
  }

  assert {
    condition     = aws_cloudwatch_metric_alarm.p2a.comparison_operator == "GreaterThanThreshold" && aws_cloudwatch_metric_alarm.p2a.threshold == 0
    error_message = "P2a alarms at > 0."
  }

  assert {
    condition = (
      try(tonumber(one(distinct(compact(concat([try(tostring(aws_cloudwatch_metric_alarm.p2a.period), "")], [for q in aws_cloudwatch_metric_alarm.p2a.metric_query : try(tostring(q.metric[0].period), "")]))))), 0) * aws_cloudwatch_metric_alarm.p2a.evaluation_periods == 300
      && coalesce(aws_cloudwatch_metric_alarm.p2a.datapoints_to_alarm, aws_cloudwatch_metric_alarm.p2a.evaluation_periods) == 1
      && aws_cloudwatch_metric_alarm.p2a.treat_missing_data == "notBreaching"
    )
    error_message = "P2a: over 5 min, missing data not breaching."
  }

  # P2b: loom_poolmgr_pool_reason{reason="spend_cap"} = 1 for 5 consecutive minutes.
  assert {
    condition = (
      contains(compact(concat([aws_cloudwatch_metric_alarm.p2b.metric_name], [for q in aws_cloudwatch_metric_alarm.p2b.metric_query : try(q.metric[0].metric_name, "")])), "loom_poolmgr_pool_reason")
      && toset(compact(concat([aws_cloudwatch_metric_alarm.p2b.namespace], [for q in aws_cloudwatch_metric_alarm.p2b.metric_query : try(q.metric[0].namespace, "")]))) == toset(["Loom/Platform"])
      && anytrue([for d in concat([aws_cloudwatch_metric_alarm.p2b.dimensions], [for q in aws_cloudwatch_metric_alarm.p2b.metric_query : try(q.metric[0].dimensions, null)]) : lookup(coalesce(d, {}), "reason", "") == "spend_cap"])
    )
    error_message = "P2b reads Loom/Platform loom_poolmgr_pool_reason with reason = spend_cap."
  }

  assert {
    condition = (
      (aws_cloudwatch_metric_alarm.p2b.comparison_operator == "GreaterThanOrEqualToThreshold" && aws_cloudwatch_metric_alarm.p2b.threshold == 1)
      || (aws_cloudwatch_metric_alarm.p2b.comparison_operator == "GreaterThanThreshold" && aws_cloudwatch_metric_alarm.p2b.threshold == 0)
    )
    error_message = "P2b alarms when the gauge is 1 (>= 1, or > 0)."
  }

  assert {
    condition = (
      try(tonumber(one(distinct(compact(concat([try(tostring(aws_cloudwatch_metric_alarm.p2b.period), "")], [for q in aws_cloudwatch_metric_alarm.p2b.metric_query : try(tostring(q.metric[0].period), "")]))))), 0) * aws_cloudwatch_metric_alarm.p2b.evaluation_periods == 300
      && coalesce(aws_cloudwatch_metric_alarm.p2b.datapoints_to_alarm, aws_cloudwatch_metric_alarm.p2b.evaluation_periods) == aws_cloudwatch_metric_alarm.p2b.evaluation_periods
      && aws_cloudwatch_metric_alarm.p2b.treat_missing_data == "notBreaching"
    )
    error_message = "P2b: for 5 consecutive minutes (every datapoint), missing data not breaching."
  }

  # P3: loom_poolmgr_ca_expiry_seconds < 2,592,000, 1-h period, breaching.
  assert {
    condition = (
      contains(compact(concat([aws_cloudwatch_metric_alarm.p3.metric_name], [for q in aws_cloudwatch_metric_alarm.p3.metric_query : try(q.metric[0].metric_name, "")])), "loom_poolmgr_ca_expiry_seconds")
      && toset(compact(concat([aws_cloudwatch_metric_alarm.p3.namespace], [for q in aws_cloudwatch_metric_alarm.p3.metric_query : try(q.metric[0].namespace, "")]))) == toset(["Loom/Platform"])
    )
    error_message = "P3 reads Loom/Platform loom_poolmgr_ca_expiry_seconds."
  }

  assert {
    condition     = aws_cloudwatch_metric_alarm.p3.comparison_operator == "LessThanThreshold" && aws_cloudwatch_metric_alarm.p3.threshold == 2592000
    error_message = "P3 alarms at < 2,592,000 s (30 days)."
  }

  assert {
    condition = (
      toset(compact(concat([try(tostring(aws_cloudwatch_metric_alarm.p3.period), "")], [for q in aws_cloudwatch_metric_alarm.p3.metric_query : try(tostring(q.metric[0].period), "")]))) == toset(["3600"])
      && aws_cloudwatch_metric_alarm.p3.treat_missing_data == "breaching"
    )
    error_message = "P3: 1-h period, missing data breaching (a manager that refuses to start publishes nothing)."
  }
}

run "h1_h2_heartbeat_and_instance" {
  command = plan

  # H1: up{job=...} < 1 for 5 min, breaching; one alarm per job.
  assert {
    condition = alltrue([
      for job, a in {
        gateway = aws_cloudwatch_metric_alarm.h1_gateway
        billing = aws_cloudwatch_metric_alarm.h1_billing
        poolmgr = aws_cloudwatch_metric_alarm.h1_poolmgr
      } :
      contains(compact(concat([a.metric_name], [for q in a.metric_query : try(q.metric[0].metric_name, "")])), "up")
      && toset(compact(concat([a.namespace], [for q in a.metric_query : try(q.metric[0].namespace, "")]))) == toset(["Loom/Platform"])
      && anytrue([for d in concat([a.dimensions], [for q in a.metric_query : try(q.metric[0].dimensions, null)]) : lookup(coalesce(d, {}), "job", "") == job])
    ])
    error_message = "H1 reads Loom/Platform up with job = gateway, billing and poolmgr (h1_<job>)."
  }

  assert {
    condition = alltrue([
      for a in [aws_cloudwatch_metric_alarm.h1_gateway, aws_cloudwatch_metric_alarm.h1_billing, aws_cloudwatch_metric_alarm.h1_poolmgr] :
      a.comparison_operator == "LessThanThreshold" && a.threshold == 1
    ])
    error_message = "H1 alarms at < 1."
  }

  assert {
    condition = alltrue([
      for a in [aws_cloudwatch_metric_alarm.h1_gateway, aws_cloudwatch_metric_alarm.h1_billing, aws_cloudwatch_metric_alarm.h1_poolmgr] :
      try(tonumber(one(distinct(compact(concat([try(tostring(a.period), "")], [for q in a.metric_query : try(tostring(q.metric[0].period), "")]))))), 0) * a.evaluation_periods == 300
      && coalesce(a.datapoints_to_alarm, a.evaluation_periods) == a.evaluation_periods
      && a.treat_missing_data == "breaching"
    ])
    error_message = "H1: for 5 min (every datapoint), missing data breaching (box or agent down)."
  }

  # H2: AWS/EC2 StatusCheckFailed_System -> recover (platform_instance), _Instance -> notify; breaching.
  assert {
    condition = alltrue([
      for m, a in {
        StatusCheckFailed_System   = aws_cloudwatch_metric_alarm.h2_system
        StatusCheckFailed_Instance = aws_cloudwatch_metric_alarm.h2_instance
      } :
      a.namespace == "AWS/EC2" && a.metric_name == m
      && lookup(coalesce(a.dimensions, {}), "InstanceId", "") == aws_instance.platform.id
      && ((a.comparison_operator == "GreaterThanOrEqualToThreshold" && a.threshold == 1) || (a.comparison_operator == "GreaterThanThreshold" && a.threshold == 0))
      && a.treat_missing_data == "breaching"
    ])
    error_message = "H2: AWS/EC2 StatusCheckFailed_System (h2_system) and StatusCheckFailed_Instance (h2_instance) for the instance, alarming on a failed check, missing data breaching."
  }
}

run "g_gateway" {
  command = plan

  # G1: loom_gateway_settle_spool_depth > 0 for 5 min, not breaching.
  assert {
    condition = (
      contains(compact(concat([aws_cloudwatch_metric_alarm.g1.metric_name], [for q in aws_cloudwatch_metric_alarm.g1.metric_query : try(q.metric[0].metric_name, "")])), "loom_gateway_settle_spool_depth")
      && toset(compact(concat([aws_cloudwatch_metric_alarm.g1.namespace], [for q in aws_cloudwatch_metric_alarm.g1.metric_query : try(q.metric[0].namespace, "")]))) == toset(["Loom/Platform"])
      && aws_cloudwatch_metric_alarm.g1.comparison_operator == "GreaterThanThreshold" && aws_cloudwatch_metric_alarm.g1.threshold == 0
    )
    error_message = "G1 reads Loom/Platform loom_gateway_settle_spool_depth, alarming at > 0."
  }

  assert {
    condition = (
      try(tonumber(one(distinct(compact(concat([try(tostring(aws_cloudwatch_metric_alarm.g1.period), "")], [for q in aws_cloudwatch_metric_alarm.g1.metric_query : try(tostring(q.metric[0].period), "")]))))), 0) * aws_cloudwatch_metric_alarm.g1.evaluation_periods == 300
      && coalesce(aws_cloudwatch_metric_alarm.g1.datapoints_to_alarm, aws_cloudwatch_metric_alarm.g1.evaluation_periods) == aws_cloudwatch_metric_alarm.g1.evaluation_periods
      && aws_cloudwatch_metric_alarm.g1.treat_missing_data == "notBreaching"
    )
    error_message = "G1: for 5 min (every datapoint), missing data not breaching."
  }

  # G2: loom_gateway_settle_failures_total increased (alert at once), not breaching.
  assert {
    condition = (
      contains(compact(concat([aws_cloudwatch_metric_alarm.g2.metric_name], [for q in aws_cloudwatch_metric_alarm.g2.metric_query : try(q.metric[0].metric_name, "")])), "loom_gateway_settle_failures_total")
      && toset(compact(concat([aws_cloudwatch_metric_alarm.g2.namespace], [for q in aws_cloudwatch_metric_alarm.g2.metric_query : try(q.metric[0].namespace, "")]))) == toset(["Loom/Platform"])
      && ((aws_cloudwatch_metric_alarm.g2.comparison_operator == "GreaterThanThreshold" && aws_cloudwatch_metric_alarm.g2.threshold == 0) || (aws_cloudwatch_metric_alarm.g2.comparison_operator == "GreaterThanOrEqualToThreshold" && aws_cloudwatch_metric_alarm.g2.threshold == 1))
      && coalesce(aws_cloudwatch_metric_alarm.g2.datapoints_to_alarm, aws_cloudwatch_metric_alarm.g2.evaluation_periods) == 1
      && aws_cloudwatch_metric_alarm.g2.treat_missing_data == "notBreaching"
    )
    error_message = "G2 reads Loom/Platform loom_gateway_settle_failures_total and alarms on any increase at once (one datapoint), missing data not breaching."
  }

  # G3: loom_gateway_client_cert_expiry_seconds < 172,800, breaching.
  assert {
    condition = (
      contains(compact(concat([aws_cloudwatch_metric_alarm.g3.metric_name], [for q in aws_cloudwatch_metric_alarm.g3.metric_query : try(q.metric[0].metric_name, "")])), "loom_gateway_client_cert_expiry_seconds")
      && toset(compact(concat([aws_cloudwatch_metric_alarm.g3.namespace], [for q in aws_cloudwatch_metric_alarm.g3.metric_query : try(q.metric[0].namespace, "")]))) == toset(["Loom/Platform"])
      && aws_cloudwatch_metric_alarm.g3.comparison_operator == "LessThanThreshold" && aws_cloudwatch_metric_alarm.g3.threshold == 172800
      && aws_cloudwatch_metric_alarm.g3.treat_missing_data == "breaching"
    )
    error_message = "G3 reads Loom/Platform loom_gateway_client_cert_expiry_seconds, alarming at < 172,800 s, missing data breaching."
  }
}

run "s_stripe" {
  command = plan

  # S1a: loom_billing_stripe_unapplied_stale_total > 0 over 5 min, not breaching.
  assert {
    condition = (
      contains(compact(concat([aws_cloudwatch_metric_alarm.s1a.metric_name], [for q in aws_cloudwatch_metric_alarm.s1a.metric_query : try(q.metric[0].metric_name, "")])), "loom_billing_stripe_unapplied_stale_total")
      && toset(compact(concat([aws_cloudwatch_metric_alarm.s1a.namespace], [for q in aws_cloudwatch_metric_alarm.s1a.metric_query : try(q.metric[0].namespace, "")]))) == toset(["Loom/Platform"])
      && aws_cloudwatch_metric_alarm.s1a.comparison_operator == "GreaterThanThreshold" && aws_cloudwatch_metric_alarm.s1a.threshold == 0
    )
    error_message = "S1a reads Loom/Platform loom_billing_stripe_unapplied_stale_total, alarming at > 0."
  }

  assert {
    condition = (
      try(tonumber(one(distinct(compact(concat([try(tostring(aws_cloudwatch_metric_alarm.s1a.period), "")], [for q in aws_cloudwatch_metric_alarm.s1a.metric_query : try(tostring(q.metric[0].period), "")]))))), 0) * aws_cloudwatch_metric_alarm.s1a.evaluation_periods == 300
      && coalesce(aws_cloudwatch_metric_alarm.s1a.datapoints_to_alarm, aws_cloudwatch_metric_alarm.s1a.evaluation_periods) == 1
      && aws_cloudwatch_metric_alarm.s1a.treat_missing_data == "notBreaching"
    )
    error_message = "S1a: over 5 min, missing data not breaching."
  }

  # S1b: loom_billing_stripe_unapplied_stale_events > 0 for 24 h, not breaching.
  assert {
    condition = (
      contains(compact(concat([aws_cloudwatch_metric_alarm.s1b.metric_name], [for q in aws_cloudwatch_metric_alarm.s1b.metric_query : try(q.metric[0].metric_name, "")])), "loom_billing_stripe_unapplied_stale_events")
      && toset(compact(concat([aws_cloudwatch_metric_alarm.s1b.namespace], [for q in aws_cloudwatch_metric_alarm.s1b.metric_query : try(q.metric[0].namespace, "")]))) == toset(["Loom/Platform"])
      && aws_cloudwatch_metric_alarm.s1b.comparison_operator == "GreaterThanThreshold" && aws_cloudwatch_metric_alarm.s1b.threshold == 0
    )
    error_message = "S1b reads Loom/Platform loom_billing_stripe_unapplied_stale_events, alarming at > 0."
  }

  assert {
    condition = (
      try(tonumber(one(distinct(compact(concat([try(tostring(aws_cloudwatch_metric_alarm.s1b.period), "")], [for q in aws_cloudwatch_metric_alarm.s1b.metric_query : try(tostring(q.metric[0].period), "")]))))), 0) * aws_cloudwatch_metric_alarm.s1b.evaluation_periods == 86400
      && coalesce(aws_cloudwatch_metric_alarm.s1b.datapoints_to_alarm, aws_cloudwatch_metric_alarm.s1b.evaluation_periods) == aws_cloudwatch_metric_alarm.s1b.evaluation_periods
      && aws_cloudwatch_metric_alarm.s1b.treat_missing_data == "notBreaching"
    )
    error_message = "S1b: for 24 h (every datapoint), missing data not breaching."
  }

  # S2a / S2b: loom_billing_stripe_webhooks_total{result=...} >= 1 in 15 min, not breaching.
  assert {
    condition = alltrue([
      for result, a in {
        signature_mismatch = aws_cloudwatch_metric_alarm.s2a
        stale_timestamp    = aws_cloudwatch_metric_alarm.s2b
      } :
      contains(compact(concat([a.metric_name], [for q in a.metric_query : try(q.metric[0].metric_name, "")])), "loom_billing_stripe_webhooks_total")
      && toset(compact(concat([a.namespace], [for q in a.metric_query : try(q.metric[0].namespace, "")]))) == toset(["Loom/Platform"])
      && anytrue([for d in concat([a.dimensions], [for q in a.metric_query : try(q.metric[0].dimensions, null)]) : lookup(coalesce(d, {}), "result", "") == result])
    ])
    error_message = "S2a reads Loom/Platform loom_billing_stripe_webhooks_total with result = signature_mismatch; S2b with result = stale_timestamp."
  }

  assert {
    condition = alltrue([
      for a in [aws_cloudwatch_metric_alarm.s2a, aws_cloudwatch_metric_alarm.s2b] :
      a.comparison_operator == "GreaterThanOrEqualToThreshold" && a.threshold == 1
      && try(tonumber(one(distinct(compact(concat([try(tostring(a.period), "")], [for q in a.metric_query : try(tostring(q.metric[0].period), "")]))))), 0) * a.evaluation_periods == 900
      && coalesce(a.datapoints_to_alarm, a.evaluation_periods) == 1
      && a.treat_missing_data == "notBreaching"
    ])
    error_message = "S2a and S2b: >= 1 in 15 min, missing data not breaching."
  }

  # S2c: loom_billing_stripe_webhook_errors_total >= 1 in 1 h, not breaching.
  assert {
    condition = (
      contains(compact(concat([aws_cloudwatch_metric_alarm.s2c.metric_name], [for q in aws_cloudwatch_metric_alarm.s2c.metric_query : try(q.metric[0].metric_name, "")])), "loom_billing_stripe_webhook_errors_total")
      && toset(compact(concat([aws_cloudwatch_metric_alarm.s2c.namespace], [for q in aws_cloudwatch_metric_alarm.s2c.metric_query : try(q.metric[0].namespace, "")]))) == toset(["Loom/Platform"])
      && aws_cloudwatch_metric_alarm.s2c.comparison_operator == "GreaterThanOrEqualToThreshold" && aws_cloudwatch_metric_alarm.s2c.threshold == 1
    )
    error_message = "S2c reads Loom/Platform loom_billing_stripe_webhook_errors_total, alarming at >= 1."
  }

  assert {
    condition = (
      try(tonumber(one(distinct(compact(concat([try(tostring(aws_cloudwatch_metric_alarm.s2c.period), "")], [for q in aws_cloudwatch_metric_alarm.s2c.metric_query : try(tostring(q.metric[0].period), "")]))))), 0) * aws_cloudwatch_metric_alarm.s2c.evaluation_periods == 3600
      && coalesce(aws_cloudwatch_metric_alarm.s2c.datapoints_to_alarm, aws_cloudwatch_metric_alarm.s2c.evaluation_periods) == 1
      && aws_cloudwatch_metric_alarm.s2c.treat_missing_data == "notBreaching"
    )
    error_message = "S2c: in 1 h, missing data not breaching."
  }

  # S4: loom_billing_stripe_handler_failures_total increased, not breaching.
  assert {
    condition = (
      contains(compact(concat([aws_cloudwatch_metric_alarm.s4.metric_name], [for q in aws_cloudwatch_metric_alarm.s4.metric_query : try(q.metric[0].metric_name, "")])), "loom_billing_stripe_handler_failures_total")
      && toset(compact(concat([aws_cloudwatch_metric_alarm.s4.namespace], [for q in aws_cloudwatch_metric_alarm.s4.metric_query : try(q.metric[0].namespace, "")]))) == toset(["Loom/Platform"])
      && ((aws_cloudwatch_metric_alarm.s4.comparison_operator == "GreaterThanThreshold" && aws_cloudwatch_metric_alarm.s4.threshold == 0) || (aws_cloudwatch_metric_alarm.s4.comparison_operator == "GreaterThanOrEqualToThreshold" && aws_cloudwatch_metric_alarm.s4.threshold == 1))
      && coalesce(aws_cloudwatch_metric_alarm.s4.datapoints_to_alarm, aws_cloudwatch_metric_alarm.s4.evaluation_periods) == 1
      && aws_cloudwatch_metric_alarm.s4.treat_missing_data == "notBreaching"
    )
    error_message = "S4 reads Loom/Platform loom_billing_stripe_handler_failures_total and alarms on any increase, missing data not breaching."
  }

  # S5: loom_billing_stripe_retry_last_success_seconds older than 5 min, breaching.
  assert {
    condition = (
      contains(compact(concat([aws_cloudwatch_metric_alarm.s5.metric_name], [for q in aws_cloudwatch_metric_alarm.s5.metric_query : try(q.metric[0].metric_name, "")])), "loom_billing_stripe_retry_last_success_seconds")
      && toset(compact(concat([aws_cloudwatch_metric_alarm.s5.namespace], [for q in aws_cloudwatch_metric_alarm.s5.metric_query : try(q.metric[0].namespace, "")]))) == toset(["Loom/Platform"])
      && aws_cloudwatch_metric_alarm.s5.treat_missing_data == "breaching"
    )
    error_message = "S5 reads Loom/Platform loom_billing_stripe_retry_last_success_seconds, missing data breaching."
  }
}

run "d1_r1_disk_and_rds" {
  command = plan

  # D1: disk_used_percent > 80% on / and /var/lib/loom, not breaching. The agent may only
  # publish to Loom/Platform (the instance role's namespace condition, §5.2).
  assert {
    condition = alltrue([
      for path, a in {
        "/"             = aws_cloudwatch_metric_alarm.d1_root
        "/var/lib/loom" = aws_cloudwatch_metric_alarm.d1_data
      } :
      contains(compact(concat([a.metric_name], [for q in a.metric_query : try(q.metric[0].metric_name, "")])), "disk_used_percent")
      && toset(compact(concat([a.namespace], [for q in a.metric_query : try(q.metric[0].namespace, "")]))) == toset(["Loom/Platform"])
      && anytrue([for d in concat([a.dimensions], [for q in a.metric_query : try(q.metric[0].dimensions, null)]) : lookup(coalesce(d, {}), "path", "") == path])
      && a.comparison_operator == "GreaterThanThreshold" && a.threshold == 80
      && a.treat_missing_data == "notBreaching"
    ])
    error_message = "D1: Loom/Platform disk_used_percent > 80 with path = / (d1_root) and path = /var/lib/loom (d1_data), missing data not breaching."
  }

  # R1: AWS/RDS FreeStorageSpace < 2 GB; CPUUtilization > 80% for 15 min; not breaching.
  assert {
    condition = (
      aws_cloudwatch_metric_alarm.r1_storage.namespace == "AWS/RDS"
      && aws_cloudwatch_metric_alarm.r1_storage.metric_name == "FreeStorageSpace"
      && lookup(coalesce(aws_cloudwatch_metric_alarm.r1_storage.dimensions, {}), "DBInstanceIdentifier", "") == aws_db_instance.platform.identifier
      && aws_cloudwatch_metric_alarm.r1_storage.comparison_operator == "LessThanThreshold"
      && contains([2000000000, 2147483648], aws_cloudwatch_metric_alarm.r1_storage.threshold)
      && aws_cloudwatch_metric_alarm.r1_storage.treat_missing_data == "notBreaching"
    )
    error_message = "R1 storage: AWS/RDS FreeStorageSpace for the instance < 2 GB (bytes), missing data not breaching."
  }

  assert {
    condition = (
      aws_cloudwatch_metric_alarm.r1_cpu.namespace == "AWS/RDS"
      && aws_cloudwatch_metric_alarm.r1_cpu.metric_name == "CPUUtilization"
      && lookup(coalesce(aws_cloudwatch_metric_alarm.r1_cpu.dimensions, {}), "DBInstanceIdentifier", "") == aws_db_instance.platform.identifier
      && aws_cloudwatch_metric_alarm.r1_cpu.comparison_operator == "GreaterThanThreshold"
      && aws_cloudwatch_metric_alarm.r1_cpu.threshold == 80
      && aws_cloudwatch_metric_alarm.r1_cpu.period * aws_cloudwatch_metric_alarm.r1_cpu.evaluation_periods == 900
      && coalesce(aws_cloudwatch_metric_alarm.r1_cpu.datapoints_to_alarm, aws_cloudwatch_metric_alarm.r1_cpu.evaluation_periods) == aws_cloudwatch_metric_alarm.r1_cpu.evaluation_periods
      && aws_cloudwatch_metric_alarm.r1_cpu.treat_missing_data == "notBreaching"
    )
    error_message = "R1 CPU: AWS/RDS CPUUtilization for the instance > 80 for 15 min (every datapoint), missing data not breaching."
  }
}
