# The platform's alarms (docs/phase1-m3-spec.md §5.3); H3 is in healthcheck.tf. Every alarm
# notifies loom-platform-alerts on ALARM and OK. Metrics from the services reach the
# Loom/Platform namespace through the CloudWatch agent's Prometheus scraping
# (deploy/cloudwatch-agent.json), only the series named here.
#
# Counter math (A9): the agent sends a Prometheus counter as the delta since the previous
# scrape (it computes the delta itself), so a counter alarm reads Sum over its window, and
# every counter is initialised to 0 at process start so the first increment is a delta.
# Gauges read Maximum (alarm when high) or Minimum (alarm when low, or "every datapoint").
# The dimensions are the label values the agent's allowlist keeps.

resource "aws_sns_topic" "alerts" {
  name = "loom-platform-alerts"
}

resource "aws_sns_topic_subscription" "ops_email" {
  topic_arn = aws_sns_topic.alerts.arn
  protocol  = "email"
  endpoint  = var.ops_email
}

# ---- P1: the serving reaper (actions follow serving_reaper_enabled, off in M3) ----------

resource "aws_cloudwatch_metric_alarm" "p1a" {
  alarm_name          = "loom-platform-p1a-serving-reaper-errors"
  alarm_description   = "P1a: the serving reaper raised an error in the last hour"
  namespace           = "AWS/Lambda"
  metric_name         = "Errors"
  dimensions          = { FunctionName = aws_lambda_function.serving_reaper.function_name }
  statistic           = "Sum"
  comparison_operator = "GreaterThanOrEqualToThreshold"
  threshold           = 1
  period              = 3600
  evaluation_periods  = 1
  treat_missing_data  = "notBreaching"
  actions_enabled     = var.serving_reaper_enabled
  alarm_actions       = local.alarm_topic_arns
  ok_actions          = local.alarm_topic_arns
}

resource "aws_cloudwatch_metric_alarm" "p1b" {
  alarm_name          = "loom-platform-p1b-serving-reaper-not-running"
  alarm_description   = "P1b: the serving reaper did not run in two consecutive 15-minute periods"
  namespace           = "AWS/Lambda"
  metric_name         = "Invocations"
  dimensions          = { FunctionName = aws_lambda_function.serving_reaper.function_name }
  statistic           = "Sum"
  comparison_operator = "LessThanThreshold"
  threshold           = 1
  period              = 900
  evaluation_periods  = 2
  datapoints_to_alarm = 2
  treat_missing_data  = "breaching"
  actions_enabled     = var.serving_reaper_enabled
  alarm_actions       = local.alarm_topic_arns
  ok_actions          = local.alarm_topic_arns
}

# ---- P2, P3: the pool manager ------------------------------------------------------------

resource "aws_cloudwatch_metric_alarm" "p2a" {
  alarm_name          = "loom-platform-p2a-spend-cap-refusal"
  alarm_description   = "P2a: the pool manager refused a create at the spend cap"
  namespace           = "Loom/Platform"
  metric_name         = "loom_poolmgr_creates_total"
  dimensions          = { result = "spend_cap" }
  statistic           = "Sum"
  comparison_operator = "GreaterThanThreshold"
  threshold           = 0
  period              = 300
  evaluation_periods  = 1
  treat_missing_data  = "notBreaching"
  alarm_actions       = local.alarm_topic_arns
  ok_actions          = local.alarm_topic_arns
}

resource "aws_cloudwatch_metric_alarm" "p2b" {
  alarm_name          = "loom-platform-p2b-model-held-at-cap"
  alarm_description   = "P2b: a model has been held at the spend cap for 5 minutes"
  namespace           = "Loom/Platform"
  metric_name         = "loom_poolmgr_pool_reason"
  dimensions          = { reason = "spend_cap" }
  statistic           = "Maximum"
  comparison_operator = "GreaterThanOrEqualToThreshold"
  threshold           = 1
  period              = 60
  evaluation_periods  = 5
  datapoints_to_alarm = 5
  treat_missing_data  = "notBreaching"
  alarm_actions       = local.alarm_topic_arns
  ok_actions          = local.alarm_topic_arns
}

resource "aws_cloudwatch_metric_alarm" "p3" {
  alarm_name          = "loom-platform-p3-pod-ca-near-expiry"
  alarm_description   = "P3: the pod CA expires in under 30 days (or the manager publishes nothing)"
  namespace           = "Loom/Platform"
  metric_name         = "loom_poolmgr_ca_expiry_seconds"
  statistic           = "Minimum"
  comparison_operator = "LessThanThreshold"
  threshold           = 2592000
  period              = 3600
  evaluation_periods  = 1
  treat_missing_data  = "breaching"
  alarm_actions       = local.alarm_topic_arns
  ok_actions          = local.alarm_topic_arns
}

# ---- H1, H2: heartbeat and instance health ---------------------------------------------

resource "aws_cloudwatch_metric_alarm" "h1_gateway" {
  alarm_name          = "loom-platform-h1-gateway-down"
  alarm_description   = "H1: the agent's scrape of the gateway failed for 5 minutes (box or agent down)"
  namespace           = "Loom/Platform"
  metric_name         = "up"
  dimensions          = { job = "gateway" }
  statistic           = "Minimum"
  comparison_operator = "LessThanThreshold"
  threshold           = 1
  period              = 60
  evaluation_periods  = 5
  datapoints_to_alarm = 5
  treat_missing_data  = "breaching"
  alarm_actions       = local.alarm_topic_arns
  ok_actions          = local.alarm_topic_arns
}

resource "aws_cloudwatch_metric_alarm" "h1_billing" {
  alarm_name          = "loom-platform-h1-billing-down"
  alarm_description   = "H1: the agent's scrape of billing failed for 5 minutes (box or agent down)"
  namespace           = "Loom/Platform"
  metric_name         = "up"
  dimensions          = { job = "billing" }
  statistic           = "Minimum"
  comparison_operator = "LessThanThreshold"
  threshold           = 1
  period              = 60
  evaluation_periods  = 5
  datapoints_to_alarm = 5
  treat_missing_data  = "breaching"
  alarm_actions       = local.alarm_topic_arns
  ok_actions          = local.alarm_topic_arns
}

resource "aws_cloudwatch_metric_alarm" "h1_poolmgr" {
  alarm_name          = "loom-platform-h1-poolmgr-down"
  alarm_description   = "H1: the agent's scrape of the pool manager failed for 5 minutes (box or agent down)"
  namespace           = "Loom/Platform"
  metric_name         = "up"
  dimensions          = { job = "poolmgr" }
  statistic           = "Minimum"
  comparison_operator = "LessThanThreshold"
  threshold           = 1
  period              = 60
  evaluation_periods  = 5
  datapoints_to_alarm = 5
  treat_missing_data  = "breaching"
  alarm_actions       = local.alarm_topic_arns
  ok_actions          = local.alarm_topic_arns
}

# A failed system check means the host is impaired: recover the instance (same id, same
# volumes, same Elastic IP). /run/loom is empty afterwards (§6.3 runbook line).
resource "aws_cloudwatch_metric_alarm" "h2_system" {
  alarm_name          = "loom-platform-h2-system-status-check"
  alarm_description   = "H2: the instance's system status check failed; EC2 recovers it"
  namespace           = "AWS/EC2"
  metric_name         = "StatusCheckFailed_System"
  dimensions          = { InstanceId = aws_instance.platform.id }
  statistic           = "Maximum"
  comparison_operator = "GreaterThanOrEqualToThreshold"
  threshold           = 1
  period              = 60
  evaluation_periods  = 2
  treat_missing_data  = "breaching"
  alarm_actions       = concat(local.alarm_topic_arns, ["arn:${local.partition}:automate:${local.region}:ec2:recover"])
  ok_actions          = local.alarm_topic_arns
}

resource "aws_cloudwatch_metric_alarm" "h2_instance" {
  alarm_name          = "loom-platform-h2-instance-status-check"
  alarm_description   = "H2: the instance's own status check failed (notify only)"
  namespace           = "AWS/EC2"
  metric_name         = "StatusCheckFailed_Instance"
  dimensions          = { InstanceId = aws_instance.platform.id }
  statistic           = "Maximum"
  comparison_operator = "GreaterThanOrEqualToThreshold"
  threshold           = 1
  period              = 60
  evaluation_periods  = 5
  treat_missing_data  = "breaching"
  alarm_actions       = local.alarm_topic_arns
  ok_actions          = local.alarm_topic_arns
}

# ---- G: the gateway ----------------------------------------------------------------------

resource "aws_cloudwatch_metric_alarm" "g1" {
  alarm_name          = "loom-platform-g1-settle-spool"
  alarm_description   = "G1: the settle spool has held records for 5 minutes"
  namespace           = "Loom/Platform"
  metric_name         = "loom_gateway_settle_spool_depth"
  statistic           = "Maximum"
  comparison_operator = "GreaterThanThreshold"
  threshold           = 0
  period              = 60
  evaluation_periods  = 5
  datapoints_to_alarm = 5
  treat_missing_data  = "notBreaching"
  alarm_actions       = local.alarm_topic_arns
  ok_actions          = local.alarm_topic_arns
}

resource "aws_cloudwatch_metric_alarm" "g2" {
  alarm_name          = "loom-platform-g2-lost-usage"
  alarm_description   = "G2: a settle failed (usage may be lost; contracts section 6)"
  namespace           = "Loom/Platform"
  metric_name         = "loom_gateway_settle_failures_total"
  statistic           = "Sum"
  comparison_operator = "GreaterThanThreshold"
  threshold           = 0
  period              = 60
  evaluation_periods  = 1
  treat_missing_data  = "notBreaching"
  alarm_actions       = local.alarm_topic_arns
  ok_actions          = local.alarm_topic_arns
}

resource "aws_cloudwatch_metric_alarm" "g3" {
  alarm_name          = "loom-platform-g3-gateway-client-cert"
  alarm_description   = "G3: the gateway's client certificate expires in under 2 days (rotation is broken)"
  namespace           = "Loom/Platform"
  metric_name         = "loom_gateway_client_cert_expiry_seconds"
  statistic           = "Minimum"
  comparison_operator = "LessThanThreshold"
  threshold           = 172800
  period              = 300
  evaluation_periods  = 1
  treat_missing_data  = "breaching"
  alarm_actions       = local.alarm_topic_arns
  ok_actions          = local.alarm_topic_arns
}

# ---- S: Stripe webhooks (§8) -------------------------------------------------------------

resource "aws_cloudwatch_metric_alarm" "s1a" {
  alarm_name          = "loom-platform-s1a-stripe-event-stuck"
  alarm_description   = "S1a: a Stripe event stayed unapplied for 15 minutes"
  namespace           = "Loom/Platform"
  metric_name         = "loom_billing_stripe_unapplied_stale_total"
  statistic           = "Sum"
  comparison_operator = "GreaterThanThreshold"
  threshold           = 0
  period              = 300
  evaluation_periods  = 1
  treat_missing_data  = "notBreaching"
  alarm_actions       = local.alarm_topic_arns
  ok_actions          = local.alarm_topic_arns
}

resource "aws_cloudwatch_metric_alarm" "s1b" {
  alarm_name          = "loom-platform-s1b-stuck-events-outstanding"
  alarm_description   = "S1b: stuck Stripe events have been outstanding for 24 hours"
  namespace           = "Loom/Platform"
  metric_name         = "loom_billing_stripe_unapplied_stale_events"
  statistic           = "Maximum"
  comparison_operator = "GreaterThanThreshold"
  threshold           = 0
  period              = 3600
  evaluation_periods  = 24
  datapoints_to_alarm = 24
  treat_missing_data  = "notBreaching"
  alarm_actions       = local.alarm_topic_arns
  ok_actions          = local.alarm_topic_arns
}

resource "aws_cloudwatch_metric_alarm" "s2a" {
  alarm_name          = "loom-platform-s2a-webhook-secret-mismatch"
  alarm_description   = "S2a: a webhook arrived with a well-formed header and no matching signature (wrong secret on the box)"
  namespace           = "Loom/Platform"
  metric_name         = "loom_billing_stripe_webhooks_total"
  dimensions          = { result = "signature_mismatch" }
  statistic           = "Sum"
  comparison_operator = "GreaterThanOrEqualToThreshold"
  threshold           = 1
  period              = 900
  evaluation_periods  = 1
  treat_missing_data  = "notBreaching"
  alarm_actions       = local.alarm_topic_arns
  ok_actions          = local.alarm_topic_arns
}

resource "aws_cloudwatch_metric_alarm" "s2b" {
  alarm_name          = "loom-platform-s2b-webhook-clock-skew"
  alarm_description   = "S2b: a webhook's signature was valid but its timestamp was outside the tolerance (clock skew)"
  namespace           = "Loom/Platform"
  metric_name         = "loom_billing_stripe_webhooks_total"
  dimensions          = { result = "stale_timestamp" }
  statistic           = "Sum"
  comparison_operator = "GreaterThanOrEqualToThreshold"
  threshold           = 1
  period              = 900
  evaluation_periods  = 1
  treat_missing_data  = "notBreaching"
  alarm_actions       = local.alarm_topic_arns
  ok_actions          = local.alarm_topic_arns
}

resource "aws_cloudwatch_metric_alarm" "s2c" {
  alarm_name          = "loom-platform-s2c-webhook-5xx"
  alarm_description   = "S2c: the webhook answered 5xx"
  namespace           = "Loom/Platform"
  metric_name         = "loom_billing_stripe_webhook_errors_total"
  statistic           = "Sum"
  comparison_operator = "GreaterThanOrEqualToThreshold"
  threshold           = 1
  period              = 3600
  evaluation_periods  = 1
  treat_missing_data  = "notBreaching"
  alarm_actions       = local.alarm_topic_arns
  ok_actions          = local.alarm_topic_arns
}

resource "aws_cloudwatch_metric_alarm" "s4" {
  alarm_name          = "loom-platform-s4-handler-failure"
  alarm_description   = "S4: a Stripe event handler raised"
  namespace           = "Loom/Platform"
  metric_name         = "loom_billing_stripe_handler_failures_total"
  statistic           = "Sum"
  comparison_operator = "GreaterThanThreshold"
  threshold           = 0
  period              = 300
  evaluation_periods  = 1
  treat_missing_data  = "notBreaching"
  alarm_actions       = local.alarm_topic_arns
  ok_actions          = local.alarm_topic_arns
}

# S5: the retry job's last success (a Unix time) is older than 5 minutes: EPOCH(m) - m.
resource "aws_cloudwatch_metric_alarm" "s5" {
  alarm_name          = "loom-platform-s5-retry-job-stopped"
  alarm_description   = "S5: the Stripe retry job has not completed a run for 5 minutes"
  comparison_operator = "GreaterThanThreshold"
  threshold           = 300
  evaluation_periods  = 1
  datapoints_to_alarm = 1
  treat_missing_data  = "breaching"
  alarm_actions       = local.alarm_topic_arns
  ok_actions          = local.alarm_topic_arns

  metric_query {
    id          = "m"
    return_data = false

    metric {
      namespace   = "Loom/Platform"
      metric_name = "loom_billing_stripe_retry_last_success_seconds"
      period      = 60
      stat        = "Maximum"
    }
  }

  metric_query {
    id          = "age"
    expression  = "EPOCH(m) - m"
    label       = "Seconds since the last completed retry run"
    return_data = true
  }
}

# ---- D1, R1: disk and RDS ----------------------------------------------------------------

resource "aws_cloudwatch_metric_alarm" "d1_root" {
  alarm_name          = "loom-platform-d1-root-disk"
  alarm_description   = "D1: the root volume is over 80% full"
  namespace           = "Loom/Platform"
  metric_name         = "disk_used_percent"
  dimensions          = { path = "/" }
  statistic           = "Maximum"
  comparison_operator = "GreaterThanThreshold"
  threshold           = 80
  period              = 300
  evaluation_periods  = 1
  treat_missing_data  = "notBreaching"
  alarm_actions       = local.alarm_topic_arns
  ok_actions          = local.alarm_topic_arns
}

resource "aws_cloudwatch_metric_alarm" "d1_data" {
  alarm_name          = "loom-platform-d1-data-disk"
  alarm_description   = "D1: the data volume (/var/lib/loom) is over 80% full"
  namespace           = "Loom/Platform"
  metric_name         = "disk_used_percent"
  dimensions          = { path = "/var/lib/loom" }
  statistic           = "Maximum"
  comparison_operator = "GreaterThanThreshold"
  threshold           = 80
  period              = 300
  evaluation_periods  = 1
  treat_missing_data  = "notBreaching"
  alarm_actions       = local.alarm_topic_arns
  ok_actions          = local.alarm_topic_arns
}

resource "aws_cloudwatch_metric_alarm" "r1_storage" {
  alarm_name          = "loom-platform-r1-rds-free-storage"
  alarm_description   = "R1: the database has under 2 GB of free storage"
  namespace           = "AWS/RDS"
  metric_name         = "FreeStorageSpace"
  dimensions          = { DBInstanceIdentifier = aws_db_instance.platform.identifier }
  statistic           = "Minimum"
  comparison_operator = "LessThanThreshold"
  threshold           = 2147483648
  period              = 300
  evaluation_periods  = 1
  treat_missing_data  = "notBreaching"
  alarm_actions       = local.alarm_topic_arns
  ok_actions          = local.alarm_topic_arns
}

resource "aws_cloudwatch_metric_alarm" "r1_cpu" {
  alarm_name          = "loom-platform-r1-rds-cpu"
  alarm_description   = "R1: the database CPU has been over 80% for 15 minutes"
  namespace           = "AWS/RDS"
  metric_name         = "CPUUtilization"
  dimensions          = { DBInstanceIdentifier = aws_db_instance.platform.identifier }
  statistic           = "Maximum"
  comparison_operator = "GreaterThanThreshold"
  threshold           = 80
  period              = 300
  evaluation_periods  = 3
  datapoints_to_alarm = 3
  treat_missing_data  = "notBreaching"
  alarm_actions       = local.alarm_topic_arns
  ok_actions          = local.alarm_topic_arns
}
