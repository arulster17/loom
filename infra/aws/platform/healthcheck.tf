# H3: is the webhook endpoint reachable from the internet (docs/phase1-m3-spec.md §5.3, §8.5)?
# Route 53's checkers request https://hooks.<domain>/stripe/health every 30 s and look for
# `stripe_webhook` in the answer. Created whether or not route53_zone_id is set: the check
# resolves the public name. Route 53 publishes its metrics only in us-east-1, so the alarm
# is created through the alias provider and its action (the alerts topic) must be in the
# same region: the precondition below fails the plan anywhere else.

resource "aws_route53_health_check" "hooks" {
  type              = "HTTPS_STR_MATCH"
  fqdn              = "hooks.${var.domain}"
  port              = 443
  resource_path     = "/stripe/health"
  search_string     = "stripe_webhook"
  enable_sni        = true
  request_interval  = 30
  failure_threshold = 3

  tags = { Name = "loom-platform-hooks" }

  lifecycle {
    precondition {
      condition     = data.aws_region.current.region == "us-east-1"
      error_message = "The platform stack must be in us-east-1: Route 53 health-check metrics (alarm H3) exist only there and the alarm's action must be in the same region."
    }
  }
}

resource "aws_cloudwatch_metric_alarm" "h3" {
  provider = aws.us_east_1

  alarm_name          = "loom-platform-h3-webhook-unreachable"
  alarm_description   = "H3: the hooks. health check has failed for 5 minutes (docs/phase1-m3-spec.md §8.5)"
  namespace           = "AWS/Route53"
  metric_name         = "HealthCheckStatus"
  dimensions          = { HealthCheckId = aws_route53_health_check.hooks.id }
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
