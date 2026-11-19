# A37, Terraform half (docs/phase1-m3-spec.md §13, §8.5, §5.2 healthcheck, §5.3 H3):
# healthcheck.tf has an aws_route53_health_check of type HTTPS_STR_MATCH for
# hooks.<domain>, port 443, path /stripe/health, search string stripe_webhook, SNI on,
# interval 30 s, failure threshold 3, with and without route53_zone_id; alarm H3 reads
# AWS/Route53 HealthCheckStatus with dimension HealthCheckId = that check's id (Minimum
# < 1, 5 x 60 s, missing data breaching), is created through the aws.us_east_1 provider
# and notifies loom-platform-alerts on ALARM and OK; the plan fails when the stack's region
# (data.aws_region.current.region) is not us-east-1.
#
# Addresses this file expects (the spec names the data source only):
#   aws_route53_health_check.hooks    the check; it carries the region precondition
#   aws_cloudwatch_metric_alarm.h3    alarm H3
#   aws_sns_topic.alerts              loom-platform-alerts
#   data.aws_region.current           the stack's region (§8.5)
# Which provider made H3 is told by its mocked ARN: alarms made through aws.us_east_1 end
# in ":alarm:mock-made-by-us-east-1-alias" (tests/mocks/aws_us_east_1).
#
# Note for the implementer: put the region precondition (lifecycle { precondition }) on
# aws_route53_health_check.hooks and nowhere else. `plan_fails_outside_us_east_1` expects
# exactly that one failure; a second region guard (a check block, a precondition on
# another resource or output, a variable validation) is an unexpected failure there.

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

run "health_check_without_a_zone" {
  command = plan

  assert {
    condition     = aws_route53_health_check.hooks.type == "HTTPS_STR_MATCH"
    error_message = "The health check type is HTTPS_STR_MATCH."
  }

  assert {
    condition     = trimsuffix(aws_route53_health_check.hooks.fqdn, ".") == "hooks.example.test" && aws_route53_health_check.hooks.ip_address == null
    error_message = "The check resolves hooks.<domain> through public DNS (fqdn, no ip_address)."
  }

  assert {
    condition = (
      aws_route53_health_check.hooks.port == 443
      && aws_route53_health_check.hooks.resource_path == "/stripe/health"
      && aws_route53_health_check.hooks.search_string == "stripe_webhook"
    )
    error_message = "Port 443, path /stripe/health, search string stripe_webhook."
  }

  assert {
    condition     = aws_route53_health_check.hooks.enable_sni == true
    error_message = "SNI on (enable_sni = true; Route 53 does not default it on for HTTPS_STR_MATCH)."
  }

  assert {
    condition     = aws_route53_health_check.hooks.request_interval == 30 && aws_route53_health_check.hooks.failure_threshold == 3
    error_message = "Every 30 s, failure threshold 3."
  }

  assert {
    condition = (
      aws_cloudwatch_metric_alarm.h3.namespace == "AWS/Route53"
      && aws_cloudwatch_metric_alarm.h3.metric_name == "HealthCheckStatus"
      && jsonencode(aws_cloudwatch_metric_alarm.h3.dimensions) == jsonencode({ HealthCheckId = aws_route53_health_check.hooks.id })
    )
    error_message = "H3 reads AWS/Route53 HealthCheckStatus with dimension HealthCheckId = aws_route53_health_check.hooks.id (and no other dimension)."
  }

  assert {
    condition = (
      aws_cloudwatch_metric_alarm.h3.statistic == "Minimum"
      && aws_cloudwatch_metric_alarm.h3.comparison_operator == "LessThanThreshold"
      && aws_cloudwatch_metric_alarm.h3.threshold == 1
    )
    error_message = "H3: Minimum < 1."
  }

  assert {
    condition = (
      aws_cloudwatch_metric_alarm.h3.period == 60
      && aws_cloudwatch_metric_alarm.h3.evaluation_periods == 5
      && coalesce(aws_cloudwatch_metric_alarm.h3.datapoints_to_alarm, 5) == 5
      && aws_cloudwatch_metric_alarm.h3.treat_missing_data == "breaching"
    )
    error_message = "H3: 5 consecutive 1-min periods, missing data breaching."
  }

  assert {
    condition     = endswith(aws_cloudwatch_metric_alarm.h3.arn, ":alarm:mock-made-by-us-east-1-alias")
    error_message = "H3 is created through the aws.us_east_1 provider (Route 53 publishes health-check metrics only in us-east-1)."
  }

  assert {
    condition = (
      aws_sns_topic.alerts.name == "loom-platform-alerts"
      && contains(coalesce(aws_cloudwatch_metric_alarm.h3.alarm_actions, []), aws_sns_topic.alerts.arn)
      && contains(coalesce(aws_cloudwatch_metric_alarm.h3.ok_actions, []), aws_sns_topic.alerts.arn)
    )
    error_message = "H3 notifies loom-platform-alerts on ALARM and on OK."
  }
}

run "health_check_with_a_zone" {
  command = plan

  variables {
    route53_zone_id = "Z0MOCKZONE0000000001"
  }

  assert {
    condition = (
      aws_route53_health_check.hooks.type == "HTTPS_STR_MATCH"
      && trimsuffix(aws_route53_health_check.hooks.fqdn, ".") == "hooks.example.test"
      && aws_route53_health_check.hooks.port == 443
      && aws_route53_health_check.hooks.resource_path == "/stripe/health"
      && aws_route53_health_check.hooks.search_string == "stripe_webhook"
      && aws_route53_health_check.hooks.enable_sni == true
      && aws_route53_health_check.hooks.request_interval == 30
      && aws_route53_health_check.hooks.failure_threshold == 3
    )
    error_message = "The health check is the same with route53_zone_id set."
  }

  assert {
    condition = (
      jsonencode(aws_cloudwatch_metric_alarm.h3.dimensions) == jsonencode({ HealthCheckId = aws_route53_health_check.hooks.id })
      && endswith(aws_cloudwatch_metric_alarm.h3.arn, ":alarm:mock-made-by-us-east-1-alias")
    )
    error_message = "H3 is the same with route53_zone_id set."
  }
}

run "health_check_follows_the_domain" {
  command = plan

  variables {
    domain = "other-placeholder.test"
  }

  assert {
    condition     = trimsuffix(aws_route53_health_check.hooks.fqdn, ".") == "hooks.other-placeholder.test"
    error_message = "The checked name is hooks.<var.domain>."
  }
}

run "plan_fails_outside_us_east_1" {
  command = plan

  override_data {
    target = data.aws_region.current
    values = {
      id     = "eu-west-1"
      region = "eu-west-1"
      name   = "eu-west-1"
    }
  }

  expect_failures = [
    aws_route53_health_check.hooks,
  ]
}

run "source_health_check_is_in_healthcheck_tf" {
  command = plan

  module {
    source = "./tests/fixtures/source_scan"
  }

  assert {
    condition     = output.healthcheck_tf_declares_health_check
    error_message = "healthcheck.tf declares the aws_route53_health_check (§5.1, §8.5)."
  }
}
