# Mock values for the `aws.us_east_1` provider alias in the platform stack's tftests
# (docs/phase1-m3-spec.md §5.5, §8.5). Loaded with `mock_provider "aws" { alias =
# "us_east_1", source = "./tests/mocks/aws_us_east_1", override_during = plan }`.
#
# The alarm ARN differs from the default provider's mock
# (`...:alarm:mock-made-by-default-provider`), so a test can tell which provider created
# an alarm (H3 must be created through this alias).

mock_data "aws_region" {
  defaults = {
    id          = "us-east-1"
    region      = "us-east-1"
    name        = "us-east-1"
    endpoint    = "ec2.us-east-1.amazonaws.com"
    description = "US East (N. Virginia)"
  }
}

mock_data "aws_caller_identity" {
  defaults = {
    id         = "123456789012"
    account_id = "123456789012"
    arn        = "arn:aws:iam::123456789012:user/mock-terraform"
    user_id    = "AIDAMOCKMOCKMOCKMOCK"
  }
}

mock_data "aws_partition" {
  defaults = {
    id                 = "aws"
    partition          = "aws"
    dns_suffix         = "amazonaws.com"
    reverse_dns_prefix = "com.amazonaws"
  }
}

mock_resource "aws_route53_health_check" {
  defaults = {
    id                = "mock-hooks-health-check"
    arn               = "arn:aws:route53:::healthcheck/mock-hooks-health-check"
    failure_threshold = 3
    enable_sni        = false
  }
}

mock_resource "aws_cloudwatch_metric_alarm" {
  defaults = {
    arn = "arn:aws:cloudwatch:us-east-1:123456789012:alarm:mock-made-by-us-east-1-alias"
  }
}

mock_resource "aws_sns_topic" {
  defaults = {
    id  = "arn:aws:sns:us-east-1:123456789012:loom-platform-alerts"
    arn = "arn:aws:sns:us-east-1:123456789012:loom-platform-alerts"
  }
}
