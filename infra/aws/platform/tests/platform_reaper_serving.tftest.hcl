# A7 (docs/phase1-m3-spec.md §13, §5.2 reaper_serving): the serving RunPod reaper
# (`loom-serving-runpod-reaper`): serving rule on, lease bucket set, dry run on, key from
# serving-runpod-key; its 15-minute schedule disabled while serving_reaper_enabled = false
# (the default) and enabled when true.
#
# Addresses this file expects (the spec names none):
#   aws_lambda_function.serving_reaper                       the Lambda
#   aws_cloudwatch_event_rule.serving_reaper                 its EventBridge schedule
#   aws_s3_bucket.leases                                     the lease bucket (§5.2 s3)
#   aws_secretsmanager_secret.platform["serving-runpod-key"] (see platform_secrets)
# Environment variable names are the ones bench/src/loom_bench/providers/runpod_reaper_lambda.py reads.

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

override_resource {
  target          = aws_secretsmanager_secret.platform["serving-runpod-key"]
  override_during = plan
  values = {
    id  = "arn:aws:secretsmanager:us-east-1:123456789012:secret:loom/platform/serving-runpod-key-AbCdEf"
    arn = "arn:aws:secretsmanager:us-east-1:123456789012:secret:loom/platform/serving-runpod-key-AbCdEf"
  }
}

run "serving_reaper_disabled_by_default" {
  command = plan

  assert {
    condition     = aws_lambda_function.serving_reaper.function_name == "loom-serving-runpod-reaper"
    error_message = "The serving reaper Lambda is loom-serving-runpod-reaper."
  }

  assert {
    condition     = aws_lambda_function.serving_reaper.handler == "runpod_reaper_lambda.lambda_handler"
    error_message = "It runs bench's runpod_reaper_lambda.py (a second instance, §5.2)."
  }

  assert {
    condition     = try(aws_lambda_function.serving_reaper.environment[0].variables["LOOM_RUNPOD_REAPER_SERVING"], "") == "true"
    error_message = "LOOM_RUNPOD_REAPER_SERVING=true (the serving rule on)."
  }

  assert {
    condition     = try(aws_lambda_function.serving_reaper.environment[0].variables["LOOM_RUNPOD_REAPER_LEASE_BUCKET"], "") == aws_s3_bucket.leases.bucket
    error_message = "LOOM_RUNPOD_REAPER_LEASE_BUCKET is the lease bucket's name."
  }

  assert {
    condition     = try(aws_lambda_function.serving_reaper.environment[0].variables["LOOM_RUNPOD_REAPER_DRY_RUN"], "true") != "false"
    error_message = "Dry run stays on (LOOM_RUNPOD_REAPER_DRY_RUN must not be \"false\")."
  }

  assert {
    condition = contains([
      aws_secretsmanager_secret.platform["serving-runpod-key"].arn,
      aws_secretsmanager_secret.platform["serving-runpod-key"].name,
    ], try(aws_lambda_function.serving_reaper.environment[0].variables["LOOM_RUNPOD_KEY_SECRET_ID"], ""))
    error_message = "LOOM_RUNPOD_KEY_SECRET_ID names the serving-runpod-key secret (its ARN or name)."
  }

  assert {
    condition     = aws_cloudwatch_event_rule.serving_reaper.schedule_expression == "rate(15 minutes)"
    error_message = "The schedule runs every 15 minutes."
  }

  assert {
    condition = (
      aws_cloudwatch_event_rule.serving_reaper.state == "DISABLED"
      || (aws_cloudwatch_event_rule.serving_reaper.state == null && aws_cloudwatch_event_rule.serving_reaper.is_enabled == false)
    )
    error_message = "With serving_reaper_enabled = false (the default) the schedule is disabled."
  }
}

run "serving_reaper_enabled_when_true" {
  command = plan

  variables {
    serving_reaper_enabled = true
  }

  assert {
    condition = (
      aws_cloudwatch_event_rule.serving_reaper.state == "ENABLED"
      || (aws_cloudwatch_event_rule.serving_reaper.state == null && aws_cloudwatch_event_rule.serving_reaper.is_enabled != false)
    )
    error_message = "With serving_reaper_enabled = true the schedule is enabled."
  }

  assert {
    condition = (
      try(aws_lambda_function.serving_reaper.environment[0].variables["LOOM_RUNPOD_REAPER_SERVING"], "") == "true"
      && try(aws_lambda_function.serving_reaper.environment[0].variables["LOOM_RUNPOD_REAPER_DRY_RUN"], "true") != "false"
    )
    error_message = "Enabling the schedule changes neither the serving rule nor dry run."
  }
}
