# A second instance of bench's runpod_reaper_lambda.py for the serving RunPod account
# (docs/phase1-m3-spec.md §5.2 reaper_serving). The bench reaper holds the bench account's
# key and cannot see serving pods, so the serving rule needs its own function, key and
# role. Dry run stays on and the schedule is disabled until M4 (var.serving_reaper_enabled).
# M4 note: LOOM_RUNPOD_REAPER_PREFIX is left at the code's default; M4 sets the serving
# pods' prefix when it turns the reaper on.

data "archive_file" "serving_reaper" {
  type        = "zip"
  source_file = "${path.module}/../../../bench/src/loom_bench/providers/runpod_reaper_lambda.py"
  output_path = "${path.module}/.build/serving_reaper.zip"
}

resource "aws_cloudwatch_log_group" "serving_reaper" {
  name              = "/aws/lambda/loom-serving-runpod-reaper"
  retention_in_days = 30
}

resource "aws_lambda_function" "serving_reaper" {
  function_name    = "loom-serving-runpod-reaper"
  description      = "Terminate serving RunPod pods past their lease (dry run until M4)"
  role             = aws_iam_role.serving_reaper.arn
  runtime          = "python3.12"
  handler          = "runpod_reaper_lambda.lambda_handler"
  filename         = data.archive_file.serving_reaper.output_path
  source_code_hash = data.archive_file.serving_reaper.output_base64sha256
  timeout          = 120
  memory_size      = 128

  environment {
    variables = {
      LOOM_RUNPOD_KEY_SECRET_ID       = aws_secretsmanager_secret.platform["serving-runpod-key"].arn
      LOOM_RUNPOD_REAPER_SERVING      = "true"
      LOOM_RUNPOD_REAPER_LEASE_BUCKET = aws_s3_bucket.leases.bucket
      LOOM_RUNPOD_REAPER_DRY_RUN      = "true"
    }
  }

  depends_on = [aws_cloudwatch_log_group.serving_reaper, aws_iam_role_policy.serving_reaper]
}

# A failed run is not retried: the next scheduled run is the retry.
resource "aws_lambda_function_event_invoke_config" "serving_reaper" {
  function_name          = aws_lambda_function.serving_reaper.function_name
  maximum_retry_attempts = 0
}

resource "aws_cloudwatch_event_rule" "serving_reaper" {
  name                = "loom-serving-runpod-reaper"
  description         = "Run the serving RunPod reaper"
  schedule_expression = "rate(15 minutes)"
  state               = var.serving_reaper_enabled ? "ENABLED" : "DISABLED"
}

resource "aws_cloudwatch_event_target" "serving_reaper" {
  rule = aws_cloudwatch_event_rule.serving_reaper.name
  arn  = aws_lambda_function.serving_reaper.arn
}

resource "aws_lambda_permission" "serving_reaper" {
  statement_id  = "AllowEventBridgeSchedule"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.serving_reaper.function_name
  principal     = "events.amazonaws.com"
  source_arn    = aws_cloudwatch_event_rule.serving_reaper.arn
}
