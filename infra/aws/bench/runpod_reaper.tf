# Scheduled RunPod reaper: terminates Loom's RunPod pods (and deletes Loom-named network
# volumes) whose TTL has passed, independent of any laptop. Separate from the EC2 reaper
# so neither role holds the other's power: only this role can read the RunPod key, and it
# has no EC2 permissions. Code: bench/src/loom_bench/providers/runpod_reaper_lambda.py.

# The RunPod API key. Terraform creates the secret empty and never holds the value: put
# it with `aws secretsmanager put-secret-value` (docs/aws-setup.md, step 8).
resource "aws_secretsmanager_secret" "runpod_reaper_key" {
  name        = var.runpod_reaper_secret_name
  description = "Loom bench: RunPod API key for the scheduled RunPod reaper (plain string)"
}

# Only the reaper role may read the value, whatever other IAM policies in the account
# allow. Admins can still put a new value, or change or delete this policy.
data "aws_iam_policy_document" "runpod_reaper_key" {
  statement {
    sid       = "OnlyTheRunpodReaperReads"
    effect    = "Deny"
    actions   = ["secretsmanager:GetSecretValue"]
    resources = ["*"]

    principals {
      type        = "*"
      identifiers = ["*"]
    }

    condition {
      test     = "ArnNotEquals"
      variable = "aws:PrincipalArn"
      values   = [aws_iam_role.runpod_reaper.arn]
    }
  }
}

resource "aws_secretsmanager_secret_policy" "runpod_reaper_key" {
  secret_arn = aws_secretsmanager_secret.runpod_reaper_key.arn
  policy     = data.aws_iam_policy_document.runpod_reaper_key.json
}

data "archive_file" "runpod_reaper" {
  type        = "zip"
  source_file = "${path.module}/../../../bench/src/loom_bench/providers/runpod_reaper_lambda.py"
  output_path = "${path.module}/.build/runpod_reaper.zip"
}

resource "aws_cloudwatch_log_group" "runpod_reaper" {
  name              = "/aws/lambda/${var.name_prefix}-runpod-reaper"
  retention_in_days = 30
}

resource "aws_iam_role" "runpod_reaper" {
  name               = "${var.name_prefix}-runpod-reaper"
  assume_role_policy = data.aws_iam_policy_document.lambda_assume.json
}

data "aws_iam_policy_document" "runpod_reaper" {
  statement {
    sid       = "Logs"
    actions   = ["logs:CreateLogStream", "logs:PutLogEvents"]
    resources = ["${aws_cloudwatch_log_group.runpod_reaper.arn}:*"]
  }

  statement {
    sid       = "ReadRunpodKey"
    actions   = ["secretsmanager:GetSecretValue"]
    resources = [aws_secretsmanager_secret.runpod_reaper_key.arn]
  }
}

resource "aws_iam_role_policy" "runpod_reaper" {
  name   = "runpod-reaper"
  role   = aws_iam_role.runpod_reaper.id
  policy = data.aws_iam_policy_document.runpod_reaper.json
}

resource "aws_lambda_function" "runpod_reaper" {
  function_name    = "${var.name_prefix}-runpod-reaper"
  description      = "Terminate Loom RunPod pods and volumes past their TTL"
  role             = aws_iam_role.runpod_reaper.arn
  runtime          = "python3.12"
  handler          = "runpod_reaper_lambda.lambda_handler"
  filename         = data.archive_file.runpod_reaper.output_path
  source_code_hash = data.archive_file.runpod_reaper.output_base64sha256
  timeout          = 120
  memory_size      = 128

  environment {
    variables = {
      LOOM_RUNPOD_KEY_SECRET_ID      = aws_secretsmanager_secret.runpod_reaper_key.arn
      LOOM_RUNPOD_REAPER_PREFIX      = var.name_prefix
      LOOM_RUNPOD_REAPER_DRY_RUN     = tostring(var.runpod_reaper_dry_run)
      LOOM_RUNPOD_REAPER_MAX_PER_RUN = tostring(var.runpod_reaper_max_per_run)
    }
  }

  depends_on = [aws_cloudwatch_log_group.runpod_reaper, aws_iam_role_policy.runpod_reaper]
}

# A failed run is not retried: the next scheduled run is the retry.
resource "aws_lambda_function_event_invoke_config" "runpod_reaper" {
  function_name          = aws_lambda_function.runpod_reaper.function_name
  maximum_retry_attempts = 0
}

resource "aws_cloudwatch_event_rule" "runpod_reaper" {
  name                = "${var.name_prefix}-runpod-reaper"
  description         = "Run the Loom RunPod TTL reaper"
  schedule_expression = "rate(15 minutes)"
}

resource "aws_cloudwatch_event_target" "runpod_reaper" {
  rule = aws_cloudwatch_event_rule.runpod_reaper.name
  arn  = aws_lambda_function.runpod_reaper.arn
}

resource "aws_lambda_permission" "runpod_reaper" {
  statement_id  = "AllowEventBridgeSchedule"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.runpod_reaper.function_name
  principal     = "events.amazonaws.com"
  source_arn    = aws_cloudwatch_event_rule.runpod_reaper.arn
}
