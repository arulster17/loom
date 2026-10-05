# Scheduled TTL reaper: terminates loom:managed instances and deletes unattached
# loom:managed volumes whose loom:ttl has passed, independent of any laptop.

data "archive_file" "reaper" {
  type        = "zip"
  source_file = "${path.module}/../../../bench/src/loom_bench/providers/aws_reaper.py"
  output_path = "${path.module}/.build/reaper.zip"
}

resource "aws_cloudwatch_log_group" "reaper" {
  name              = "/aws/lambda/${var.name_prefix}-reaper"
  retention_in_days = 30
}

data "aws_iam_policy_document" "lambda_assume" {
  statement {
    actions = ["sts:AssumeRole"]

    principals {
      type        = "Service"
      identifiers = ["lambda.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "reaper" {
  name               = "${var.name_prefix}-reaper"
  assume_role_policy = data.aws_iam_policy_document.lambda_assume.json
}

data "aws_iam_policy_document" "reaper" {
  statement {
    sid       = "Logs"
    actions   = ["logs:CreateLogStream", "logs:PutLogEvents"]
    resources = ["${aws_cloudwatch_log_group.reaper.arn}:*"]
  }

  statement {
    sid       = "Describe"
    actions   = ["ec2:DescribeInstances", "ec2:DescribeVolumes"]
    resources = ["*"]
  }

  statement {
    sid       = "TerminateManagedInstances"
    actions   = ["ec2:TerminateInstances"]
    resources = ["${local.ec2_arn}:instance/*"]

    condition {
      test     = "StringEquals"
      variable = "ec2:ResourceTag/loom:managed"
      values   = ["true"]
    }
  }

  statement {
    sid       = "DeleteManagedVolumes"
    actions   = ["ec2:DeleteVolume"]
    resources = ["${local.ec2_arn}:volume/*"]

    condition {
      test     = "StringEquals"
      variable = "ec2:ResourceTag/loom:managed"
      values   = ["true"]
    }
  }
}

resource "aws_iam_role_policy" "reaper" {
  name   = "reaper"
  role   = aws_iam_role.reaper.id
  policy = data.aws_iam_policy_document.reaper.json
}

resource "aws_lambda_function" "reaper" {
  function_name    = "${var.name_prefix}-reaper"
  role             = aws_iam_role.reaper.arn
  runtime          = "python3.12"
  handler          = "aws_reaper.lambda_handler"
  filename         = data.archive_file.reaper.output_path
  source_code_hash = data.archive_file.reaper.output_base64sha256
  timeout          = 120
  memory_size      = 128

  environment {
    variables = {
      LOOM_REAPER_MAX_AGE_HOURS = tostring(var.reaper_max_age_hours)
      LOOM_REAPER_DRY_RUN       = tostring(var.reaper_dry_run)
    }
  }

  depends_on = [aws_cloudwatch_log_group.reaper, aws_iam_role_policy.reaper]
}

resource "aws_cloudwatch_event_rule" "reaper" {
  name                = "${var.name_prefix}-reaper"
  description         = "Run the Loom bench TTL reaper"
  schedule_expression = "rate(15 minutes)"
}

resource "aws_cloudwatch_event_target" "reaper" {
  rule = aws_cloudwatch_event_rule.reaper.name
  arn  = aws_lambda_function.reaper.arn
}

resource "aws_lambda_permission" "reaper" {
  statement_id  = "AllowEventBridgeSchedule"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.reaper.function_name
  principal     = "events.amazonaws.com"
  source_arn    = aws_cloudwatch_event_rule.reaper.arn
}
