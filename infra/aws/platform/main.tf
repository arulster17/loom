# Shared lookups and names (docs/phase1-m3-spec.md §5.1). IAM and bucket policies are
# jsonencode() locals built from these, so `terraform test` can read their exact content.

data "aws_partition" "current" {}

data "aws_caller_identity" "current" {}

data "aws_region" "current" {}

data "aws_availability_zones" "available" {
  state = "available"
}

locals {
  partition  = data.aws_partition.current.partition
  region     = data.aws_region.current.region
  account_id = data.aws_caller_identity.current.account_id

  # Every alarm notifies this topic on ALARM and OK (alarms.tf).
  alarm_topic_arns = [aws_sns_topic.alerts.arn]

  # ARN prefixes the policies share.
  secrets_arn_prefix = "arn:${local.partition}:secretsmanager:${local.region}:${local.account_id}:secret:loom/platform"
  ecr_repository_arn = "arn:${local.partition}:ecr:${local.region}:${local.account_id}:repository/loom/*"
  log_group_arn      = "arn:${local.partition}:logs:${local.region}:${local.account_id}:log-group:/loom/platform*"
}
