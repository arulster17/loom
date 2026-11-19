# Secret containers only: Terraform never holds a value (no aws_secretsmanager_secret_version),
# so values stay out of the state. Values come from `loom-deploy secrets init` and the
# user's runbook steps (docs/phase1-m3-spec.md §5.2, §6.3).

locals {
  # Key after loom/platform/ and the JSON keys the secret holds (for the description).
  platform_secrets = {
    "stripe"             = "secret_key, webhook_secret"
    "auth"               = "auth_secret, google_id, google_secret"
    "internal-token"     = "the internal token (plain string)"
    "app-db-password"    = "the loom_app role's database credential (plain string)"
    "redis"              = "the Redis credential, ca_crt, server_crt, server_key"
    "pod-ca"             = "crt, key"
    "poolmgr-client"     = "crt, key"
    "ses-smtp"           = "SMTP username and credential"
    "grafana"            = "the Grafana admin credential"
    "serving-runpod-key" = "the serving RunPod API key (empty until M4; only the serving reaper reads it)"
  }
}

resource "aws_secretsmanager_secret" "platform" {
  for_each = local.platform_secrets

  name                    = "loom/platform/${each.key}"
  description             = "Loom platform: ${each.value}"
  recovery_window_in_days = 7
}

# Each secret may be read by its one reader role. The serving key's reader is the serving
# reaper, not the instance role (which also has an explicit Deny on it, iam.tf).
resource "aws_secretsmanager_secret_policy" "platform" {
  for_each = local.platform_secrets

  secret_arn = aws_secretsmanager_secret.platform[each.key].arn
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Sid    = "OnlyTheReaderRole"
      Effect = "Allow"
      Principal = {
        AWS = each.key == "serving-runpod-key" ? aws_iam_role.serving_reaper.arn : aws_iam_role.instance.arn
      }
      Action   = ["secretsmanager:GetSecretValue"]
      Resource = "*"
    }]
  })
}
