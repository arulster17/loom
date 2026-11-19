# A5 (docs/phase1-m3-spec.md §13, §5.2 secrets): every secret of §5.2 exists, none has an
# aws_secretsmanager_secret_version in Terraform, each resource policy names only its
# reader role (the instance role; for serving-runpod-key, the serving reaper's role).
#
# Addresses this file expects (the spec names none):
#   aws_secretsmanager_secret.platform         for_each, one per secret, keyed by the name
#                                               after "loom/platform/" (e.g. "stripe")
#   aws_secretsmanager_secret_policy.platform  for_each with the same keys
#   aws_iam_role.instance                      the instance role
#   aws_iam_role.serving_reaper                the serving reaper's Lambda role
# A policy may refer to its role by ARN, or by the ARN built from the role's name.

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

# Distinct role ARNs (the mock gives every role the same one).
override_resource {
  target          = aws_iam_role.instance
  override_during = plan
  values = {
    arn = "arn:aws:iam::123456789012:role/mock-instance-role"
  }
}

override_resource {
  target          = aws_iam_role.serving_reaper
  override_during = plan
  values = {
    arn = "arn:aws:iam::123456789012:role/mock-serving-reaper-role"
  }
}

run "every_secret_of_5_2_exists" {
  command = plan

  assert {
    condition = length(setsubtract(toset([
      "stripe", "auth", "internal-token", "app-db-password", "redis", "pod-ca",
      "poolmgr-client", "ses-smtp", "serving-runpod-key",
    ]), toset(keys(aws_secretsmanager_secret.platform)))) == 0
    error_message = "Missing secrets (keys of aws_secretsmanager_secret.platform): ${jsonencode(setsubtract(toset(["stripe", "auth", "internal-token", "app-db-password", "redis", "pod-ca", "poolmgr-client", "ses-smtp", "serving-runpod-key"]), toset(keys(aws_secretsmanager_secret.platform))))}."
  }

  assert {
    condition     = alltrue([for k, s in aws_secretsmanager_secret.platform : s.name == "loom/platform/${k}"])
    error_message = "Each secret is named loom/platform/<key>; got ${jsonencode({ for k, s in aws_secretsmanager_secret.platform : k => s.name })}."
  }

  assert {
    condition     = toset(keys(aws_secretsmanager_secret_policy.platform)) == toset(keys(aws_secretsmanager_secret.platform))
    error_message = "Every secret has a resource policy (aws_secretsmanager_secret_policy.platform, same keys)."
  }

  assert {
    condition     = alltrue([for k, p in aws_secretsmanager_secret_policy.platform : p.secret_arn == aws_secretsmanager_secret.platform[k].arn])
    error_message = "Each policy is attached to the secret with the same key."
  }
}

run "each_policy_names_only_its_reader_role" {
  command = plan

  assert {
    condition = alltrue([
      for k, p in aws_secretsmanager_secret_policy.platform :
      toset(regexall("arn:aws:iam::[0-9]{12}:role/[A-Za-z0-9+=,.@_/-]+", p.policy)) == (
        k == "serving-runpod-key"
        ? toset([aws_iam_role.serving_reaper.arn])
        : toset([aws_iam_role.instance.arn])
        ) || toset(regexall("arn:aws:iam::[0-9]{12}:role/[A-Za-z0-9+=,.@_/-]+", p.policy)) == (
        k == "serving-runpod-key"
        ? toset(["arn:aws:iam::123456789012:role/${aws_iam_role.serving_reaper.name}"])
        : toset(["arn:aws:iam::123456789012:role/${aws_iam_role.instance.name}"])
      )
    ])
    error_message = "Each secret policy names exactly one role: the instance role, or for serving-runpod-key only the serving reaper's role. Roles named: ${jsonencode({ for k, p in aws_secretsmanager_secret_policy.platform : k => regexall("arn:aws:iam::[0-9]{12}:role/[A-Za-z0-9+=,.@_/-]+", p.policy) })}."
  }

  assert {
    condition = alltrue([
      for p in values(aws_secretsmanager_secret_policy.platform) :
      contains(flatten([for s in jsondecode(p.policy).Statement : flatten([s.Action])]), "secretsmanager:GetSecretValue")
    ])
    error_message = "Each secret policy governs secretsmanager:GetSecretValue."
  }
}

run "source_has_no_secret_values" {
  command = plan

  module {
    source = "./tests/fixtures/source_scan"
  }

  assert {
    condition     = !contains(output.resource_types, "aws_secretsmanager_secret_version")
    error_message = "No aws_secretsmanager_secret_version: secret values never enter Terraform state (§5.2)."
  }
}
