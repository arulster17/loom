# A6 (docs/phase1-m3-spec.md §13, §5.1, §5.2 iam, reaper_serving): the instance role's
# grants and their scoping (CloudWatch namespace condition, logs, secrets, ECR pull, the
# lease prefix, the site parameter); the GitHub OIDC role trusts only this repository's
# `main` and `workflow_dispatch` and can push to ECR; the serving reaper's role has
# s3:ListBucket on the bucket and s3:GetObject on serving/leases/* only; the SES SMTP user.
#
# Addresses this file expects (the spec names none):
#   aws_iam_role.instance, aws_iam_role_policy.instance           instance role, its inline policy
#   aws_iam_role_policy_attachment.instance_ssm                   AmazonSSMManagedInstanceCore
#   aws_iam_role.github_images, aws_iam_role_policy.github_images OIDC role for images.yml
#   aws_iam_role.serving_reaper, aws_iam_role_policy.serving_reaper
#   aws_iam_user.ses_smtp, aws_iam_user_policy.ses_smtp            the SES SMTP user
#   aws_secretsmanager_secret.platform["serving-runpod-key"]       (see platform_secrets)
# Policies are jsonencode() documents (§5.1); Statement is a list, Action and Resource a
# string or a list.
#
# Mocked values the expected ARNs use: account 123456789012, region us-east-1, partition
# aws, RDS master secret arn:aws:secretsmanager:us-east-1:123456789012:secret:rds!db-mock-AbCdEf,
# ECR repositories arn:aws:ecr:us-east-1:123456789012:repository/loom/..., lease bucket
# loom-mock-leases, site parameter arn:aws:ssm:us-east-1:123456789012:parameter/loom/platform/site.

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

run "instance_role_grants_of_5_2" {
  command = plan

  # Every grant §5.2 lists is there.
  assert {
    condition = anytrue(flatten([
      for s in jsondecode(aws_iam_role_policy.instance.policy).Statement : [
        for a in flatten([s.Action]) : a == "cloudwatch:PutMetricData"
      ]
    ]))
    error_message = "The instance role may cloudwatch:PutMetricData."
  }

  assert {
    condition = anytrue(flatten([
      for s in jsondecode(aws_iam_role_policy.instance.policy).Statement : [
        for a in flatten([s.Action]) : [
          for r in flatten([s.Resource]) :
          a == "logs:PutLogEvents" && startswith(r, "arn:aws:logs:us-east-1:123456789012:log-group:/loom/platform")
        ]
      ]
    ]))
    error_message = "The instance role may logs:PutLogEvents to /loom/platform* log groups."
  }

  assert {
    condition = anytrue(flatten([
      for s in jsondecode(aws_iam_role_policy.instance.policy).Statement : [
        for a in flatten([s.Action]) : [
          for r in flatten([s.Resource]) :
          a == "secretsmanager:GetSecretValue" && startswith(r, "arn:aws:secretsmanager:us-east-1:123456789012:secret:loom/platform/")
        ]
      ]
    ]))
    error_message = "The instance role may secretsmanager:GetSecretValue on loom/platform/*."
  }

  assert {
    condition = anytrue(flatten([
      for s in jsondecode(aws_iam_role_policy.instance.policy).Statement : [
        for a in flatten([s.Action]) : [
          for r in flatten([s.Resource]) :
          a == "secretsmanager:GetSecretValue" && r == "arn:aws:secretsmanager:us-east-1:123456789012:secret:rds!db-mock-AbCdEf"
        ]
      ]
    ]))
    error_message = "The instance role may secretsmanager:GetSecretValue on the RDS master secret (aws_db_instance.platform.master_user_secret[0].secret_arn)."
  }

  assert {
    condition = alltrue([
      for want in ["ecr:BatchGetImage", "ecr:GetDownloadUrlForLayer"] :
      anytrue(flatten([
        for s in jsondecode(aws_iam_role_policy.instance.policy).Statement : [
          for a in flatten([s.Action]) : [
            for r in flatten([s.Resource]) :
            a == want && startswith(r, "arn:aws:ecr:us-east-1:123456789012:repository/loom/")
          ]
        ]
      ]))
    ]) && anytrue(flatten([for s in jsondecode(aws_iam_role_policy.instance.policy).Statement : [for a in flatten([s.Action]) : a == "ecr:GetAuthorizationToken"]]))
    error_message = "ECR pull on the Loom repositories: ecr:GetAuthorizationToken, and ecr:BatchGetImage and ecr:GetDownloadUrlForLayer on repository/loom/*."
  }

  assert {
    condition = alltrue([
      for want in ["s3:GetObject", "s3:PutObject"] :
      anytrue(flatten([
        for s in jsondecode(aws_iam_role_policy.instance.policy).Statement : [
          for a in flatten([s.Action]) : [
            for r in flatten([s.Resource]) :
            a == want && can(regex("^arn:aws:s3:::[^/*]+/serving/leases/\\*$", r))
          ]
        ]
      ]))
    ])
    error_message = "S3 read/write (s3:GetObject, s3:PutObject) on the lease bucket's serving/leases/*."
  }

  assert {
    condition = anytrue(flatten([
      for s in jsondecode(aws_iam_role_policy.instance.policy).Statement : [
        for a in flatten([s.Action]) : [
          for r in flatten([s.Resource]) :
          a == "ssm:GetParameter" && r == "arn:aws:ssm:us-east-1:123456789012:parameter/loom/platform/site"
        ]
      ]
    ]))
    error_message = "The instance role may ssm:GetParameter on /loom/platform/site (§6.11)."
  }

  assert {
    condition = (
      endswith(aws_iam_role_policy_attachment.instance_ssm.policy_arn, ":iam::aws:policy/AmazonSSMManagedInstanceCore")
      && contains([aws_iam_role.instance.name, aws_iam_role.instance.id], aws_iam_role_policy_attachment.instance_ssm.role)
    )
    error_message = "AmazonSSMManagedInstanceCore is attached to the instance role."
  }

  assert {
    condition     = aws_iam_role_policy.instance.role == aws_iam_role.instance.name || aws_iam_role_policy.instance.role == aws_iam_role.instance.id
    error_message = "aws_iam_role_policy.instance belongs to aws_iam_role.instance."
  }
}

run "instance_role_grants_are_scoped" {
  command = plan

  assert {
    condition = alltrue([
      for s in jsondecode(aws_iam_role_policy.instance.policy).Statement :
      s.Effect == "Allow" && !can(s.NotAction) && !can(s.NotResource)
    ])
    error_message = "Instance policy statements are plain Allow statements (no NotAction/NotResource)."
  }

  assert {
    condition = alltrue([
      for a in flatten([for s in jsondecode(aws_iam_role_policy.instance.policy).Statement : flatten([s.Action])]) :
      !strcontains(a, "*")
    ])
    error_message = "No wildcard actions in the instance policy."
  }

  # cloudwatch:PutMetricData only with the namespace condition, in a statement of its own.
  assert {
    condition = alltrue([
      for s in jsondecode(aws_iam_role_policy.instance.policy).Statement :
      (
        toset(flatten([s.Action])) == toset(["cloudwatch:PutMetricData"])
        && toset(flatten([try(s.Condition.StringEquals["cloudwatch:namespace"], [])])) == toset(["Loom/Platform"])
      )
      if anytrue([for a in flatten([s.Action]) : startswith(a, "cloudwatch:")])
    ])
    error_message = "cloudwatch: only PutMetricData, in a statement with Condition StringEquals cloudwatch:namespace = Loom/Platform; got ${jsonencode([for s in jsondecode(aws_iam_role_policy.instance.policy).Statement : s if anytrue([for a in flatten([s.Action]) : startswith(a, "cloudwatch:")])])}."
  }

  # Each service's actions only on its resources.
  assert {
    condition = alltrue(flatten([
      for s in jsondecode(aws_iam_role_policy.instance.policy).Statement : [
        for a in flatten([s.Action]) : [
          for r in flatten([s.Resource]) :
          startswith(a, "logs:") ? startswith(r, "arn:aws:logs:us-east-1:123456789012:log-group:/loom/platform") :
          startswith(a, "secretsmanager:") ? (startswith(r, "arn:aws:secretsmanager:us-east-1:123456789012:secret:loom/platform/") || r == "arn:aws:secretsmanager:us-east-1:123456789012:secret:rds!db-mock-AbCdEf") :
          startswith(a, "ssm:") ? (a == "ssm:GetParameter" && r == "arn:aws:ssm:us-east-1:123456789012:parameter/loom/platform/site") :
          startswith(a, "ecr:") ? (
            a == "ecr:GetAuthorizationToken" ? r == "*" : (
              contains(["ecr:BatchGetImage", "ecr:GetDownloadUrlForLayer", "ecr:BatchCheckLayerAvailability"], a)
              && startswith(r, "arn:aws:ecr:us-east-1:123456789012:repository/loom/")
            )
          ) :
          startswith(a, "s3:") ? (
            contains(["s3:GetObject", "s3:PutObject", "s3:DeleteObject"], a) && r != "*" && !startswith(r, "arn:aws:s3:::*")
          ) :
          startswith(a, "cloudwatch:") ? true :
          false
        ]
      ]
    ]))
    error_message = "Instance policy, by service: logs only on /loom/platform*; secretsmanager only on loom/platform/* and the RDS master secret; ssm only GetParameter on /loom/platform/site; ECR pull only (GetAuthorizationToken on *, the rest on repository/loom/*); S3 object read/write only, never on every bucket; no other service. Got ${aws_iam_role_policy.instance.policy}."
  }
}

run "github_oidc_role_trusts_only_main_and_workflow_dispatch" {
  command = plan

  assert {
    condition     = length(jsondecode(aws_iam_role.github_images.assume_role_policy).Statement) >= 1
    error_message = "The OIDC role has a trust policy."
  }

  assert {
    condition = alltrue([
      for s in jsondecode(aws_iam_role.github_images.assume_role_policy).Statement :
      s.Effect == "Allow"
      && toset(flatten([s.Action])) == toset(["sts:AssumeRoleWithWebIdentity"])
      && toset(keys(s.Principal)) == toset(["Federated"])
      && alltrue([for f in flatten([s.Principal.Federated]) : endswith(f, "oidc-provider/token.actions.githubusercontent.com")])
    ])
    error_message = "The OIDC role is assumable only with sts:AssumeRoleWithWebIdentity by GitHub's OIDC provider (token.actions.githubusercontent.com)."
  }

  assert {
    condition = alltrue([
      for s in jsondecode(aws_iam_role.github_images.assume_role_policy).Statement :
      toset(flatten([try(s.Condition.StringEquals["token.actions.githubusercontent.com:aud"], [])])) == toset(["sts.amazonaws.com"])
    ])
    error_message = "Condition StringEquals token.actions.githubusercontent.com:aud = sts.amazonaws.com."
  }

  assert {
    condition = alltrue([
      for s in jsondecode(aws_iam_role.github_images.assume_role_policy).Statement :
      length(flatten([
        try(s.Condition.StringEquals["token.actions.githubusercontent.com:sub"], []),
        try(s.Condition.StringLike["token.actions.githubusercontent.com:sub"], []),
      ])) > 0
      && alltrue([
        for v in flatten([
          try(s.Condition.StringEquals["token.actions.githubusercontent.com:sub"], []),
          try(s.Condition.StringLike["token.actions.githubusercontent.com:sub"], []),
        ]) :
        startswith(v, "repo:arulster17/loom:")
        && can(regex("refs/heads/main(:|$)", v))
        && !strcontains(v, "*")
        && !strcontains(v, "?")
      ])
    ])
    error_message = "Every sub the OIDC role accepts is this repository's (repo:arulster17/loom:) main branch (refs/heads/main), with no wildcard; got ${aws_iam_role.github_images.assume_role_policy}."
  }

  assert {
    condition = alltrue([
      for s in jsondecode(aws_iam_role.github_images.assume_role_policy).Statement :
      strcontains(jsonencode(try(s.Condition, {})), "workflow_dispatch")
    ])
    error_message = "The OIDC role's trust conditions restrict the event to workflow_dispatch (§6.1: manual dispatch on main)."
  }
}

run "github_oidc_role_can_push_images" {
  command = plan

  assert {
    condition = alltrue([
      for want in ["ecr:PutImage", "ecr:InitiateLayerUpload", "ecr:UploadLayerPart", "ecr:CompleteLayerUpload", "ecr:BatchCheckLayerAvailability"] :
      anytrue(flatten([
        for s in jsondecode(aws_iam_role_policy.github_images.policy).Statement : [
          for a in flatten([s.Action]) : [
            for r in flatten([s.Resource]) :
            s.Effect == "Allow" && a == want && startswith(r, "arn:aws:ecr:us-east-1:123456789012:repository/loom/")
          ]
        ]
      ]))
    ])
    error_message = "The OIDC role can push to the Loom repositories (PutImage and the layer upload actions on repository/loom/*)."
  }

  assert {
    condition = alltrue(flatten([
      for s in jsondecode(aws_iam_role_policy.github_images.policy).Statement : [
        for a in flatten([s.Action]) : [
          for r in flatten([s.Resource]) :
          !startswith(a, "ecr:") || (a == "ecr:GetAuthorizationToken" ? r == "*" : startswith(r, "arn:aws:ecr:us-east-1:123456789012:repository/loom/"))
        ]
      ]
    ]))
    error_message = "ECR actions of the OIDC role apply to the Loom repositories only (GetAuthorizationToken on *)."
  }

  assert {
    condition = alltrue([
      for a in flatten([for s in jsondecode(aws_iam_role_policy.github_images.policy).Statement : flatten([s.Action])]) :
      !strcontains(a, "*")
      && !can(regex("^ecr:(Delete|BatchDelete|Set|Create|Tag|Untag|PutLifecyclePolicy|PutImageTagMutability|PutImageScanningConfiguration|PutRegistry|PutReplication)", a))
      && !startswith(a, "ssm:")
      && !startswith(a, "secretsmanager:")
      && !startswith(a, "iam:")
    ])
    error_message = "The OIDC role pushes images only: no wildcard, no repository administration or deletion, no SSM, secrets or IAM."
  }
}

run "serving_reaper_role_reads_leases_only" {
  command = plan

  assert {
    condition = anytrue(flatten([
      for s in jsondecode(aws_iam_role_policy.serving_reaper.policy).Statement : [
        for a in flatten([s.Action]) : [
          for r in flatten([s.Resource]) :
          s.Effect == "Allow" && a == "s3:ListBucket" && can(regex("^arn:aws:s3:::[^/*]+$", r))
        ]
      ]
    ]))
    error_message = "The serving reaper role has s3:ListBucket on the lease bucket."
  }

  assert {
    condition = anytrue(flatten([
      for s in jsondecode(aws_iam_role_policy.serving_reaper.policy).Statement : [
        for a in flatten([s.Action]) : [
          for r in flatten([s.Resource]) :
          s.Effect == "Allow" && a == "s3:GetObject" && can(regex("^arn:aws:s3:::[^/*]+/serving/leases/\\*$", r))
        ]
      ]
    ]))
    error_message = "The serving reaper role has s3:GetObject on serving/leases/*."
  }

  assert {
    condition = alltrue(flatten([
      for s in jsondecode(aws_iam_role_policy.serving_reaper.policy).Statement : [
        for a in flatten([s.Action]) : [
          for r in flatten([s.Resource]) :
          !strcontains(a, "*") && (
            startswith(a, "logs:") ? true :
            a == "s3:ListBucket" ? can(regex("^arn:aws:s3:::[^/*]+$", r)) :
            a == "s3:GetObject" ? can(regex("^arn:aws:s3:::[^/*]+/serving/leases/\\*$", r)) :
            a == "secretsmanager:GetSecretValue" ? startswith(r, "arn:aws:secretsmanager:us-east-1:123456789012:secret:loom/platform/serving-runpod-key") :
            false
          )
        ]
      ]
    ]))
    error_message = "The serving reaper role may only write its logs, read its key (serving-runpod-key), list the lease bucket and get serving/leases/*: no other S3 action or prefix (P1). Got ${aws_iam_role_policy.serving_reaper.policy}."
  }

  assert {
    condition     = aws_iam_role_policy.serving_reaper.role == aws_iam_role.serving_reaper.name || aws_iam_role_policy.serving_reaper.role == aws_iam_role.serving_reaper.id
    error_message = "aws_iam_role_policy.serving_reaper belongs to aws_iam_role.serving_reaper."
  }
}

run "ses_smtp_user_sends_from_the_domain_identity_only" {
  command = plan

  assert {
    condition     = aws_iam_user.ses_smtp.name == "loom-ses-smtp"
    error_message = "The SES SMTP user is loom-ses-smtp."
  }

  assert {
    condition = (
      toset(flatten([for s in jsondecode(aws_iam_user_policy.ses_smtp.policy).Statement : flatten([s.Action])])) == toset(["ses:SendRawEmail"])
      && toset(flatten([for s in jsondecode(aws_iam_user_policy.ses_smtp.policy).Statement : flatten([s.Resource])])) == toset(["arn:aws:ses:us-east-1:123456789012:identity/example.test"])
    )
    error_message = "loom-ses-smtp may only ses:SendRawEmail on the domain identity; got ${aws_iam_user_policy.ses_smtp.policy}."
  }
}

run "source_policies_are_jsonencode_and_no_access_key" {
  command = plan

  module {
    source = "./tests/fixtures/source_scan"
  }

  assert {
    condition     = !contains(output.data_types, "aws_iam_policy_document")
    error_message = "IAM and bucket policies are jsonencode() locals, not aws_iam_policy_document (§5.1)."
  }

  assert {
    condition     = !contains(output.resource_types, "aws_iam_access_key")
    error_message = "The SES SMTP access key is made by the runbook, not Terraform (its secret stays out of state)."
  }
}
