# A6 (docs/phase1-m3-spec.md §13, §5.1, §5.2 iam, reaper_serving): the instance role's
# grants and their scoping (CloudWatch namespace condition, logs, secrets, ECR pull, the
# lease prefix, the site parameter), as an exact allowlist of actions per resource; the
# GitHub OIDC role trusts only this repository's `main` and `workflow_dispatch` and may
# only push to ECR (an allowlist too); the serving reaper's role has s3:ListBucket on the
# bucket and s3:GetObject on serving/leases/* only; trust policies (instance role: EC2
# only; serving reaper: Lambda only); the instance profile; the SES SMTP user.
#
# Allowlists, beyond the grants §5.2 lists:
#   instance role  tolerated, not required: secretsmanager:PutSecretValue and
#                  secretsmanager:DescribeSecret on loom/platform/* (`loom-deploy secrets
#                  init` and `ses-smtp-password` run on the box, §6.3); logs:CreateLogGroup
#                  and logs:DescribeLogStreams on /loom/platform*; ecr:BatchCheckLayerAvailability;
#                  ssm:GetParameters on the site parameter; s3:GetObject on an object
#                  resource ending /releases/* (the release manifest, §6.1, §6.6 step 1).
#   OIDC role      required: the ECR push actions on repository/loom/* and
#                  ecr:GetAuthorizationToken on *; tolerated: ecr:BatchGetImage on
#                  repository/loom/* (AWS's documented push permissions include it) and
#                  s3:PutObject on an object resource ending /releases/* (the workflow
#                  uploads releases/<sha>.json, §6.1).
# The release-manifest grants (the two /releases/* tolerances) are tolerated pending a spec
# decision: §5.2 does not list them, §6.1 and §6.6 need them, and the bucket is unnamed.
#
# Addresses this file expects (the spec names none):
#   aws_iam_role.instance, aws_iam_role_policy.instance           instance role, its inline policy
#   aws_iam_instance_profile.instance                             the instance's profile (role = instance role)
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
    condition = alltrue([
      for want in ["logs:CreateLogStream", "logs:PutLogEvents"] :
      anytrue(flatten([
        for s in jsondecode(aws_iam_role_policy.instance.policy).Statement : [
          for a in flatten([s.Action]) : [
            for r in flatten([s.Resource]) :
            a == want && startswith(r, "arn:aws:logs:us-east-1:123456789012:log-group:/loom/platform")
          ]
        ]
      ]))
    ])
    error_message = "The instance role may write logs (logs:CreateLogStream, logs:PutLogEvents) to /loom/platform* log groups."
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

  # Exact allowlist: every (action, resource) pair the policy grants is one of these
  # (§5.2, plus the tolerances in this file's header). The lease prefix may be written
  # from the bucket's arn or from its name.
  assert {
    condition = alltrue(flatten([
      for s in jsondecode(aws_iam_role_policy.instance.policy).Statement : [
        for a in flatten([s.Action]) : [
          for r in flatten([s.Resource]) :
          a == "secretsmanager:GetSecretValue" ? (
            startswith(r, "arn:aws:secretsmanager:us-east-1:123456789012:secret:loom/platform/")
            || r == "arn:aws:secretsmanager:us-east-1:123456789012:secret:rds!db-mock-AbCdEf"
          ) :
          contains(["secretsmanager:PutSecretValue", "secretsmanager:DescribeSecret"], a) ? (
            startswith(r, "arn:aws:secretsmanager:us-east-1:123456789012:secret:loom/platform/")
          ) :
          contains(["logs:CreateLogStream", "logs:PutLogEvents", "logs:CreateLogGroup", "logs:DescribeLogStreams"], a) ? (
            startswith(r, "arn:aws:logs:us-east-1:123456789012:log-group:/loom/platform")
          ) :
          a == "ecr:GetAuthorizationToken" ? r == "*" :
          contains(["ecr:BatchGetImage", "ecr:GetDownloadUrlForLayer", "ecr:BatchCheckLayerAvailability"], a) ? (
            startswith(r, "arn:aws:ecr:us-east-1:123456789012:repository/loom/")
          ) :
          a == "s3:PutObject" ? contains([
            "${aws_s3_bucket.leases.arn}/serving/leases/*",
            "arn:aws:s3:::${aws_s3_bucket.leases.bucket}/serving/leases/*",
          ], r) :
          a == "s3:GetObject" ? (
            contains([
              "${aws_s3_bucket.leases.arn}/serving/leases/*",
              "arn:aws:s3:::${aws_s3_bucket.leases.bucket}/serving/leases/*",
            ], r)
            || can(regex("^arn:aws:s3:::[^/*?]+/([^*?]+/)?releases/\\*$", r))
          ) :
          a == "cloudwatch:PutMetricData" ? true :
          contains(["ssm:GetParameter", "ssm:GetParameters"], a) ? r == "arn:aws:ssm:us-east-1:123456789012:parameter/loom/platform/site" :
          false
        ]
      ]
    ]))
    error_message = "Instance policy allowlist: secretsmanager:GetSecretValue on loom/platform/* and the RDS master secret (PutSecretValue/DescribeSecret on loom/platform/* only); logs:CreateLogStream/PutLogEvents (CreateLogGroup/DescribeLogStreams) on /loom/platform* only; ECR pull only (GetAuthorizationToken on *; BatchGetImage, GetDownloadUrlForLayer, BatchCheckLayerAvailability on repository/loom/*); s3:GetObject/PutObject on the lease bucket's serving/leases/* (s3:GetObject also on .../releases/*); cloudwatch:PutMetricData; ssm:GetParameter(s) on /loom/platform/site; nothing else. Got ${aws_iam_role_policy.instance.policy}."
  }

  assert {
    condition = alltrue([
      for a in flatten([for s in jsondecode(aws_iam_role_policy.instance.policy).Statement : flatten([s.Action])]) :
      !can(regex("(?i):(Delete|Put[A-Za-z]*Policy|Attach|Detach|PassRole|Create(Role|User|Policy|AccessKey|Secret)|Update|Restore|Tag|Untag|Rotate|Cancel)", a))
    ])
    error_message = "No Delete*, Put*Policy, IAM, tagging or other administrative action in the instance policy; got ${aws_iam_role_policy.instance.policy}."
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
    condition = anytrue(flatten([
      for s in jsondecode(aws_iam_role_policy.github_images.policy).Statement : [
        for a in flatten([s.Action]) : [
          for r in flatten([s.Resource]) :
          s.Effect == "Allow" && a == "ecr:GetAuthorizationToken" && r == "*"
        ]
      ]
    ]))
    error_message = "The OIDC role may ecr:GetAuthorizationToken (resource *), which docker login to ECR needs."
  }

  assert {
    condition = alltrue([
      for s in jsondecode(aws_iam_role_policy.github_images.policy).Statement :
      s.Effect == "Allow" && !can(s.NotAction) && !can(s.NotResource)
    ])
    error_message = "OIDC policy statements are plain Allow statements (no NotAction/NotResource)."
  }

  # Allowlist ("ECR push only", §5.2): every (action, resource) pair is an ECR push action
  # on repository/loom/*, GetAuthorizationToken on *, or the release manifest upload
  # (tolerated, see the header).
  assert {
    condition = alltrue(flatten([
      for s in jsondecode(aws_iam_role_policy.github_images.policy).Statement : [
        for a in flatten([s.Action]) : [
          for r in flatten([s.Resource]) :
          a == "ecr:GetAuthorizationToken" ? r == "*" :
          contains([
            "ecr:PutImage",
            "ecr:InitiateLayerUpload",
            "ecr:UploadLayerPart",
            "ecr:CompleteLayerUpload",
            "ecr:BatchCheckLayerAvailability",
            "ecr:BatchGetImage",
          ], a) ? startswith(r, "arn:aws:ecr:us-east-1:123456789012:repository/loom/") :
          a == "s3:PutObject" ? can(regex("^arn:aws:s3:::[^/*?]+/([^*?]+/)?releases/\\*$", r)) :
          false
        ]
      ]
    ]))
    error_message = "The OIDC role may only push images: ecr:GetAuthorizationToken on *; PutImage, InitiateLayerUpload, UploadLayerPart, CompleteLayerUpload, BatchCheckLayerAvailability (BatchGetImage) on repository/loom/*; s3:PutObject only on an object resource ending /releases/*; no other action. Got ${aws_iam_role_policy.github_images.policy}."
  }
}

run "trust_policies_and_instance_profile" {
  command = plan

  assert {
    condition = (
      length(jsondecode(aws_iam_role.instance.assume_role_policy).Statement) >= 1
      && alltrue([
        for s in jsondecode(aws_iam_role.instance.assume_role_policy).Statement :
        s.Effect == "Allow"
        && toset(flatten([s.Action])) == toset(["sts:AssumeRole"])
        && toset(keys(s.Principal)) == toset(["Service"])
        && toset(flatten([s.Principal.Service])) == toset(["ec2.amazonaws.com"])
      ])
    )
    error_message = "The instance role trusts ec2.amazonaws.com only (sts:AssumeRole); got ${aws_iam_role.instance.assume_role_policy}."
  }

  assert {
    condition = (
      length(jsondecode(aws_iam_role.serving_reaper.assume_role_policy).Statement) >= 1
      && alltrue([
        for s in jsondecode(aws_iam_role.serving_reaper.assume_role_policy).Statement :
        s.Effect == "Allow"
        && toset(flatten([s.Action])) == toset(["sts:AssumeRole"])
        && toset(keys(s.Principal)) == toset(["Service"])
        && toset(flatten([s.Principal.Service])) == toset(["lambda.amazonaws.com"])
      ])
    )
    error_message = "The serving reaper role trusts lambda.amazonaws.com only (sts:AssumeRole); got ${aws_iam_role.serving_reaper.assume_role_policy}."
  }

  assert {
    condition = (
      contains([aws_iam_instance_profile.instance.name, aws_iam_instance_profile.instance.id], aws_instance.platform.iam_instance_profile)
      && contains([aws_iam_role.instance.name, aws_iam_role.instance.id], aws_iam_instance_profile.instance.role)
    )
    error_message = "aws_instance.platform uses aws_iam_instance_profile.instance, whose role is aws_iam_role.instance."
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
