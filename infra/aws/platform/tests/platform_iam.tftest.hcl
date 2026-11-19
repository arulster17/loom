# A6 (docs/phase1-m3-spec.md §13, §5.1, §5.2 iam, §5.6, reaper_serving): the instance
# role's grants and their scoping (CloudWatch namespace condition, logs, secrets, ECR
# pull, the lease prefix, the releases prefix, the site parameter), as an exact allowlist
# of actions per resource, plus the required explicit Deny on serving-runpod-key; the
# GitHub OIDC role trusts only this repository's GitHub environment `platform-images`
# and may only push to ECR and upload to the releases bucket's releases/* (an allowlist
# too); the serving reaper's role has s3:ListBucket on the bucket and s3:GetObject on
# serving/leases/* only; trust policies (instance role: EC2 only; serving reaper: Lambda
# only); the instance profile; the SES SMTP user.
#
# §5.6 settles what §5.2 left open. "Exactly" means §5.2 plus §5.6: optional grants may be
# left out, nothing else may be added.
#   instance role  required (§5.6): secretsmanager:PutSecretValue and DescribeSecret on
#                  loom/platform/*; a Deny of secretsmanager:* (or of a list of actions
#                  that includes GetSecretValue) on the serving-runpod-key secret's ARN,
#                  with no condition; s3:GetObject on the releases bucket's releases/*;
#                  s3:DeleteObject on the lease bucket's serving/leases/*, the only Delete
#                  action allowed, and only there.
#                  optional: secretsmanager:ListSecretVersionIds on loom/platform/*;
#                  logs:CreateLogGroup and logs:DescribeLogStreams on /loom/platform*;
#                  ecr:BatchCheckLayerAvailability; ssm:GetParameters on the site parameter.
#                  The allow-side checks read only the Allow statements; the Deny
#                  statements are checked on their own (secretsmanager actions on the
#                  serving key only).
#   OIDC role      required: the ECR push actions on repository/loom/*,
#                  ecr:GetAuthorizationToken on *, s3:PutObject on the releases bucket's
#                  releases/*; optional: ecr:BatchGetImage on repository/loom/*.
#                  Trust: Condition StringEquals token.actions.githubusercontent.com:sub =
#                  repo:arulster17/loom:environment:platform-images and :aud =
#                  sts.amazonaws.com, and no other condition (IAM cannot test a branch and
#                  an event together; the environment holds the branch rule, §5.6).
#   SES SMTP user  ses:SendRawEmail on identity/* and the configuration set's ARN, with
#                  Condition StringEquals ses:FromAddress = no-reply@<domain>.
# The releases prefix must be on the releases bucket itself (written from
# aws_s3_bucket.releases or var.releases_bucket), not on any bucket.
#
# Addresses this file expects (the spec names none):
#   aws_iam_role.instance, aws_iam_role_policy.instance           instance role, its inline policy
#   aws_iam_instance_profile.instance                             the instance's profile (role = instance role)
#   aws_iam_role_policy_attachment.instance_ssm                   AmazonSSMManagedInstanceCore
#   aws_iam_role.github_images, aws_iam_role_policy.github_images OIDC role for images.yml
#   aws_iam_role.serving_reaper, aws_iam_role_policy.serving_reaper
#   aws_iam_user.ses_smtp, aws_iam_user_policy.ses_smtp            the SES SMTP user
#   aws_secretsmanager_secret.platform["serving-runpod-key"]       (see platform_secrets)
#   aws_s3_bucket.leases, aws_s3_bucket.releases                   the two buckets (§5.2 s3, §5.6)
# Policies are jsonencode() documents (§5.1); Statement is a list, Action and Resource a
# string or a list.
#
# Mocked values the expected ARNs use: account 123456789012, region us-east-1, partition
# aws, RDS master secret arn:aws:secretsmanager:us-east-1:123456789012:secret:rds!db-mock-AbCdEf,
# ECR repositories arn:aws:ecr:us-east-1:123456789012:repository/loom/..., lease bucket
# loom-mock-leases, releases bucket loom-test-releases (overridden below: the mock gives
# every bucket the lease bucket's ARN), site parameter
# arn:aws:ssm:us-east-1:123456789012:parameter/loom/platform/site, configuration set
# arn:aws:ses:us-east-1:123456789012:configuration-set/<name>.

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
  domain          = "example.test"
  brand_name      = "Example Brand"
  ops_email       = "ops@example.test"
  budget_usd      = 75
  releases_bucket = "loom-test-releases"
}

override_resource {
  target          = aws_secretsmanager_secret.platform["serving-runpod-key"]
  override_during = plan
  values = {
    id  = "arn:aws:secretsmanager:us-east-1:123456789012:secret:loom/platform/serving-runpod-key-AbCdEf"
    arn = "arn:aws:secretsmanager:us-east-1:123456789012:secret:loom/platform/serving-runpod-key-AbCdEf"
  }
}

override_resource {
  target          = aws_s3_bucket.releases
  override_during = plan
  values = {
    id  = "loom-test-releases"
    arn = "arn:aws:s3:::loom-test-releases"
  }
}

run "instance_role_grants_of_5_2" {
  command = plan

  # Every grant §5.2 and §5.6 require is there, in an Allow statement.
  assert {
    condition = anytrue(flatten([
      for s in jsondecode(aws_iam_role_policy.instance.policy).Statement : [
        for a in flatten([s.Action]) : s.Effect == "Allow" && a == "cloudwatch:PutMetricData"
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
            s.Effect == "Allow" && a == want && startswith(r, "arn:aws:logs:us-east-1:123456789012:log-group:/loom/platform")
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
          s.Effect == "Allow" && a == "secretsmanager:GetSecretValue" && startswith(r, "arn:aws:secretsmanager:us-east-1:123456789012:secret:loom/platform/")
        ]
      ]
    ]))
    error_message = "The instance role may secretsmanager:GetSecretValue on loom/platform/*."
  }

  # §5.6: `secrets init` and `ses-smtp-password` store values on the box and check first.
  assert {
    condition = alltrue([
      for want in ["secretsmanager:PutSecretValue", "secretsmanager:DescribeSecret"] :
      anytrue(flatten([
        for s in jsondecode(aws_iam_role_policy.instance.policy).Statement : [
          for a in flatten([s.Action]) : [
            for r in flatten([s.Resource]) :
            s.Effect == "Allow" && a == want && startswith(r, "arn:aws:secretsmanager:us-east-1:123456789012:secret:loom/platform/")
          ]
        ]
      ]))
    ])
    error_message = "The instance role may secretsmanager:PutSecretValue and secretsmanager:DescribeSecret on loom/platform/* (§5.6, required); got ${aws_iam_role_policy.instance.policy}."
  }

  assert {
    condition = anytrue(flatten([
      for s in jsondecode(aws_iam_role_policy.instance.policy).Statement : [
        for a in flatten([s.Action]) : [
          for r in flatten([s.Resource]) :
          s.Effect == "Allow" && a == "secretsmanager:GetSecretValue" && r == "arn:aws:secretsmanager:us-east-1:123456789012:secret:rds!db-mock-AbCdEf"
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
            s.Effect == "Allow" && a == want && startswith(r, "arn:aws:ecr:us-east-1:123456789012:repository/loom/")
          ]
        ]
      ]))
    ]) && anytrue(flatten([for s in jsondecode(aws_iam_role_policy.instance.policy).Statement : [for a in flatten([s.Action]) : s.Effect == "Allow" && a == "ecr:GetAuthorizationToken"]]))
    error_message = "ECR pull on the Loom repositories: ecr:GetAuthorizationToken, and ecr:BatchGetImage and ecr:GetDownloadUrlForLayer on repository/loom/*."
  }

  # The lease prefix may be written from the bucket's arn or from its name.
  assert {
    condition = alltrue([
      for want in ["s3:GetObject", "s3:PutObject", "s3:DeleteObject"] :
      anytrue(flatten([
        for s in jsondecode(aws_iam_role_policy.instance.policy).Statement : [
          for a in flatten([s.Action]) : [
            for r in flatten([s.Resource]) :
            s.Effect == "Allow" && a == want && contains([
              "${aws_s3_bucket.leases.arn}/serving/leases/*",
              "arn:aws:s3:::${aws_s3_bucket.leases.bucket}/serving/leases/*",
            ], r)
          ]
        ]
      ]))
    ])
    error_message = "S3 read/write on the lease bucket's serving/leases/*: s3:GetObject, s3:PutObject and (§5.6, required) s3:DeleteObject; got ${aws_iam_role_policy.instance.policy}."
  }

  # §5.6: `apply` and the runbook fetch the release manifest and the files beside it.
  assert {
    condition = anytrue(flatten([
      for s in jsondecode(aws_iam_role_policy.instance.policy).Statement : [
        for a in flatten([s.Action]) : [
          for r in flatten([s.Resource]) :
          s.Effect == "Allow" && a == "s3:GetObject" && contains([
            "${aws_s3_bucket.releases.arn}/releases/*",
            "arn:aws:s3:::${aws_s3_bucket.releases.bucket}/releases/*",
            "arn:aws:s3:::loom-test-releases/releases/*",
          ], r)
        ]
      ]
    ]))
    error_message = "The instance role may s3:GetObject on the releases bucket's releases/* (arn:aws:s3:::loom-test-releases/releases/*, §5.6); got ${aws_iam_role_policy.instance.policy}."
  }

  # §5.6: an explicit Deny keeps the box from reading the serving RunPod key, which the
  # loom/platform/* grants would otherwise allow.
  assert {
    condition = anytrue([
      for s in jsondecode(aws_iam_role_policy.instance.policy).Statement :
      s.Effect == "Deny"
      && can(s.Action) && can(s.Resource) && length(try(s.Condition, {})) == 0
      && (
        contains(flatten([s.Action]), "secretsmanager:*")
        || contains(flatten([s.Action]), "secretsmanager:GetSecretValue")
      )
      && contains(flatten([s.Resource]), aws_secretsmanager_secret.platform["serving-runpod-key"].arn)
    ])
    error_message = "The instance policy has a Deny of secretsmanager:* (or of actions including secretsmanager:GetSecretValue) on the serving-runpod-key secret's ARN, with no condition (§5.6); got ${aws_iam_role_policy.instance.policy}."
  }

  assert {
    condition = anytrue(flatten([
      for s in jsondecode(aws_iam_role_policy.instance.policy).Statement : [
        for a in flatten([s.Action]) : [
          for r in flatten([s.Resource]) :
          s.Effect == "Allow" && a == "ssm:GetParameter" && r == "arn:aws:ssm:us-east-1:123456789012:parameter/loom/platform/site"
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
      contains(["Allow", "Deny"], s.Effect) && can(s.Action) && can(s.Resource)
      && !can(s.NotAction) && !can(s.NotResource) && !can(s.Principal) && !can(s.NotPrincipal)
    ])
    error_message = "Instance policy statements are plain Allow or Deny statements with Action and Resource (no NotAction/NotResource/Principal)."
  }

  # The Deny statements only take away: secretsmanager actions on the serving key only
  # (§5.6). The allow-side checks below read only the Allow statements.
  assert {
    condition = alltrue(flatten([
      for s in jsondecode(aws_iam_role_policy.instance.policy).Statement : [
        for a in flatten([s.Action]) : [
          for r in flatten([s.Resource]) :
          startswith(a, "secretsmanager:")
          && startswith(r, "arn:aws:secretsmanager:us-east-1:123456789012:secret:loom/platform/serving-runpod-key")
        ]
      ] if s.Effect == "Deny"
    ]))
    error_message = "The instance policy's only Deny is on the serving-runpod-key secret (secretsmanager actions); got ${aws_iam_role_policy.instance.policy}."
  }

  assert {
    condition = alltrue([
      for a in flatten([for s in jsondecode(aws_iam_role_policy.instance.policy).Statement : flatten([s.Action]) if s.Effect == "Allow"]) :
      !strcontains(a, "*")
    ])
    error_message = "No wildcard actions in the instance policy's Allow statements."
  }

  # cloudwatch:PutMetricData only with the namespace condition, in a statement of its own.
  assert {
    condition = alltrue([
      for s in jsondecode(aws_iam_role_policy.instance.policy).Statement :
      (
        toset(flatten([s.Action])) == toset(["cloudwatch:PutMetricData"])
        && toset(flatten([try(s.Condition.StringEquals["cloudwatch:namespace"], [])])) == toset(["Loom/Platform"])
      )
      if s.Effect == "Allow" && anytrue([for a in flatten([s.Action]) : startswith(a, "cloudwatch:")])
    ])
    error_message = "cloudwatch: only PutMetricData, in a statement with Condition StringEquals cloudwatch:namespace = Loom/Platform; got ${jsonencode([for s in jsondecode(aws_iam_role_policy.instance.policy).Statement : s if anytrue([for a in flatten([s.Action]) : startswith(a, "cloudwatch:")])])}."
  }

  # Exact allowlist: every (action, resource) pair the policy allows is one of these
  # (§5.2 plus §5.6). The lease and releases prefixes may be written from the bucket's arn
  # or from its name.
  assert {
    condition = alltrue(flatten([
      for s in jsondecode(aws_iam_role_policy.instance.policy).Statement : [
        for a in flatten([s.Action]) : [
          for r in flatten([s.Resource]) :
          s.Effect != "Allow" ? true :
          a == "secretsmanager:GetSecretValue" ? (
            startswith(r, "arn:aws:secretsmanager:us-east-1:123456789012:secret:loom/platform/")
            || r == "arn:aws:secretsmanager:us-east-1:123456789012:secret:rds!db-mock-AbCdEf"
          ) :
          contains(["secretsmanager:PutSecretValue", "secretsmanager:DescribeSecret", "secretsmanager:ListSecretVersionIds"], a) ? (
            startswith(r, "arn:aws:secretsmanager:us-east-1:123456789012:secret:loom/platform/")
          ) :
          contains(["logs:CreateLogStream", "logs:PutLogEvents", "logs:CreateLogGroup", "logs:DescribeLogStreams"], a) ? (
            startswith(r, "arn:aws:logs:us-east-1:123456789012:log-group:/loom/platform")
          ) :
          a == "ecr:GetAuthorizationToken" ? r == "*" :
          contains(["ecr:BatchGetImage", "ecr:GetDownloadUrlForLayer", "ecr:BatchCheckLayerAvailability"], a) ? (
            startswith(r, "arn:aws:ecr:us-east-1:123456789012:repository/loom/")
          ) :
          contains(["s3:PutObject", "s3:DeleteObject"], a) ? contains([
            "${aws_s3_bucket.leases.arn}/serving/leases/*",
            "arn:aws:s3:::${aws_s3_bucket.leases.bucket}/serving/leases/*",
          ], r) :
          a == "s3:GetObject" ? contains([
            "${aws_s3_bucket.leases.arn}/serving/leases/*",
            "arn:aws:s3:::${aws_s3_bucket.leases.bucket}/serving/leases/*",
            "${aws_s3_bucket.releases.arn}/releases/*",
            "arn:aws:s3:::${aws_s3_bucket.releases.bucket}/releases/*",
          ], r) :
          a == "cloudwatch:PutMetricData" ? true :
          contains(["ssm:GetParameter", "ssm:GetParameters"], a) ? r == "arn:aws:ssm:us-east-1:123456789012:parameter/loom/platform/site" :
          false
        ]
      ]
    ]))
    error_message = "Instance policy allowlist: secretsmanager:GetSecretValue on loom/platform/* and the RDS master secret (PutSecretValue/DescribeSecret/ListSecretVersionIds on loom/platform/* only); logs:CreateLogStream/PutLogEvents (CreateLogGroup/DescribeLogStreams) on /loom/platform* only; ECR pull only (GetAuthorizationToken on *; BatchGetImage, GetDownloadUrlForLayer, BatchCheckLayerAvailability on repository/loom/*); s3:GetObject/PutObject/DeleteObject on the lease bucket's serving/leases/* and s3:GetObject on the releases bucket's releases/* (no other bucket or prefix); cloudwatch:PutMetricData; ssm:GetParameter(s) on /loom/platform/site; nothing else. Got ${aws_iam_role_policy.instance.policy}."
  }

  # No destructive or administrative action, per (action, resource) pair. The one
  # exemption is s3:DeleteObject on the lease prefix (the forms the allowlist accepts).
  assert {
    condition = alltrue(flatten([
      for s in jsondecode(aws_iam_role_policy.instance.policy).Statement : [
        for a in flatten([s.Action]) : [
          for r in flatten([s.Resource]) :
          s.Effect != "Allow" || (
            a == "s3:DeleteObject" && contains([
              "${aws_s3_bucket.leases.arn}/serving/leases/*",
              "arn:aws:s3:::${aws_s3_bucket.leases.bucket}/serving/leases/*",
            ], r)
          )
          || !can(regex("(?i):(Delete|Put[A-Za-z]*Policy|Attach|Detach|PassRole|Create(Role|User|Policy|AccessKey|Secret)|Update|Restore|Tag|Untag|Rotate|Cancel)", a))
        ]
      ]
    ]))
    error_message = "No Delete*, Put*Policy, IAM, tagging or other administrative action in the instance policy (s3:DeleteObject only on the lease bucket's serving/leases/*); got ${aws_iam_role_policy.instance.policy}."
  }
}

run "github_oidc_role_trusts_only_the_platform_images_environment" {
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

  # §5.6: the subject is exactly the GitHub environment platform-images of this repository
  # (the environment's deployment-branch rule holds "main only"; images.yml, the only
  # workflow naming it, runs on workflow_dispatch only). No other subject, no wildcard.
  assert {
    condition = alltrue([
      for s in jsondecode(aws_iam_role.github_images.assume_role_policy).Statement :
      toset(flatten([try(s.Condition.StringEquals["token.actions.githubusercontent.com:sub"], [])])) == toset(["repo:arulster17/loom:environment:platform-images"])
    ])
    error_message = "Every trust statement has Condition StringEquals token.actions.githubusercontent.com:sub = repo:arulster17/loom:environment:platform-images and no other subject; got ${aws_iam_role.github_images.assume_role_policy}."
  }

  # Nothing else is tested: StringEquals on aud and sub are the only conditions. A
  # refs/heads/main or workflow_dispatch check (job_workflow_ref, event_name, a second sub)
  # would never match an environment's token (IAM cannot test a branch and an event
  # together, §5.6).
  assert {
    condition = alltrue([
      for s in jsondecode(aws_iam_role.github_images.assume_role_policy).Statement :
      toset(keys(try(s.Condition, {}))) == toset(["StringEquals"])
      && toset(keys(s.Condition.StringEquals)) == toset([
        "token.actions.githubusercontent.com:aud",
        "token.actions.githubusercontent.com:sub",
      ])
      && !strcontains(jsonencode(s.Condition), "refs/heads")
      && !strcontains(jsonencode(s.Condition), "workflow_dispatch")
    ])
    error_message = "The OIDC trust's only conditions are StringEquals on token.actions.githubusercontent.com:aud and :sub (no StringLike, no ref, event or workflow check); got ${aws_iam_role.github_images.assume_role_policy}."
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

  # §5.6: the workflow uploads releases/<sha>.json and the files beside it.
  assert {
    condition = anytrue(flatten([
      for s in jsondecode(aws_iam_role_policy.github_images.policy).Statement : [
        for a in flatten([s.Action]) : [
          for r in flatten([s.Resource]) :
          s.Effect == "Allow" && a == "s3:PutObject" && contains([
            "${aws_s3_bucket.releases.arn}/releases/*",
            "arn:aws:s3:::${aws_s3_bucket.releases.bucket}/releases/*",
            "arn:aws:s3:::loom-test-releases/releases/*",
          ], r)
        ]
      ]
    ]))
    error_message = "The OIDC role may s3:PutObject on the releases bucket's releases/* (arn:aws:s3:::loom-test-releases/releases/*, §5.6); got ${aws_iam_role_policy.github_images.policy}."
  }

  assert {
    condition = alltrue([
      for s in jsondecode(aws_iam_role_policy.github_images.policy).Statement :
      s.Effect == "Allow" && !can(s.NotAction) && !can(s.NotResource)
    ])
    error_message = "OIDC policy statements are plain Allow statements (no NotAction/NotResource)."
  }

  # Allowlist ("ECR push only", §5.2, plus §5.6): every (action, resource) pair is an ECR
  # push action on repository/loom/*, GetAuthorizationToken on *, or the upload to the
  # releases bucket's releases/*.
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
          a == "s3:PutObject" ? contains([
            "${aws_s3_bucket.releases.arn}/releases/*",
            "arn:aws:s3:::${aws_s3_bucket.releases.bucket}/releases/*",
          ], r) :
          false
        ]
      ]
    ]))
    error_message = "The OIDC role may only push images and upload releases: ecr:GetAuthorizationToken on *; PutImage, InitiateLayerUpload, UploadLayerPart, CompleteLayerUpload, BatchCheckLayerAvailability (BatchGetImage) on repository/loom/*; s3:PutObject only on the releases bucket's releases/*; no other action. Got ${aws_iam_role_policy.github_images.policy}."
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

# §5.6 replaces "on the domain identity only": in the SES sandbox SES also authorises the
# recipient's identity and the configuration set, so the grant is on identity/* and the
# configuration set, and the condition keeps the sender to no-reply@<domain>.
run "ses_smtp_user_sends_only_as_no_reply" {
  command = plan

  assert {
    condition     = aws_iam_user.ses_smtp.name == "loom-ses-smtp"
    error_message = "The SES SMTP user is loom-ses-smtp."
  }

  assert {
    condition = (
      contains([aws_iam_user.ses_smtp.name, aws_iam_user.ses_smtp.id], aws_iam_user_policy.ses_smtp.user)
    )
    error_message = "aws_iam_user_policy.ses_smtp belongs to aws_iam_user.ses_smtp."
  }

  assert {
    condition = (
      length(jsondecode(aws_iam_user_policy.ses_smtp.policy).Statement) >= 1
      && alltrue([
        for s in jsondecode(aws_iam_user_policy.ses_smtp.policy).Statement :
        s.Effect == "Allow" && can(s.Action) && can(s.Resource) && !can(s.NotAction) && !can(s.NotResource)
      ])
      && toset(flatten([for s in jsondecode(aws_iam_user_policy.ses_smtp.policy).Statement : flatten([s.Action])])) == toset(["ses:SendRawEmail"])
    )
    error_message = "loom-ses-smtp may only ses:SendRawEmail (plain Allow statements); got ${aws_iam_user_policy.ses_smtp.policy}."
  }

  # identity/* (any identity: the recipient's in the sandbox) and the stack's
  # configuration set, both required, nothing else.
  assert {
    condition = (
      contains(flatten([for s in jsondecode(aws_iam_user_policy.ses_smtp.policy).Statement : flatten([s.Resource])]), "arn:aws:ses:us-east-1:123456789012:identity/*")
      && anytrue([
        for r in flatten([for s in jsondecode(aws_iam_user_policy.ses_smtp.policy).Statement : flatten([s.Resource])]) :
        can(regex("^arn:aws:ses:us-east-1:123456789012:configuration-set/[^*?/]+$", r))
      ])
      && alltrue([
        for r in flatten([for s in jsondecode(aws_iam_user_policy.ses_smtp.policy).Statement : flatten([s.Resource])]) :
        r == "arn:aws:ses:us-east-1:123456789012:identity/*" || can(regex("^arn:aws:ses:us-east-1:123456789012:configuration-set/[^*?/]+$", r))
      ])
    )
    error_message = "loom-ses-smtp's resources are arn:aws:ses:us-east-1:123456789012:identity/* and the configuration set's ARN, nothing else; got ${aws_iam_user_policy.ses_smtp.policy}."
  }

  assert {
    condition = alltrue([
      for s in jsondecode(aws_iam_user_policy.ses_smtp.policy).Statement :
      toset(flatten([try(s.Condition.StringEquals["ses:FromAddress"], [])])) == toset(["no-reply@example.test"])
    ])
    error_message = "Every SES statement has Condition StringEquals ses:FromAddress = no-reply@<domain> (no-reply@example.test); got ${aws_iam_user_policy.ses_smtp.policy}."
  }
}

run "ses_smtp_from_address_follows_the_domain" {
  command = plan

  variables {
    domain = "other-placeholder.test"
  }

  assert {
    condition = alltrue([
      for s in jsondecode(aws_iam_user_policy.ses_smtp.policy).Statement :
      toset(flatten([try(s.Condition.StringEquals["ses:FromAddress"], [])])) == toset(["no-reply@other-placeholder.test"])
    ])
    error_message = "ses:FromAddress is no-reply@var.domain; got ${aws_iam_user_policy.ses_smtp.policy}."
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
