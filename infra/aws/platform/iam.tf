# Identities of the platform stack (docs/phase1-m3-spec.md §5.2 iam, §5.6): the instance
# role, the GitHub OIDC role for the image workflow, the serving reaper's role and the SES
# SMTP user. Policies are jsonencode() locals (§5.1) so tests can read their exact content.
# §5.2 plus §5.6 is the whole allowlist; the optional grants of §5.6 are left out except
# those marked below.

# ---- the instance role -------------------------------------------------------------

resource "aws_iam_role" "instance" {
  name = "loom-platform-instance"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "ec2.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })
}

resource "aws_iam_role_policy_attachment" "instance_ssm" {
  role       = aws_iam_role.instance.name
  policy_arn = "arn:${local.partition}:iam::aws:policy/AmazonSSMManagedInstanceCore"
}

resource "aws_iam_instance_profile" "instance" {
  name = "loom-platform-instance"
  role = aws_iam_role.instance.name
}

locals {
  instance_policy = {
    Version = "2012-10-17"
    Statement = [
      {
        Sid      = "PutMetricsInTheLoomNamespaceOnly"
        Effect   = "Allow"
        Action   = ["cloudwatch:PutMetricData"]
        Resource = "*"
        Condition = {
          StringEquals = { "cloudwatch:namespace" = "Loom/Platform" }
        }
      },
      {
        Sid    = "Logs"
        Effect = "Allow"
        Action = [
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:DescribeLogStreams",
          "logs:PutLogEvents",
        ]
        Resource = [local.log_group_arn]
      },
      {
        # `secrets init` and `ses-smtp-password` store what Loom makes and check first.
        Sid    = "PlatformSecrets"
        Effect = "Allow"
        Action = [
          "secretsmanager:DescribeSecret",
          "secretsmanager:GetSecretValue",
          "secretsmanager:PutSecretValue",
        ]
        Resource = ["${local.secrets_arn_prefix}/*"]
      },
      {
        Sid      = "ReadTheRdsMasterSecret"
        Effect   = "Allow"
        Action   = ["secretsmanager:GetSecretValue"]
        Resource = [aws_db_instance.platform.master_user_secret[0].secret_arn]
      },
      {
        # The loom/platform/* grants would otherwise let the box read the serving RunPod
        # key: only the serving reaper reads it (§5.6).
        Sid      = "NeverTheServingRunpodKey"
        Effect   = "Deny"
        Action   = ["secretsmanager:*"]
        Resource = [aws_secretsmanager_secret.platform["serving-runpod-key"].arn]
      },
      {
        Sid      = "EcrLogin"
        Effect   = "Allow"
        Action   = ["ecr:GetAuthorizationToken"]
        Resource = "*"
      },
      {
        Sid    = "EcrPull"
        Effect = "Allow"
        Action = [
          "ecr:BatchCheckLayerAvailability",
          "ecr:BatchGetImage",
          "ecr:GetDownloadUrlForLayer",
        ]
        Resource = [local.ecr_repository_arn]
      },
      {
        # The lease records (M4); the delete is the pool manager releasing a lease.
        Sid    = "LeaseRecords"
        Effect = "Allow"
        Action = [
          "s3:DeleteObject",
          "s3:GetObject",
          "s3:PutObject",
        ]
        Resource = ["${aws_s3_bucket.leases.arn}/serving/leases/*"]
      },
      {
        Sid      = "FetchReleases"
        Effect   = "Allow"
        Action   = ["s3:GetObject"]
        Resource = ["${aws_s3_bucket.releases.arn}/releases/*"]
      },
      {
        Sid      = "ReadTheSiteParameter"
        Effect   = "Allow"
        Action   = ["ssm:GetParameter", "ssm:GetParameters"]
        Resource = [aws_ssm_parameter.site.arn]
      },
    ]
  }
}

resource "aws_iam_role_policy" "instance" {
  name   = "platform"
  role   = aws_iam_role.instance.id
  policy = jsonencode(local.instance_policy)
}

# ---- the GitHub OIDC role (image workflow) -----------------------------------------

# An account holds one provider per URL: create it, or look up the existing one.
resource "aws_iam_openid_connect_provider" "github" {
  count = var.create_github_oidc_provider ? 1 : 0

  url            = "https://token.actions.githubusercontent.com"
  client_id_list = ["sts.amazonaws.com"]
}

data "aws_iam_openid_connect_provider" "github" {
  count = var.create_github_oidc_provider ? 0 : 1

  url = "https://token.actions.githubusercontent.com"
}

locals {
  github_oidc_provider_arn = var.create_github_oidc_provider ? aws_iam_openid_connect_provider.github[0].arn : data.aws_iam_openid_connect_provider.github[0].arn

  # IAM cannot test a branch and an event together, so the trust names the GitHub
  # environment `platform-images`: its deployment-branch rule (main only) is set in GitHub,
  # and images.yml, the only workflow naming it, runs on workflow_dispatch only (§5.6).
  github_subject = "repo:arulster17/loom:environment:platform-images"
}

resource "aws_iam_role" "github_images" {
  name = "loom-platform-github-images"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Federated = local.github_oidc_provider_arn }
      Action    = "sts:AssumeRoleWithWebIdentity"
      Condition = {
        StringEquals = {
          "token.actions.githubusercontent.com:aud" = "sts.amazonaws.com"
          "token.actions.githubusercontent.com:sub" = local.github_subject
        }
      }
    }]
  })
}

resource "aws_iam_role_policy" "github_images" {
  name = "push-images-and-releases"
  role = aws_iam_role.github_images.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid      = "EcrLogin"
        Effect   = "Allow"
        Action   = ["ecr:GetAuthorizationToken"]
        Resource = "*"
      },
      {
        Sid    = "EcrPush"
        Effect = "Allow"
        Action = [
          "ecr:BatchCheckLayerAvailability",
          "ecr:BatchGetImage",
          "ecr:CompleteLayerUpload",
          "ecr:InitiateLayerUpload",
          "ecr:PutImage",
          "ecr:UploadLayerPart",
        ]
        Resource = [local.ecr_repository_arn]
      },
      {
        Sid      = "UploadReleases"
        Effect   = "Allow"
        Action   = ["s3:PutObject"]
        Resource = ["${aws_s3_bucket.releases.arn}/releases/*"]
      },
    ]
  })
}

# ---- the serving reaper's role -----------------------------------------------------

resource "aws_iam_role" "serving_reaper" {
  name = "loom-serving-runpod-reaper"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "lambda.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })
}

resource "aws_iam_role_policy" "serving_reaper" {
  name = "serving-reaper"
  role = aws_iam_role.serving_reaper.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid      = "Logs"
        Effect   = "Allow"
        Action   = ["logs:CreateLogStream", "logs:PutLogEvents"]
        Resource = ["${aws_cloudwatch_log_group.serving_reaper.arn}:*"]
      },
      {
        Sid      = "ReadTheServingKey"
        Effect   = "Allow"
        Action   = ["secretsmanager:GetSecretValue"]
        Resource = [aws_secretsmanager_secret.platform["serving-runpod-key"].arn]
      },
      {
        Sid      = "ListTheLeaseBucket"
        Effect   = "Allow"
        Action   = ["s3:ListBucket"]
        Resource = [aws_s3_bucket.leases.arn]
      },
      {
        Sid      = "ReadLeases"
        Effect   = "Allow"
        Action   = ["s3:GetObject"]
        Resource = ["${aws_s3_bucket.leases.arn}/serving/leases/*"]
      },
    ]
  })
}

# ---- the SES SMTP user -------------------------------------------------------------

# Its access key is created by the runbook, not Terraform, so the secret stays out of the
# state. In the SES sandbox SES also authorises the recipient's identity and the
# configuration set; the condition keeps the sender to one address (§5.6).
resource "aws_iam_user" "ses_smtp" {
  name = "loom-ses-smtp"
}

resource "aws_iam_user_policy" "ses_smtp" {
  name = "send-as-no-reply"
  user = aws_iam_user.ses_smtp.name
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect = "Allow"
      Action = ["ses:SendRawEmail"]
      Resource = [
        "arn:${local.partition}:ses:${local.region}:${local.account_id}:identity/*",
        aws_sesv2_configuration_set.platform.arn,
      ]
      Condition = {
        StringEquals = { "ses:FromAddress" = "no-reply@${var.domain}" }
      }
    }]
  })
}
