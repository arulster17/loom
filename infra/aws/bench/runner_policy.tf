# What the identity running `bench run` (a person or CI) needs, and no more.
# Attach the managed policy `runner_policy_arn` to that user or role.

data "aws_iam_policy_document" "runner" {
  statement {
    sid       = "LaunchManagedGpuInstances"
    actions   = ["ec2:RunInstances"]
    resources = ["${local.ec2_arn}:instance/*"]

    condition {
      test     = "StringEquals"
      variable = "aws:RequestTag/loom:managed"
      values   = ["true"]
    }

    condition {
      test     = "Null"
      variable = "aws:RequestTag/loom:ttl"
      values   = ["false"]
    }

    condition {
      test     = "Null"
      variable = "aws:RequestTag/loom:experiment"
      values   = ["false"]
    }

    condition {
      test     = "Null"
      variable = "aws:RequestTag/loom:owner"
      values   = ["false"]
    }

    condition {
      test     = "StringEquals"
      variable = "ec2:InstanceType"
      values   = var.allowed_instance_types
    }

    condition {
      test     = "StringEquals"
      variable = "ec2:MetadataHttpTokens"
      values   = ["required"]
    }

    condition {
      test     = "NumericLessThanEquals"
      variable = "ec2:MetadataHttpPutResponseHopLimit"
      values   = ["1"]
    }

    condition {
      test     = "ArnEquals"
      variable = "ec2:InstanceProfile"
      values   = [aws_iam_instance_profile.instance.arn]
    }
  }

  statement {
    sid       = "LaunchManagedVolumes"
    actions   = ["ec2:RunInstances"]
    resources = ["${local.ec2_arn}:volume/*"]

    condition {
      test     = "StringEquals"
      variable = "aws:RequestTag/loom:managed"
      values   = ["true"]
    }
  }

  statement {
    sid       = "LaunchFromAmazonImages"
    actions   = ["ec2:RunInstances"]
    resources = ["arn:${local.partition}:ec2:${var.region}::image/*"]

    condition {
      test     = "StringEquals"
      variable = "ec2:Owner"
      values   = ["amazon"]
    }
  }

  statement {
    sid     = "LaunchIntoBenchNetwork"
    actions = ["ec2:RunInstances"]
    resources = [
      "${local.ec2_arn}:subnet/*",
      "${local.ec2_arn}:network-interface/*",
      "${local.ec2_arn}:security-group/${aws_security_group.instance.id}",
      "${local.ec2_arn}:spot-instances-request/*",
    ]
  }

  statement {
    sid       = "TagOnLaunchOnly"
    actions   = ["ec2:CreateTags"]
    resources = ["${local.ec2_arn}:instance/*", "${local.ec2_arn}:volume/*"]

    condition {
      test     = "StringEquals"
      variable = "ec2:CreateAction"
      values   = ["RunInstances"]
    }
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

  statement {
    sid = "Describe"
    actions = [
      "ec2:DescribeImages",
      "ec2:DescribeInstances",
      "ec2:DescribeSpotPriceHistory",
      "ec2:DescribeSubnets",
      "ec2:DescribeVolumes",
    ]
    resources = ["*"]
  }

  statement {
    sid       = "PassInstanceRole"
    actions   = ["iam:PassRole"]
    resources = [aws_iam_role.instance.arn]

    condition {
      test     = "StringEquals"
      variable = "iam:PassedToService"
      values   = ["ec2.amazonaws.com"]
    }
  }

  statement {
    sid       = "CreateSpotServiceLinkedRole"
    actions   = ["iam:CreateServiceLinkedRole"]
    resources = ["arn:${local.partition}:iam::*:role/aws-service-role/spot.amazonaws.com/AWSServiceRoleForEC2Spot"]

    condition {
      test     = "StringEquals"
      variable = "iam:AWSServiceName"
      values   = ["spot.amazonaws.com"]
    }
  }

  statement {
    sid       = "SendCommandToManagedHosts"
    actions   = ["ssm:SendCommand"]
    resources = ["${local.ec2_arn}:instance/*"]

    condition {
      test     = "StringEquals"
      variable = "ssm:resourceTag/loom:managed"
      values   = ["true"]
    }
  }

  statement {
    sid       = "SendRunShellScript"
    actions   = ["ssm:SendCommand"]
    resources = ["arn:${local.partition}:ssm:${var.region}::document/AWS-RunShellScript"]
  }

  statement {
    sid = "TrackCommands"
    actions = [
      "ssm:CancelCommand",
      "ssm:DescribeInstanceInformation",
      "ssm:GetCommandInvocation",
    ]
    resources = ["*"]
  }

  statement {
    sid       = "ReadDlamiParameter"
    actions   = ["ssm:GetParameter"]
    resources = ["arn:${local.partition}:ssm:${var.region}::parameter/aws/service/deeplearning/*"]
  }

  statement {
    sid       = "BenchObjects"
    actions   = ["s3:GetObject", "s3:PutObject"]
    resources = ["${aws_s3_bucket.bench.arn}/*"]
  }

  statement {
    sid       = "BenchBucketList"
    actions   = ["s3:ListBucket"]
    resources = [aws_s3_bucket.bench.arn]
  }
}

resource "aws_iam_policy" "runner" {
  name        = "${var.name_prefix}-runner"
  description = "Run Loom bench experiments: launch, drive and terminate loom:managed GPU hosts"
  policy      = data.aws_iam_policy_document.runner.json
}
