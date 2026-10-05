data "aws_partition" "current" {}

data "aws_caller_identity" "current" {}

locals {
  partition = data.aws_partition.current.partition
  ec2_arn   = "arn:${local.partition}:ec2:${var.region}:${data.aws_caller_identity.current.account_id}"
}

# --- Results bucket ----------------------------------------------------------

resource "aws_s3_bucket" "bench" {
  bucket_prefix = "${var.name_prefix}-"
}

resource "aws_s3_bucket_public_access_block" "bench" {
  bucket                  = aws_s3_bucket.bench.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_ownership_controls" "bench" {
  bucket = aws_s3_bucket.bench.id

  rule {
    object_ownership = "BucketOwnerEnforced"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "bench" {
  bucket = aws_s3_bucket.bench.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_lifecycle_configuration" "bench" {
  bucket = aws_s3_bucket.bench.id

  rule {
    id     = "expire-runs"
    status = "Enabled"

    filter {
      prefix = "runs/"
    }

    expiration {
      days = var.runs_expiry_days
    }

    abort_incomplete_multipart_upload {
      days_after_initiation = 7
    }
  }

  rule {
    id     = "expire-ssm-output"
    status = "Enabled"

    filter {
      prefix = "ssm/"
    }

    expiration {
      days = var.runs_expiry_days
    }
  }
}

data "aws_iam_policy_document" "bucket" {
  statement {
    sid       = "DenyInsecureTransport"
    effect    = "Deny"
    actions   = ["s3:*"]
    resources = [aws_s3_bucket.bench.arn, "${aws_s3_bucket.bench.arn}/*"]

    principals {
      type        = "*"
      identifiers = ["*"]
    }

    condition {
      test     = "Bool"
      variable = "aws:SecureTransport"
      values   = ["false"]
    }
  }
}

resource "aws_s3_bucket_policy" "bench" {
  bucket     = aws_s3_bucket.bench.id
  policy     = data.aws_iam_policy_document.bucket.json
  depends_on = [aws_s3_bucket_public_access_block.bench]
}

# --- Network: default VPC, no ingress --------------------------------------------

data "aws_vpc" "default" {
  default = true
}

data "aws_subnets" "default" {
  filter {
    name   = "vpc-id"
    values = [data.aws_vpc.default.id]
  }

  filter {
    name   = "default-for-az"
    values = ["true"]
  }
}

resource "aws_security_group" "instance" {
  name_prefix = "${var.name_prefix}-instance-"
  description = "Loom bench GPU hosts: no ingress (managed over SSM); egress for images and weights"
  vpc_id      = data.aws_vpc.default.id

  egress {
    description = "Docker Hub, Hugging Face, PyPI, AWS APIs"
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  lifecycle {
    create_before_destroy = true
  }
}

# --- GPU host role ---------------------------------------------------------------

data "aws_secretsmanager_secret" "hf_token" {
  name = var.hf_token_secret_name
}

data "aws_iam_policy_document" "ec2_assume" {
  statement {
    actions = ["sts:AssumeRole"]

    principals {
      type        = "Service"
      identifiers = ["ec2.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "instance" {
  name               = "${var.name_prefix}-instance"
  assume_role_policy = data.aws_iam_policy_document.ec2_assume.json
}

resource "aws_iam_role_policy_attachment" "instance_ssm" {
  role       = aws_iam_role.instance.name
  policy_arn = "arn:${local.partition}:iam::aws:policy/AmazonSSMManagedInstanceCore"
}

data "aws_iam_policy_document" "instance" {
  statement {
    sid       = "BenchObjects"
    actions   = ["s3:GetObject", "s3:PutObject"]
    resources = ["${aws_s3_bucket.bench.arn}/*"]
  }

  statement {
    sid       = "SsmCommandOutput"
    actions   = ["s3:GetEncryptionConfiguration"]
    resources = [aws_s3_bucket.bench.arn]
  }

  statement {
    sid       = "HuggingFaceToken"
    actions   = ["secretsmanager:GetSecretValue"]
    resources = [data.aws_secretsmanager_secret.hf_token.arn]
  }
}

resource "aws_iam_role_policy" "instance" {
  name   = "bench"
  role   = aws_iam_role.instance.id
  policy = data.aws_iam_policy_document.instance.json
}

resource "aws_iam_instance_profile" "instance" {
  name = "${var.name_prefix}-instance"
  role = aws_iam_role.instance.name
}
