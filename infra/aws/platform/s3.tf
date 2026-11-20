# Two private buckets with the same settings: public access blocked, SSE, a TLS-only
# policy, versioning off (docs/phase1-m3-spec.md §5.2 s3, §5.6):
#   leases    serving/leases/<pod id> records (M4; unused in M3)
#   releases  releases/<sha>.json and releases/<sha>/ (§6.12), named by var.releases_bucket

locals {
  # S3 names are global; the lease bucket's carries the account id.
  leases_bucket_name = "loom-platform-leases-${local.account_id}"
}

resource "aws_s3_bucket" "leases" {
  bucket = local.leases_bucket_name
}

resource "aws_s3_bucket_public_access_block" "leases" {
  bucket = aws_s3_bucket.leases.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_server_side_encryption_configuration" "leases" {
  bucket = aws_s3_bucket.leases.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_policy" "leases" {
  bucket = aws_s3_bucket.leases.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Sid       = "DenyInsecureTransport"
      Effect    = "Deny"
      Principal = "*"
      Action    = "s3:*"
      Resource  = [aws_s3_bucket.leases.arn, "${aws_s3_bucket.leases.arn}/*"]
      Condition = { Bool = { "aws:SecureTransport" = "false" } }
    }]
  })

  depends_on = [aws_s3_bucket_public_access_block.leases]
}

resource "aws_s3_bucket" "releases" {
  bucket = var.releases_bucket
}

resource "aws_s3_bucket_public_access_block" "releases" {
  bucket = aws_s3_bucket.releases.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_server_side_encryption_configuration" "releases" {
  bucket = aws_s3_bucket.releases.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_policy" "releases" {
  bucket = aws_s3_bucket.releases.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Sid       = "DenyInsecureTransport"
      Effect    = "Deny"
      Principal = "*"
      Action    = "s3:*"
      Resource  = [aws_s3_bucket.releases.arn, "${aws_s3_bucket.releases.arn}/*"]
      Condition = { Bool = { "aws:SecureTransport" = "false" } }
    }]
  })

  depends_on = [aws_s3_bucket_public_access_block.releases]
}
