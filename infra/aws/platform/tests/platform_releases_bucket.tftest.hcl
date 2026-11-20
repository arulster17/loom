# §5.6 "Releases bucket" and §5.2 s3 (docs/phase1-m3-spec.md; A6's releases/* grants are
# in platform_iam, the bucket's name in the site parameter in platform_site): a second
# bucket, named by var.releases_bucket (no default, checked in platform_site), with the
# lease bucket's settings: private (a public access block with all four settings on, and
# a bucket policy that allows nothing to everyone), SSE, a TLS-only bucket policy,
# versioning off; output `releases_bucket`. It holds releases/<sha>.json and
# releases/<sha>/ (§6.12).
#
# Addresses this file expects (the spec names none):
#   aws_s3_bucket.releases                                       the bucket (bucket = var.releases_bucket)
#   aws_s3_bucket_public_access_block.releases                   its public access block
#   aws_s3_bucket_server_side_encryption_configuration.releases  its default encryption
#   aws_s3_bucket_policy.releases                                its bucket policy (jsonencode, §5.1)
#   output "releases_bucket"                                     the bucket's name
# Versioning off: no aws_s3_bucket_versioning naming "Enabled" refers to
# aws_s3_bucket.releases (read from the source; an aws_s3_bucket_versioning with status
# "Suspended" or "Disabled", or none at all, is off).
#
# The mock gives every bucket the lease bucket's id and ARN, so this file overrides the
# releases bucket's: id loom-test-releases, ARN arn:aws:s3:::loom-test-releases.

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
  target          = aws_s3_bucket.releases
  override_during = plan
  values = {
    id  = "loom-test-releases"
    arn = "arn:aws:s3:::loom-test-releases"
  }
}

run "releases_bucket_is_named_by_the_variable" {
  command = plan

  assert {
    condition     = aws_s3_bucket.releases.bucket == "loom-test-releases"
    error_message = "The releases bucket is named var.releases_bucket; got ${jsonencode(aws_s3_bucket.releases.bucket)}."
  }

  assert {
    condition     = output.releases_bucket == "loom-test-releases"
    error_message = "Output releases_bucket is the releases bucket's name; got ${jsonencode(output.releases_bucket)}."
  }
}

run "releases_bucket_follows_the_variable" {
  command = plan

  variables {
    releases_bucket = "other-test-releases"
  }

  assert {
    condition     = aws_s3_bucket.releases.bucket == "other-test-releases"
    error_message = "The releases bucket's name follows var.releases_bucket; got ${jsonencode(aws_s3_bucket.releases.bucket)}."
  }
}

run "releases_bucket_is_private_and_encrypted" {
  command = plan

  assert {
    condition = (
      contains([aws_s3_bucket.releases.id, aws_s3_bucket.releases.bucket], aws_s3_bucket_public_access_block.releases.bucket)
      && aws_s3_bucket_public_access_block.releases.block_public_acls == true
      && aws_s3_bucket_public_access_block.releases.block_public_policy == true
      && aws_s3_bucket_public_access_block.releases.ignore_public_acls == true
      && aws_s3_bucket_public_access_block.releases.restrict_public_buckets == true
    )
    error_message = "The releases bucket has a public access block with block_public_acls, block_public_policy, ignore_public_acls and restrict_public_buckets all true."
  }

  assert {
    condition = (
      contains([aws_s3_bucket.releases.id, aws_s3_bucket.releases.bucket], aws_s3_bucket_server_side_encryption_configuration.releases.bucket)
      && length(flatten([
        for r in aws_s3_bucket_server_side_encryption_configuration.releases.rule : [
          for d in r.apply_server_side_encryption_by_default : d.sse_algorithm
        ]
      ])) > 0
      && alltrue(flatten([
        for r in aws_s3_bucket_server_side_encryption_configuration.releases.rule : [
          for d in r.apply_server_side_encryption_by_default :
          contains(["AES256", "aws:kms", "aws:kms:dsse"], d.sse_algorithm)
        ]
      ]))
    )
    error_message = "The releases bucket has default server-side encryption (apply_server_side_encryption_by_default with AES256 or aws:kms)."
  }

  # Private: no statement allows anything to everyone (Principal "*" or {AWS = "*"}).
  assert {
    condition = alltrue([
      for s in jsondecode(aws_s3_bucket_policy.releases.policy).Statement :
      s.Effect != "Allow" || !contains(flatten([try(s.Principal.AWS, s.Principal, [])]), "*")
    ])
    error_message = "The releases bucket's policy allows nothing to the public (no Allow with Principal *); got ${aws_s3_bucket_policy.releases.policy}."
  }
}

run "releases_bucket_is_tls_only" {
  command = plan

  assert {
    condition     = contains([aws_s3_bucket.releases.id, aws_s3_bucket.releases.bucket], aws_s3_bucket_policy.releases.bucket)
    error_message = "aws_s3_bucket_policy.releases is on the releases bucket."
  }

  # A Deny of s3:* to everyone on the bucket and its objects when aws:SecureTransport is
  # false.
  assert {
    condition = anytrue([
      for s in jsondecode(aws_s3_bucket_policy.releases.policy).Statement :
      s.Effect == "Deny"
      && contains(flatten([try(s.Principal.AWS, s.Principal, [])]), "*")
      && contains(flatten([s.Action]), "s3:*")
      && length(setsubtract(toset(["arn:aws:s3:::loom-test-releases", "arn:aws:s3:::loom-test-releases/*"]), toset(flatten([s.Resource])))) == 0
      && contains([for v in flatten([try(s.Condition.Bool["aws:SecureTransport"], [])]) : tostring(v)], "false")
    ])
    error_message = "The releases bucket's policy denies s3:* to Principal * on arn:aws:s3:::loom-test-releases and arn:aws:s3:::loom-test-releases/* when Bool aws:SecureTransport = false; got ${aws_s3_bucket_policy.releases.policy}."
  }
}

run "releases_bucket_versioning_is_off" {
  command = plan

  module {
    source = "./tests/fixtures/source_scan"
  }

  assert {
    condition     = !contains(output.s3_versioning_enabled_buckets, "releases")
    error_message = "Versioning stays off on the releases bucket (like the lease bucket's): no aws_s3_bucket_versioning with status Enabled refers to aws_s3_bucket.releases; versioned buckets found: ${jsonencode(output.s3_versioning_enabled_buckets)}."
  }

  assert {
    condition     = contains(output.resource_types, "aws_s3_bucket")
    error_message = "The stack declares its S3 buckets (aws_s3_bucket)."
  }
}
