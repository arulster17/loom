# A35, Terraform half (docs/phase1-m3-spec.md §13, §6.11, §6.12, §5.2 site/iam, §5.6): the
# variables `domain`, `brand_name` and `releases_bucket` have no defaults; they fill the
# SSM parameter /loom/platform/site (String), whose JSON has exactly the 7 keys of §6.12:
# domain, brand_name, ops_email, releases_bucket (from the variables, the bucket name may
# come from the bucket), db_host, db_port (a JSON number) and db_master_secret_arn (from
# the RDS instance's address, port and master_user_secret[0].secret_arn); only the
# instance role may read it. (The scan for domain literals and loom-deploy's rendering are
# pytest, another area.)
#
# Addresses this file expects (the spec names none):
#   aws_ssm_parameter.site                         the site parameter (value or insecure_value)
#   aws_db_instance.platform                       the RDS instance (see platform_rds)
#   aws_s3_bucket.releases                         the releases bucket (see platform_releases_bucket)
#   aws_iam_role_policy.instance                   the instance role's inline policy
#   aws_iam_role_policy.serving_reaper, aws_iam_role_policy.github_images,
#   aws_iam_user_policy.ses_smtp                   the stack's other identities' policies
# A run cannot omit a variable that has no default (terraform test stops with "No value
# for required variable" before any assertion), so "no default" is read from
# variables.tf through the source scan.
#
# The RDS instance and the releases bucket get distinct mocked values here, so the
# expected JSON is built from them: the address, port and secret come from the instance,
# whatever the configuration sets.

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
  target          = aws_db_instance.platform
  override_during = plan
  values = {
    address  = "loom-site-db.csite.us-east-1.rds.amazonaws.com"
    endpoint = "loom-site-db.csite.us-east-1.rds.amazonaws.com:6543"
    port     = 6543
    master_user_secret = [{
      kms_key_id    = "arn:aws:kms:us-east-1:123456789012:key/mock-site"
      secret_arn    = "arn:aws:secretsmanager:us-east-1:123456789012:secret:rds!db-site-XyZaBc"
      secret_status = "active"
    }]
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

run "domain_brand_name_and_releases_bucket_have_no_defaults" {
  command = plan

  module {
    source = "./tests/fixtures/source_scan"
  }

  assert {
    condition     = output.variable_declared["domain"] == 1 && output.variable_declared["brand_name"] == 1
    error_message = "variables.tf declares variable \"domain\" and variable \"brand_name\" once each; found ${jsonencode(output.variable_declared)}."
  }

  assert {
    condition     = !output.variable_has_default["domain"] && !output.variable_has_default["brand_name"]
    error_message = "domain and brand_name must have no default (terraform plan asks for them, §6.11); defaults found: ${jsonencode(output.variable_has_default)}."
  }

  assert {
    condition     = output.variable_declared["releases_bucket"] == 1 && !output.variable_has_default["releases_bucket"]
    error_message = "variables.tf declares variable \"releases_bucket\" once, with no default (S3 names are global, §5.6); found ${jsonencode(output.variable_declared)}, defaults ${jsonencode(output.variable_has_default)}."
  }
}

run "site_parameter_holds_the_keys_of_6_12" {
  command = plan

  assert {
    condition     = aws_ssm_parameter.site.name == "/loom/platform/site" && aws_ssm_parameter.site.type == "String"
    error_message = "The site parameter is /loom/platform/site, type String."
  }

  assert {
    condition     = coalesce(aws_ssm_parameter.site.tier, "Standard") == "Standard"
    error_message = "The site parameter is Standard tier (free)."
  }

  assert {
    condition = jsonencode(jsondecode(
      aws_ssm_parameter.site.insecure_value == null || aws_ssm_parameter.site.insecure_value == ""
      ? nonsensitive(aws_ssm_parameter.site.value)
      : aws_ssm_parameter.site.insecure_value
      )) == jsonencode({
      brand_name           = "Example Brand"
      db_host              = aws_db_instance.platform.address
      db_master_secret_arn = aws_db_instance.platform.master_user_secret[0].secret_arn
      db_port              = aws_db_instance.platform.port
      domain               = "example.test"
      ops_email            = "ops@example.test"
      releases_bucket      = "loom-test-releases"
    })
    error_message = "The site parameter is exactly the JSON of §6.12: {\"domain\", \"brand_name\", \"ops_email\", \"releases_bucket\" (the variables), \"db_host\" (aws_db_instance.platform.address), \"db_port\" (its port, a JSON number), \"db_master_secret_arn\" (master_user_secret[0].secret_arn)}, nothing else. Got ${aws_ssm_parameter.site.insecure_value == null || aws_ssm_parameter.site.insecure_value == "" ? nonsensitive(aws_ssm_parameter.site.value) : aws_ssm_parameter.site.insecure_value}."
  }

  # The mocked instance's values, so the comparison above is not between two defaults.
  assert {
    condition = (
      aws_db_instance.platform.address == "loom-site-db.csite.us-east-1.rds.amazonaws.com"
      && aws_db_instance.platform.master_user_secret[0].secret_arn == "arn:aws:secretsmanager:us-east-1:123456789012:secret:rds!db-site-XyZaBc"
    )
    error_message = "Test setup: the RDS instance's address and master secret come from this file's override."
  }
}

run "site_parameter_follows_the_variables" {
  command = plan

  variables {
    domain          = "other-placeholder.test"
    brand_name      = "Other Name"
    ops_email       = "other-ops@other-placeholder.test"
    releases_bucket = "other-test-releases"
  }

  override_resource {
    target          = aws_s3_bucket.releases
    override_during = plan
    values = {
      id  = "other-test-releases"
      arn = "arn:aws:s3:::other-test-releases"
    }
  }

  assert {
    condition = jsonencode(jsondecode(
      aws_ssm_parameter.site.insecure_value == null || aws_ssm_parameter.site.insecure_value == ""
      ? nonsensitive(aws_ssm_parameter.site.value)
      : aws_ssm_parameter.site.insecure_value
      )) == jsonencode({
      brand_name           = "Other Name"
      db_host              = aws_db_instance.platform.address
      db_master_secret_arn = aws_db_instance.platform.master_user_secret[0].secret_arn
      db_port              = aws_db_instance.platform.port
      domain               = "other-placeholder.test"
      ops_email            = "other-ops@other-placeholder.test"
      releases_bucket      = "other-test-releases"
    })
    error_message = "The site parameter changes with var.domain, var.brand_name, var.ops_email and var.releases_bucket."
  }
}

run "only_the_instance_role_reads_the_site_parameter" {
  command = plan

  assert {
    condition = anytrue(flatten([
      for s in jsondecode(aws_iam_role_policy.instance.policy).Statement : [
        for a in flatten([s.Action]) : [
          for r in flatten([s.Resource]) :
          s.Effect == "Allow" && a == "ssm:GetParameter" && r == aws_ssm_parameter.site.arn
        ]
      ]
    ]))
    error_message = "The instance role may ssm:GetParameter on the site parameter."
  }

  # An Allow statement without Action (NotAction) fails here: s.Action does not exist.
  assert {
    condition = alltrue([
      for p in [
        aws_iam_role_policy.serving_reaper.policy,
        aws_iam_role_policy.github_images.policy,
        aws_iam_user_policy.ses_smtp.policy,
      ] :
      alltrue([
        for a in flatten([for s in jsondecode(p).Statement : flatten([s.Action]) if s.Effect == "Allow"]) :
        !startswith(lower(a), "ssm:") && !can(regex("^[^:]*[*?]", a))
      ])
    ])
    error_message = "No other role or user of the stack may read SSM parameters (serving reaper, OIDC role, SES SMTP user): no ssm: action and no wildcard service prefix (*, *:*)."
  }
}
