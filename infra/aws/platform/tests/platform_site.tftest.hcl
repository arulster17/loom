# A35, Terraform half (docs/phase1-m3-spec.md §13, §6.11, §5.2 site/iam): the variables
# `domain` and `brand_name` have no defaults; they fill the SSM parameter
# /loom/platform/site (String, JSON {"domain", "brand_name"}); only the instance role may
# read it. (The scan for domain literals and loom-deploy's rendering are pytest, another
# area.)
#
# Addresses this file expects (the spec names none):
#   aws_ssm_parameter.site                         the site parameter (value or insecure_value)
#   aws_iam_role_policy.instance                   the instance role's inline policy
#   aws_iam_role_policy.serving_reaper, aws_iam_role_policy.github_images,
#   aws_iam_user_policy.ses_smtp                   the stack's other identities' policies
# A run cannot omit a variable that has no default (terraform test stops with "No value
# for required variable" before any assertion), so "no default" is read from
# variables.tf through the source scan.

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

run "domain_and_brand_name_have_no_defaults" {
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
}

run "site_parameter_holds_domain_and_brand_name" {
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
    )) == jsonencode({ brand_name = "Example Brand", domain = "example.test" })
    error_message = "The site parameter is the JSON {\"domain\": var.domain, \"brand_name\": var.brand_name} and nothing else."
  }
}

run "site_parameter_follows_the_variables" {
  command = plan

  variables {
    domain     = "other-placeholder.test"
    brand_name = "Other Name"
  }

  assert {
    condition = jsonencode(jsondecode(
      aws_ssm_parameter.site.insecure_value == null || aws_ssm_parameter.site.insecure_value == ""
      ? nonsensitive(aws_ssm_parameter.site.value)
      : aws_ssm_parameter.site.insecure_value
    )) == jsonencode({ brand_name = "Other Name", domain = "other-placeholder.test" })
    error_message = "The site parameter changes with var.domain and var.brand_name."
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
