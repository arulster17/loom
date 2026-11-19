# A10 (docs/phase1-m3-spec.md §13, §5.2 dns/ses): with route53_zone_id set, Terraform
# writes the records of §5.2 (A for the apex, api and hooks to the Elastic IP; CAA
# `0 issue "letsencrypt.org"`; SES DKIM CNAMEs; MAIL FROM MX and SPF TXT; DMARC TXT with
# p=quarantine); without it, the same records are an output for the user's DNS host.
#
# Addresses and shapes this file expects (the spec names none):
#   aws_route53_record.platform      every record, as one resource with count or for_each
#                                    (no instances when route53_zone_id is unset)
#   output "dns_records"             without route53_zone_id: a list of objects
#                                    { name, type, records } (records: list of strings);
#                                    names fully qualified, a trailing dot allowed
#   aws_sesv2_email_identity.domain  the domain identity (Easy DKIM, configuration set)
#   aws_sesv2_email_identity_mail_from_attributes.domain
#                                    the custom MAIL FROM subdomain (the MX record's name)
# TXT values may be written with or without surrounding quotes. Mocked values: Elastic IP
# 203.0.113.10, DKIM tokens mocktokena/b/c, region us-east-1.
#
# Notes for the implementer:
# - Easy DKIM: write an (empty) `dkim_signing_attributes {}` block on the identity. The
#   mock fills `dkim_signing_attributes.tokens` only when the block is in the
#   configuration; without it the CNAME assertions fail.
# - Use static `for_each` keys for the records (e.g. "apex-a", "dkim-0", ...), never keys
#   built from the Elastic IP or the DKIM tokens: those values are known at plan time only
#   under the mocks, and a real `terraform plan` refuses unknown for_each keys.

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

run "records_as_output_without_a_zone" {
  command = plan

  assert {
    condition     = length([for r in aws_route53_record.platform : r]) == 0
    error_message = "Without route53_zone_id Terraform writes no Route 53 records."
  }

  assert {
    condition = toset([
      for r in output.dns_records : trimsuffix(lower(r.name), ".")
      if upper(r.type) == "A"
    ]) == toset(["example.test", "api.example.test", "hooks.example.test"])
    error_message = "A records for the apex, api and hooks; got ${jsonencode(output.dns_records)}."
  }

  assert {
    condition = alltrue([
      for r in output.dns_records : toset(r.records) == toset(["203.0.113.10"])
      if upper(r.type) == "A"
    ])
    error_message = "The A records point at the Elastic IP."
  }

  assert {
    condition = anytrue([
      for r in output.dns_records :
      upper(r.type) == "CAA" && trimsuffix(lower(r.name), ".") == "example.test" && contains(r.records, "0 issue \"letsencrypt.org\"")
    ])
    error_message = "A CAA record at the apex: 0 issue \"letsencrypt.org\"."
  }

  assert {
    condition = jsonencode({
      for r in output.dns_records : trimsuffix(lower(r.name), ".") => [for v in r.records : trimsuffix(lower(v), ".")]
      if upper(r.type) == "CNAME"
      }) == jsonencode({
      "mocktokena._domainkey.example.test" = ["mocktokena.dkim.amazonses.com"]
      "mocktokenb._domainkey.example.test" = ["mocktokenb.dkim.amazonses.com"]
      "mocktokenc._domainkey.example.test" = ["mocktokenc.dkim.amazonses.com"]
    })
    error_message = "Three SES Easy DKIM CNAMEs, <token>._domainkey.<domain> -> <token>.dkim.amazonses.com."
  }

  assert {
    condition = length([
      for r in output.dns_records : r
      if upper(r.type) == "MX"
      && endswith(trimsuffix(lower(r.name), "."), ".example.test")
      && anytrue([for v in r.records : strcontains(lower(v), "feedback-smtp.us-east-1.amazonses.com")])
    ]) == 1
    error_message = "One MAIL FROM MX record on a subdomain of the domain, to feedback-smtp.us-east-1.amazonses.com."
  }

  assert {
    condition = alltrue([
      for mx in output.dns_records :
      anytrue([
        for r in output.dns_records :
        upper(r.type) == "TXT"
        && trimsuffix(lower(r.name), ".") == trimsuffix(lower(mx.name), ".")
        && anytrue([for v in r.records : startswith(trim(v, "\""), "v=spf1") && strcontains(v, "include:amazonses.com")])
      ])
      if upper(mx.type) == "MX"
    ])
    error_message = "An SPF TXT record (v=spf1 include:amazonses.com) on the MAIL FROM subdomain."
  }

  assert {
    condition = (
      contains([aws_sesv2_email_identity.domain.email_identity, aws_sesv2_email_identity.domain.id], aws_sesv2_email_identity_mail_from_attributes.domain.email_identity)
      && toset([
        for r in output.dns_records : trimsuffix(lower(r.name), ".")
        if upper(r.type) == "MX"
      ]) == toset([trimsuffix(lower(aws_sesv2_email_identity_mail_from_attributes.domain.mail_from_domain), ".")])
    )
    error_message = "The SES identity has a custom MAIL FROM domain (aws_sesv2_email_identity_mail_from_attributes) equal to the MX record's name."
  }

  assert {
    condition = anytrue([
      for r in output.dns_records :
      upper(r.type) == "TXT"
      && trimsuffix(lower(r.name), ".") == "_dmarc.example.test"
      && anytrue([for v in r.records : startswith(trim(v, "\""), "v=DMARC1") && strcontains(v, "p=quarantine")])
    ])
    error_message = "A DMARC TXT record at _dmarc.<domain> with p=quarantine."
  }

  assert {
    condition = (
      aws_sesv2_email_identity.domain.email_identity == "example.test"
      && coalesce(aws_sesv2_email_identity.domain.configuration_set_name, "") != ""
    )
    error_message = "The SES identity is the domain, with a configuration set."
  }
}

run "records_in_route53_with_a_zone" {
  command = plan

  variables {
    route53_zone_id = "Z0MOCKZONE0000000001"
  }

  assert {
    condition     = alltrue([for r in aws_route53_record.platform : r.zone_id == "Z0MOCKZONE0000000001"])
    error_message = "Every record is in var.route53_zone_id."
  }

  assert {
    condition = toset([
      for r in aws_route53_record.platform : jsonencode({
        type    = upper(r.type)
        name    = trimsuffix(lower(r.name), ".")
        records = sort([for v in r.records : trim(v, "\"")])
      })
      ]) == toset([
      for r in run.records_as_output_without_a_zone.dns_records : jsonencode({
        type    = upper(r.type)
        name    = trimsuffix(lower(r.name), ".")
        records = sort([for v in r.records : trim(v, "\"")])
      })
    ])
    error_message = "With route53_zone_id the Route 53 records are the same list the output gives without it (type, name, values)."
  }

  assert {
    condition = anytrue([
      for r in aws_route53_record.platform :
      upper(r.type) == "CAA" && trimsuffix(lower(r.name), ".") == "example.test" && contains(r.records, "0 issue \"letsencrypt.org\"")
    ])
    error_message = "A CAA record at the apex: 0 issue \"letsencrypt.org\"."
  }
}
