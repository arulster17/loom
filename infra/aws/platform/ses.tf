# SES identity for the domain: Easy DKIM, a custom MAIL FROM subdomain and a configuration
# set (docs/phase1-m3-spec.md §5.2 ses). The DNS records are in dns.tf. Production access
# is a console request (§3).

resource "aws_sesv2_configuration_set" "platform" {
  configuration_set_name = "loom-platform"

  delivery_options {
    tls_policy = "REQUIRE"
  }

  reputation_options {
    reputation_metrics_enabled = true
  }

  sending_options {
    sending_enabled = true
  }
}

resource "aws_sesv2_email_identity" "domain" {
  email_identity         = var.domain
  configuration_set_name = aws_sesv2_configuration_set.platform.configuration_set_name

  # An empty block selects Easy DKIM (SES generates the three tokens).
  dkim_signing_attributes {}
}

resource "aws_sesv2_email_identity_mail_from_attributes" "domain" {
  email_identity         = aws_sesv2_email_identity.domain.email_identity
  mail_from_domain       = "mail.${var.domain}"
  behavior_on_mx_failure = "USE_DEFAULT_VALUE"
}
