# DNS records (docs/phase1-m3-spec.md §5.2 dns). With route53_zone_id set, Terraform
# writes them; without it the same list is the dns_records output for the user's DNS host.
# Keys are static (never built from the Elastic IP or the DKIM tokens, which a plan does not
# know yet).

locals {
  dkim_tokens = aws_sesv2_email_identity.domain.dkim_signing_attributes[0].tokens
  mail_from   = aws_sesv2_email_identity_mail_from_attributes.domain.mail_from_domain

  dns_records = merge(
    {
      "apex-a"  = { name = var.domain, type = "A", records = [aws_eip.platform.public_ip] }
      "api-a"   = { name = "api.${var.domain}", type = "A", records = [aws_eip.platform.public_ip] }
      "hooks-a" = { name = "hooks.${var.domain}", type = "A", records = [aws_eip.platform.public_ip] }

      # Only Let's Encrypt may issue certificates for the domain and its subdomains.
      "apex-caa" = { name = var.domain, type = "CAA", records = ["0 issue \"letsencrypt.org\""] }

      # Custom MAIL FROM: the MX sends bounces to SES, the SPF record authorises it.
      "mail-from-mx"  = { name = local.mail_from, type = "MX", records = ["10 feedback-smtp.${local.region}.amazonses.com"] }
      "mail-from-spf" = { name = local.mail_from, type = "TXT", records = ["\"v=spf1 include:amazonses.com ~all\""] }

      "dmarc" = { name = "_dmarc.${var.domain}", type = "TXT", records = ["\"v=DMARC1; p=quarantine\""] }
    },
    {
      for i in range(3) : "dkim-${i}" => {
        name    = "${local.dkim_tokens[i]}._domainkey.${var.domain}"
        type    = "CNAME"
        records = ["${local.dkim_tokens[i]}.dkim.amazonses.com"]
      }
    },
  )
}

resource "aws_route53_record" "platform" {
  for_each = var.route53_zone_id == null ? {} : local.dns_records

  zone_id = var.route53_zone_id
  name    = each.value.name
  type    = each.value.type
  ttl     = 300
  records = each.value.records
}
