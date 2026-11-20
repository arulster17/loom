# The site parameter: the one place the domain and the product name are set, plus the keys
# loom-deploy reads (docs/phase1-m3-spec.md §6.11, §6.12). Standard tier, free. Nothing
# here is secret: the RDS master secret is named by its ARN only. Only the instance role
# may read it (iam.tf).

resource "aws_ssm_parameter" "site" {
  name = "/loom/platform/site"
  type = "String"
  tier = "Standard"

  insecure_value = jsonencode({
    domain               = var.domain
    brand_name           = var.brand_name
    ops_email            = var.ops_email
    releases_bucket      = aws_s3_bucket.releases.bucket
    db_host              = aws_db_instance.platform.address
    db_port              = aws_db_instance.platform.port
    db_master_secret_arn = aws_db_instance.platform.master_user_secret[0].secret_arn
  })
}
