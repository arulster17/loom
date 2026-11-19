output "releases_bucket" {
  description = "Name of the releases bucket (releases/<sha>.json and releases/<sha>/)."
  value       = aws_s3_bucket.releases.bucket
}

output "lease_bucket" {
  description = "Name of the lease bucket (serving/leases/<pod id>, M4)."
  value       = aws_s3_bucket.leases.bucket
}

output "dns_records" {
  description = "Every DNS record the platform needs (apex, api and hooks A records, CAA, SES DKIM, MAIL FROM MX and SPF, DMARC). Terraform writes them when route53_zone_id is set; otherwise create them at your DNS host."
  value       = [for k in sort(keys(local.dns_records)) : local.dns_records[k]]
}

output "elastic_ip" {
  description = "The box's public address; becomes gateway_egress_cidrs in M4."
  value       = aws_eip.platform.public_ip
}

output "instance_id" {
  description = "The box's instance id (aws ssm start-session --target <id>)."
  value       = aws_instance.platform.id
}

output "rds_endpoint" {
  description = "The database's address and port."
  value       = aws_db_instance.platform.endpoint
}

output "rds_master_secret_arn" {
  description = "The RDS-managed secret holding the loom_owner login."
  value       = aws_db_instance.platform.master_user_secret[0].secret_arn
}

output "ecr_repository_urls" {
  description = "ECR repository URL per image."
  value       = { for k, r in aws_ecr_repository.loom : k => r.repository_url }
}

output "github_images_role_arn" {
  description = "The role images.yml assumes through GitHub OIDC (environment platform-images)."
  value       = aws_iam_role.github_images.arn
}

output "alerts_topic_arn" {
  description = "The SNS topic every alarm notifies (confirm the email subscription)."
  value       = aws_sns_topic.alerts.arn
}

output "site_parameter" {
  description = "The SSM parameter holding the domain, the product name and the keys loom-deploy reads."
  value       = aws_ssm_parameter.site.name
}

output "ses_smtp_user" {
  description = "The IAM user whose access key the runbook creates for SMTP."
  value       = aws_iam_user.ses_smtp.name
}

output "serving_reaper_function" {
  description = "The serving reaper Lambda (dry run, schedule disabled until M4)."
  value       = aws_lambda_function.serving_reaper.function_name
}
