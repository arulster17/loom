output "state_bucket" {
  description = "The bucket name for infra/aws/platform/backend.hcl."
  value       = aws_s3_bucket.state.bucket
}
