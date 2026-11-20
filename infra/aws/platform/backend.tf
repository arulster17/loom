# Remote state in the bucket infra/aws/state creates, with S3 native locking (no DynamoDB
# table). Partial configuration: `terraform init -backend-config=backend.hcl`, with
# backend.hcl written from backend.hcl.example (gitignored). CI and the agents use
# `terraform init -backend=false`.
terraform {
  backend "s3" {
    use_lockfile = true
  }
}
