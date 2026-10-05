output "region" {
  value = var.region
}

output "bucket" {
  value = aws_s3_bucket.bench.bucket
}

output "instance_profile_name" {
  value = aws_iam_instance_profile.instance.name
}

output "instance_role_arn" {
  value = aws_iam_role.instance.arn
}

output "security_group_id" {
  value = aws_security_group.instance.id
}

output "subnet_ids" {
  value = sort(data.aws_subnets.default.ids)
}

output "hf_token_secret_name" {
  value = var.hf_token_secret_name
}

output "reaper_function_name" {
  value = aws_lambda_function.reaper.function_name
}

output "runner_policy_arn" {
  description = "Attach to the user or role that runs `bench run`."
  value       = aws_iam_policy.runner.arn
}

output "runner_policy_json" {
  value = data.aws_iam_policy_document.runner.json
}

output "aws_settings_yaml" {
  description = "AwsSettings for loom_bench: terraform output -raw aws_settings_yaml > aws.local.yaml"
  value = yamlencode({
    region                = var.region
    bucket                = aws_s3_bucket.bench.bucket
    instance_profile_name = aws_iam_instance_profile.instance.name
    security_group_id     = aws_security_group.instance.id
    subnet_ids            = sort(data.aws_subnets.default.ids)
    hf_token_secret_name  = var.hf_token_secret_name
    owner                 = var.owner
    name_prefix           = var.name_prefix
  })
}
