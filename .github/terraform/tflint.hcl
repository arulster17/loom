# tflint config for the M3 Terraform stacks (infra/aws/platform, infra/aws/state), read by
# CI job `terraform` with --config, so a .tflint.hcl inside a stack cannot replace it.
# docs/phase1-m3-spec.md §5.5 and A11: the AWS ruleset, shallow rules only (deep checking
# would call AWS, and the job has no credentials). Every finding is fixed, or listed with a
# reason in infra/aws/platform/README.md.

config {
  format           = "compact"
  call_module_type = "local"
}

plugin "terraform" {
  enabled = true
  preset  = "recommended"
}

plugin "aws" {
  enabled    = true
  version    = "0.50.0"
  source     = "github.com/terraform-linters/tflint-ruleset-aws"
  deep_check = false
}
