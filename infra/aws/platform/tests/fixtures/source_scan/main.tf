# Test helper for the platform stack's tftests: reads the stack's own *.tf files (no
# providers, no resources) and reports the facts a plan cannot show: which resource and
# data source types are declared, `lifecycle { prevent_destroy }`, and whether a variable
# has a default. Run from a test with `module { source = "./tests/fixtures/source_scan" }`.
# The patterns assume `terraform fmt` layout (CI checks it): top-level blocks start at
# column 0 and end with a `}` alone at column 0.

locals {
  stack_dir = "${path.module}/../../.."
  files     = { for f in fileset(local.stack_dir, "*.tf") : f => file("${local.stack_dir}/${f}") }
  all       = join("\n", [for f in sort(keys(local.files)) : local.files[f]])

  # One string per top-level block: a one-line block, or everything up to the first `}`
  # at column 0.
  block_re = "(?ms)^%s\\s+%s\\s*\\{(?:[^\\n]*\\}$|.*?^\\})"

  resource_types = distinct(flatten(regexall("(?m)^resource\\s+\"([A-Za-z0-9_]+)\"\\s+\"", local.all)))
  data_types     = distinct(flatten(regexall("(?m)^data\\s+\"([A-Za-z0-9_]+)\"\\s+\"", local.all)))

  ebs_volume_blocks = regexall(format(local.block_re, "resource", "\"aws_ebs_volume\"\\s+\"[^\"]+\""), local.all)

  variable_blocks = {
    for v in ["domain", "brand_name"] :
    v => regexall(format(local.block_re, "variable", "\"${v}\""), local.all)
  }

  healthcheck_tf = lookup(local.files, "healthcheck.tf", "")
}

output "tf_files" {
  value = sort(keys(local.files))
}

output "resource_types" {
  value = local.resource_types
}

output "data_types" {
  value = local.data_types
}

output "ebs_volume_count" {
  value = length(local.ebs_volume_blocks)
}

output "ebs_volumes_prevent_destroy" {
  value = [for b in local.ebs_volume_blocks : length(regexall("(?m)^\\s*prevent_destroy\\s*=\\s*true\\s*$", b)) > 0]
}

# For each of domain and brand_name: how many `variable` blocks declare it, and whether
# any of them sets `default`.
output "variable_declared" {
  value = { for v, blocks in local.variable_blocks : v => length(blocks) }
}

output "variable_has_default" {
  value = { for v, blocks in local.variable_blocks : v => anytrue([for b in blocks : length(regexall("(^|[\\s{])default\\s*=", b)) > 0]) }
}

output "healthcheck_tf_declares_health_check" {
  value = length(regexall("(?m)^resource\\s+\"aws_route53_health_check\"\\s+\"", local.healthcheck_tf)) > 0
}
