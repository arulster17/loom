# Extra regions the aws_ec2 provider may launch GPU hosts in (AwsSettings.extra_regions)
# when the primary region (var.region) has no capacity. Each gets a no-ingress security
# group in its default VPC (modules/host_network). The reaper Lambda and the runner
# policy below cover every host region (local.host_regions).
#
# Provider configurations cannot be made per element of a list, so each supported extra
# region has its own provider alias and module block; var.extra_host_regions turns them
# on (count) and sets their order in aws_settings_yaml.

provider "aws" {
  alias  = "us_east_2"
  region = "us-east-2"

  default_tags {
    tags = {
      "loom:stack" = var.name_prefix
      "loom:owner" = var.owner
    }
  }
}

provider "aws" {
  alias  = "us_west_2"
  region = "us-west-2"

  default_tags {
    tags = {
      "loom:stack" = var.name_prefix
      "loom:owner" = var.owner
    }
  }
}

module "host_network_us_east_2" {
  source = "./modules/host_network"
  count  = contains(var.extra_host_regions, "us-east-2") ? 1 : 0

  providers = {
    aws = aws.us_east_2
  }

  region      = "us-east-2"
  name_prefix = var.name_prefix
}

module "host_network_us_west_2" {
  source = "./modules/host_network"
  count  = contains(var.extra_host_regions, "us-west-2") ? 1 : 0

  providers = {
    aws = aws.us_west_2
  }

  region      = "us-west-2"
  name_prefix = var.name_prefix
}

locals {
  extra_networks_by_region = {
    for m in concat(module.host_network_us_east_2, module.host_network_us_west_2) : m.region => m
  }

  # In var.extra_host_regions order: the order the provider sweeps them in.
  extra_host_networks = [for r in var.extra_host_regions : local.extra_networks_by_region[r]]

  host_regions  = concat([var.region], var.extra_host_regions)
  host_ec2_arns = [for r in local.host_regions : "arn:${local.partition}:ec2:${r}:${data.aws_caller_identity.current.account_id}"]

  host_security_group_arns = concat(
    ["${local.ec2_arn}:security-group/${aws_security_group.instance.id}"],
    [for m in local.extra_host_networks : "arn:${local.partition}:ec2:${m.region}:${data.aws_caller_identity.current.account_id}:security-group/${m.security_group_id}"],
  )
}
