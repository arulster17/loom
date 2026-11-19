# A2 (docs/phase1-m3-spec.md §13, §4, §5.2 network/ec2/rds): no ingress except 80/443 to
# the instance; RDS ingress only from the instance's security group; RDS not public; no
# NAT gateway.
#
# Addresses this file expects (the spec names none):
#   aws_security_group.instance  the instance's security group, ingress as inline blocks
#   aws_security_group.rds       the database's security group, ingress as inline blocks
#   aws_instance.platform        the EC2 instance
#   aws_db_instance.platform     the RDS instance
# Ingress is written inline so a plan shows all of it: the stack declares no
# aws_security_group_rule / aws_vpc_security_group_ingress_rule (checked from the source).

mock_provider "aws" {
  source          = "./tests/mocks/aws"
  override_during = plan
}

mock_provider "aws" {
  alias           = "us_east_1"
  source          = "./tests/mocks/aws_us_east_1"
  override_during = plan
}

variables {
  domain          = "example.test"
  brand_name      = "Example Brand"
  ops_email       = "ops@example.test"
  budget_usd      = 75
  releases_bucket = "loom-test-releases"
}

# Distinct ids for the two groups, so "from the instance's group" can be told apart.
override_resource {
  target          = aws_security_group.instance
  override_during = plan
  values = {
    id  = "sg-0mockinstance000001"
    arn = "arn:aws:ec2:us-east-1:123456789012:security-group/sg-0mockinstance000001"
  }
}

override_resource {
  target          = aws_security_group.rds
  override_during = plan
  values = {
    id  = "sg-0mockrds00000000001"
    arn = "arn:aws:ec2:us-east-1:123456789012:security-group/sg-0mockrds00000000001"
  }
}

run "instance_ingress_is_only_80_and_443_from_anywhere" {
  command = plan

  assert {
    condition = toset([
      for r in aws_security_group.instance.ingress :
      "${contains(["6", "tcp"], lower(r.protocol)) ? "tcp" : lower(r.protocol)}/${r.from_port}-${r.to_port}"
    ]) == toset(["tcp/80-80", "tcp/443-443"])
    error_message = "The instance's security group must allow exactly tcp 80 and tcp 443 in (no 22, no other port, no all-protocol rule); got ${jsonencode([for r in aws_security_group.instance.ingress : { protocol = r.protocol, from = r.from_port, to = r.to_port }])}."
  }

  assert {
    condition = alltrue([
      for r in aws_security_group.instance.ingress :
      contains(coalesce(r.cidr_blocks, []), "0.0.0.0/0")
      && length(coalesce(r.security_groups, [])) == 0
      && r.self != true
      && length(coalesce(r.prefix_list_ids, [])) == 0
    ])
    error_message = "80 and 443 are open to the internet (0.0.0.0/0), not to other groups or prefix lists."
  }

  assert {
    condition     = toset(aws_instance.platform.vpc_security_group_ids) == toset([aws_security_group.instance.id])
    error_message = "The instance must be in its own security group only (another group could open more ports)."
  }
}

run "rds_ingress_only_from_the_instance_group_and_not_public" {
  command = plan

  assert {
    condition     = length(aws_security_group.rds.ingress) == 1
    error_message = "The database's security group must have exactly one ingress rule; got ${length(aws_security_group.rds.ingress)}."
  }

  assert {
    condition = alltrue([
      for r in aws_security_group.rds.ingress :
      contains(["6", "tcp"], lower(r.protocol)) && r.from_port == 5432 && r.to_port == 5432
      && toset(coalesce(r.security_groups, [])) == toset([aws_security_group.instance.id])
      && length(coalesce(r.cidr_blocks, [])) == 0
      && length(coalesce(r.ipv6_cidr_blocks, [])) == 0
      && length(coalesce(r.prefix_list_ids, [])) == 0
      && r.self != true
    ])
    error_message = "RDS ingress must be tcp 5432 from the instance's security group only (no CIDR, no self); got ${jsonencode(aws_security_group.rds.ingress)}."
  }

  assert {
    condition     = aws_db_instance.platform.publicly_accessible == false
    error_message = "RDS must not be publicly accessible."
  }

  assert {
    condition     = toset(aws_db_instance.platform.vpc_security_group_ids) == toset([aws_security_group.rds.id])
    error_message = "RDS must be in the database security group only."
  }
}

run "source_has_no_nat_gateway_and_no_separate_ingress_rules" {
  command = plan

  module {
    source = "./tests/fixtures/source_scan"
  }

  assert {
    condition     = contains(output.tf_files, "network.tf")
    error_message = "infra/aws/platform/network.tf is missing (§5.1)."
  }

  assert {
    condition     = !contains(output.resource_types, "aws_nat_gateway")
    error_message = "No NAT gateway (§5.2: RDS needs no egress)."
  }

  assert {
    condition = length(setintersection(toset(output.resource_types), toset([
      "aws_security_group_rule",
      "aws_vpc_security_group_ingress_rule",
    ]))) == 0
    error_message = "Write security group ingress inline, so the tests can see every rule (A2)."
  }
}
