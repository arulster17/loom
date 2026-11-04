# The network a bench GPU host needs in one extra region: the region's default VPC and
# its default-for-AZ subnets (read, not created) and a security group with no ingress.
# Security groups and subnets are regional, so each extra region gets its own; the
# instance profile, bucket and HF token secret stay in the primary region (IAM is global).

terraform {
  required_providers {
    aws = {
      source = "hashicorp/aws"
    }
  }
}

variable "region" {
  description = "The region this module's aws provider is configured for."
  type        = string
}

variable "name_prefix" {
  description = "Prefix for every resource name."
  type        = string
}

data "aws_vpc" "default" {
  default = true
}

data "aws_subnets" "default" {
  filter {
    name   = "vpc-id"
    values = [data.aws_vpc.default.id]
  }

  filter {
    name   = "default-for-az"
    values = ["true"]
  }
}

resource "aws_security_group" "instance" {
  name_prefix = "${var.name_prefix}-instance-"
  description = "Loom bench GPU hosts: no ingress (managed over SSM); egress for images and weights"
  vpc_id      = data.aws_vpc.default.id

  egress {
    description = "Docker Hub, Hugging Face, PyPI, AWS APIs"
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  lifecycle {
    create_before_destroy = true
  }
}

output "region" {
  value = var.region
}

output "security_group_id" {
  value = aws_security_group.instance.id
}

output "subnet_ids" {
  value = sort(data.aws_subnets.default.ids)
}
