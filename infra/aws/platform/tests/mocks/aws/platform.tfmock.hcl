# Mock values for the default `aws` provider in the platform stack's tftests
# (docs/phase1-m3-spec.md §5.5: no credentials, no AWS calls).
#
# The tests load this file with `mock_provider "aws" { source = "./tests/mocks/aws",
# override_during = plan }`, so these values are known at plan time. Mock defaults only
# fill computed attributes the configuration leaves unset: for an attribute the spec
# requires the configuration to set, the default here is AWS's own default or a
# placeholder that fails the matching assertion, never the required value.

# Data sources (§5.5: AMI parameter, caller identity, partition, region, AZs).

mock_data "aws_region" {
  defaults = {
    id          = "us-east-1"
    region      = "us-east-1"
    name        = "us-east-1"
    endpoint    = "ec2.us-east-1.amazonaws.com"
    description = "US East (N. Virginia)"
  }
}

mock_data "aws_availability_zones" {
  defaults = {
    id       = "us-east-1"
    names    = ["us-east-1a", "us-east-1b", "us-east-1c"]
    zone_ids = ["use1-az1", "use1-az2", "use1-az4"]
  }
}

mock_data "aws_caller_identity" {
  defaults = {
    id         = "123456789012"
    account_id = "123456789012"
    arn        = "arn:aws:iam::123456789012:user/mock-terraform"
    user_id    = "AIDAMOCKMOCKMOCKMOCK"
  }
}

mock_data "aws_partition" {
  defaults = {
    id                 = "aws"
    partition          = "aws"
    dns_suffix         = "amazonaws.com"
    reverse_dns_prefix = "com.amazonaws"
  }
}

# The Amazon Linux 2023 arm64 AMI id from the SSM public parameter (§5.2).
mock_data "aws_ssm_parameter" {
  defaults = {
    arn            = "arn:aws:ssm:us-east-1::parameter/aws/service/ami-amazon-linux-latest/al2023-ami-kernel-default-arm64"
    type           = "String"
    value          = "ami-0a1b2c3d4e5f60718"
    insecure_value = "ami-0a1b2c3d4e5f60718"
    version        = 1
  }
}

mock_data "aws_iam_openid_connect_provider" {
  defaults = {
    arn = "arn:aws:iam::123456789012:oidc-provider/token.actions.githubusercontent.com"
    url = "token.actions.githubusercontent.com"
  }
}

mock_data "aws_iam_policy" {
  defaults = {
    arn = "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore"
  }
}

# Resources: computed attributes the tests read.

mock_resource "aws_vpc" {
  defaults = {
    id  = "vpc-0mock0000000000001"
    arn = "arn:aws:ec2:us-east-1:123456789012:vpc/vpc-0mock0000000000001"
  }
}

mock_resource "aws_subnet" {
  defaults = {
    id  = "subnet-0mock000000000001"
    arn = "arn:aws:ec2:us-east-1:123456789012:subnet/subnet-0mock000000000001"
  }
}

# No inline rules unless the configuration writes them (AWS creates none).
mock_resource "aws_security_group" {
  defaults = {
    id      = "sg-0mock00000000000001"
    arn     = "arn:aws:ec2:us-east-1:123456789012:security-group/sg-0mock00000000000001"
    ingress = []
    egress  = []
  }
}

mock_resource "aws_instance" {
  defaults = {
    id                      = "i-0mock00000000000001"
    arn                     = "arn:aws:ec2:us-east-1:123456789012:instance/i-0mock00000000000001"
    ami                     = "ami-not-set-by-the-configuration"
    instance_type           = "not-set-by-the-configuration"
    key_name                = ""
    disable_api_termination = false
    vpc_security_group_ids  = []
    public_ip               = "203.0.113.99"
    private_ip              = "10.42.0.10"
    # Without metadata_options in the configuration, IMDSv1 stays allowed.
    metadata_options = {
      http_endpoint               = "enabled"
      http_tokens                 = "optional"
      http_put_response_hop_limit = 2
    }
    root_block_device = {
      encrypted   = false
      volume_type = "gp2"
      volume_size = 8
    }
  }
}

mock_resource "aws_ebs_volume" {
  defaults = {
    id        = "vol-0mock00000000000001"
    arn       = "arn:aws:ec2:us-east-1:123456789012:volume/vol-0mock00000000000001"
    encrypted = false
    type      = "gp2"
    size      = 0
  }
}

mock_resource "aws_eip" {
  defaults = {
    id            = "eipalloc-0mock000000001"
    allocation_id = "eipalloc-0mock000000001"
    public_ip     = "203.0.113.10"
    public_dns    = "ec2-203-0-113-10.compute-1.amazonaws.com"
  }
}

mock_resource "aws_db_instance" {
  defaults = {
    id                      = "db-MOCKMOCKMOCKMOCKMOCKMOCK"
    resource_id             = "db-MOCKMOCKMOCKMOCKMOCKMOCK"
    arn                     = "arn:aws:rds:us-east-1:123456789012:db:loom-mock-db"
    identifier              = "loom-mock-db"
    address                 = "loom-mock-db.cmock.us-east-1.rds.amazonaws.com"
    endpoint                = "loom-mock-db.cmock.us-east-1.rds.amazonaws.com:5432"
    port                    = 5432
    engine                  = "not-set-by-the-configuration"
    engine_version          = "not-set-by-the-configuration"
    username                = "not-set-by-the-configuration"
    allocated_storage       = 0
    storage_type            = "gp2"
    multi_az                = false
    backup_retention_period = 1
    parameter_group_name    = "default.postgres16"
    vpc_security_group_ids  = []
    master_user_secret = [{
      kms_key_id    = "arn:aws:kms:us-east-1:123456789012:key/mock"
      secret_arn    = "arn:aws:secretsmanager:us-east-1:123456789012:secret:rds!db-mock-AbCdEf"
      secret_status = "active"
    }]
  }
}

mock_resource "aws_db_parameter_group" {
  defaults = {
    id   = "loom-mock-pg"
    name = "loom-mock-pg"
    arn  = "arn:aws:rds:us-east-1:123456789012:pg:loom-mock-pg"
  }
}

mock_resource "aws_db_subnet_group" {
  defaults = {
    id   = "loom-mock-subnets"
    name = "loom-mock-subnets"
    arn  = "arn:aws:rds:us-east-1:123456789012:subgrp:loom-mock-subnets"
  }
}

# Every secret shares this ARN unless a test overrides one instance.
mock_resource "aws_secretsmanager_secret" {
  defaults = {
    id  = "arn:aws:secretsmanager:us-east-1:123456789012:secret:loom/platform/mock-AbCdEf"
    arn = "arn:aws:secretsmanager:us-east-1:123456789012:secret:loom/platform/mock-AbCdEf"
  }
}

# Every role shares this ARN unless a test overrides one role.
mock_resource "aws_iam_role" {
  defaults = {
    id        = "loom-mock-role"
    arn       = "arn:aws:iam::123456789012:role/loom-mock-role"
    unique_id = "AROAMOCKMOCKMOCKMOCK"
  }
}

mock_resource "aws_iam_instance_profile" {
  defaults = {
    id  = "loom-mock-profile"
    arn = "arn:aws:iam::123456789012:instance-profile/loom-mock-profile"
  }
}

mock_resource "aws_iam_user" {
  defaults = {
    id        = "loom-ses-smtp"
    arn       = "arn:aws:iam::123456789012:user/loom-ses-smtp"
    unique_id = "AIDAMOCKSESSMTPMOCK"
  }
}

mock_resource "aws_iam_openid_connect_provider" {
  defaults = {
    id  = "arn:aws:iam::123456789012:oidc-provider/token.actions.githubusercontent.com"
    arn = "arn:aws:iam::123456789012:oidc-provider/token.actions.githubusercontent.com"
  }
}

mock_resource "aws_ecr_repository" {
  defaults = {
    id             = "loom/mock"
    arn            = "arn:aws:ecr:us-east-1:123456789012:repository/loom/mock"
    registry_id    = "123456789012"
    repository_url = "123456789012.dkr.ecr.us-east-1.amazonaws.com/loom/mock"
  }
}

mock_resource "aws_s3_bucket" {
  defaults = {
    id                          = "loom-mock-leases"
    bucket                      = "loom-mock-leases"
    arn                         = "arn:aws:s3:::loom-mock-leases"
    bucket_domain_name          = "loom-mock-leases.s3.amazonaws.com"
    bucket_regional_domain_name = "loom-mock-leases.s3.us-east-1.amazonaws.com"
  }
}

mock_resource "aws_ssm_parameter" {
  defaults = {
    arn            = "arn:aws:ssm:us-east-1:123456789012:parameter/loom/platform/site"
    tier           = "Standard"
    value          = ""
    insecure_value = ""
    version        = 1
  }
}

mock_resource "aws_sns_topic" {
  defaults = {
    id  = "arn:aws:sns:us-east-1:123456789012:loom-platform-alerts"
    arn = "arn:aws:sns:us-east-1:123456789012:loom-platform-alerts"
  }
}

mock_resource "aws_cloudwatch_log_group" {
  defaults = {
    arn = "arn:aws:logs:us-east-1:123456789012:log-group:/loom/platform/mock"
  }
}

mock_resource "aws_lambda_function" {
  defaults = {
    arn           = "arn:aws:lambda:us-east-1:123456789012:function:loom-serving-runpod-reaper"
    invoke_arn    = "arn:aws:apigateway:us-east-1:lambda:path/2015-03-31/functions/arn:aws:lambda:us-east-1:123456789012:function:loom-serving-runpod-reaper/invocations"
    qualified_arn = "arn:aws:lambda:us-east-1:123456789012:function:loom-serving-runpod-reaper:$LATEST"
    version       = "$LATEST"
  }
}

mock_resource "aws_cloudwatch_event_rule" {
  defaults = {
    arn = "arn:aws:events:us-east-1:123456789012:rule/loom-mock-rule"
  }
}

# A health check made through the default provider (the H3 test tells the providers
# apart by the alarm's ARN, not the check's id).
mock_resource "aws_route53_health_check" {
  defaults = {
    id                = "mock-hooks-health-check"
    arn               = "arn:aws:route53:::healthcheck/mock-hooks-health-check"
    failure_threshold = 3
    enable_sni        = false
  }
}

# Alarms made through the default provider carry this ARN; through aws.us_east_1 they
# carry the one in ../aws_us_east_1.
mock_resource "aws_cloudwatch_metric_alarm" {
  defaults = {
    arn = "arn:aws:cloudwatch:us-east-1:123456789012:alarm:mock-made-by-default-provider"
  }
}

mock_resource "aws_sesv2_email_identity" {
  defaults = {
    arn = "arn:aws:ses:us-east-1:123456789012:identity/example.test"
    # Applies when the configuration writes a dkim_signing_attributes block (Easy DKIM).
    dkim_signing_attributes = {
      tokens = ["mocktokena", "mocktokenb", "mocktokenc"]
    }
  }
}

mock_resource "aws_ses_domain_dkim" {
  defaults = {
    dkim_tokens = ["mocktokena", "mocktokenb", "mocktokenc"]
  }
}

mock_resource "aws_sesv2_configuration_set" {
  defaults = {
    arn = "arn:aws:ses:us-east-1:123456789012:configuration-set/loom-mock"
  }
}

mock_resource "aws_route53_record" {
  defaults = {
    fqdn = "mock.example.test"
  }
}
