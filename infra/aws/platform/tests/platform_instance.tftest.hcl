# A3 (docs/phase1-m3-spec.md §13, §5.2 ec2, §5.3 H2): IMDSv2 required, hop limit 1,
# encrypted volumes, data volume `prevent_destroy`, no key pair, recover action on the
# system status check.
#
# Addresses this file expects (the spec names none):
#   aws_instance.platform                       the EC2 instance
#   aws_ebs_volume.data                         the 20 GB data volume (/var/lib/loom)
#   aws_cloudwatch_metric_alarm.h2_system       H2, StatusCheckFailed_System -> recover
# `prevent_destroy` is not in a plan, so it is read from the source: every aws_ebs_volume
# block in the stack (the data volume is the only one; the root volume is the instance's
# root_block_device).

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
  domain     = "example.test"
  brand_name = "Example Brand"
  ops_email  = "ops@example.test"
  budget_usd = 75
}

run "instance_metadata_volumes_and_access" {
  command = plan

  assert {
    condition     = try(aws_instance.platform.metadata_options[0].http_tokens, null) == "required"
    error_message = "IMDSv2 must be required: metadata_options { http_tokens = \"required\" }."
  }

  assert {
    condition     = try(aws_instance.platform.metadata_options[0].http_put_response_hop_limit, null) == 1
    error_message = "metadata_options { http_put_response_hop_limit = 1 } (containers get no instance credentials, §4)."
  }

  assert {
    condition     = try(aws_instance.platform.root_block_device[0].encrypted, false) == true
    error_message = "The root volume must be encrypted."
  }

  assert {
    condition = (
      try(aws_instance.platform.root_block_device[0].volume_type, null) == "gp3"
      && try(aws_instance.platform.root_block_device[0].volume_size, null) == 30
    )
    error_message = "The root volume is 30 GB gp3 (§5.2)."
  }

  assert {
    condition     = aws_instance.platform.key_name == null || aws_instance.platform.key_name == ""
    error_message = "No key pair on the instance (SSM only, no SSH)."
  }

  assert {
    condition     = aws_instance.platform.disable_api_termination == true
    error_message = "disable_api_termination must be set (§5.2)."
  }

  assert {
    condition     = aws_instance.platform.instance_type == "t4g.medium"
    error_message = "With the default variables the instance is a t4g.medium (§15 D3)."
  }

  assert {
    condition     = aws_instance.platform.ami == "ami-0a1b2c3d4e5f60718"
    error_message = "The AMI comes from the SSM public parameter (Amazon Linux 2023 arm64), here its mocked value."
  }
}

run "instance_type_follows_the_variable" {
  command = plan

  variables {
    instance_type = "t4g.large"
  }

  assert {
    condition     = aws_instance.platform.instance_type == "t4g.large"
    error_message = "aws_instance.platform.instance_type must come from var.instance_type."
  }
}

run "data_volume_encrypted_gp3_20gb" {
  command = plan

  assert {
    condition     = aws_ebs_volume.data.encrypted == true
    error_message = "The data volume must be encrypted."
  }

  assert {
    condition     = aws_ebs_volume.data.type == "gp3" && aws_ebs_volume.data.size == 20
    error_message = "The data volume is 20 GB gp3 (§5.2)."
  }
}

run "recover_action_on_the_system_status_check" {
  command = plan

  assert {
    condition = (
      aws_cloudwatch_metric_alarm.h2_system.namespace == "AWS/EC2"
      && aws_cloudwatch_metric_alarm.h2_system.metric_name == "StatusCheckFailed_System"
      && lookup(coalesce(aws_cloudwatch_metric_alarm.h2_system.dimensions, {}), "InstanceId", "") == aws_instance.platform.id
    )
    error_message = "H2 (system) reads AWS/EC2 StatusCheckFailed_System for the instance's InstanceId."
  }

  assert {
    condition     = contains(aws_cloudwatch_metric_alarm.h2_system.alarm_actions, "arn:aws:automate:us-east-1:ec2:recover")
    error_message = "H2 (system) must have the EC2 recover action (arn:aws:automate:<region>:ec2:recover)."
  }
}

run "source_data_volume_prevent_destroy_and_no_key_pair" {
  command = plan

  module {
    source = "./tests/fixtures/source_scan"
  }

  assert {
    condition     = output.ebs_volume_count >= 1 && alltrue(output.ebs_volumes_prevent_destroy)
    error_message = "Every aws_ebs_volume (the data volume) needs lifecycle { prevent_destroy = true }; found ${output.ebs_volume_count} volume block(s), prevent_destroy: ${jsonencode(output.ebs_volumes_prevent_destroy)}."
  }

  assert {
    condition     = !contains(output.resource_types, "aws_key_pair")
    error_message = "No key pair (no SSH, §4)."
  }
}
