# A4 (docs/phase1-m3-spec.md §13, §5.2 rds): force SSL parameter, encryption, 7-day
# backups, deletion protection variable, final snapshot, managed master password. The
# "no `password =` anywhere" half is a grep test in another area's pytest.
#
# Addresses this file expects (the spec names none):
#   aws_db_instance.platform         the RDS instance
#   aws_db_parameter_group.platform  its parameter group

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

run "rds_defaults" {
  command = plan

  assert {
    condition = (
      aws_db_instance.platform.engine == "postgres"
      && startswith(aws_db_instance.platform.engine_version, "16")
    )
    error_message = "PostgreSQL 16 (compose runs 16); got ${aws_db_instance.platform.engine} ${aws_db_instance.platform.engine_version}."
  }

  assert {
    condition     = aws_db_instance.platform.instance_class == "db.t4g.micro"
    error_message = "db.t4g.micro (§5.2)."
  }

  assert {
    condition     = aws_db_instance.platform.allocated_storage == 20 && aws_db_instance.platform.storage_type == "gp3"
    error_message = "20 GB gp3 (§5.2)."
  }

  assert {
    condition     = aws_db_instance.platform.storage_encrypted == true
    error_message = "storage_encrypted must be true."
  }

  assert {
    condition     = aws_db_instance.platform.multi_az == false
    error_message = "Single AZ (§5.2)."
  }

  assert {
    condition     = aws_db_instance.platform.backup_retention_period == 7
    error_message = "Backups are kept 7 days."
  }

  assert {
    condition     = aws_db_instance.platform.deletion_protection == true
    error_message = "deletion_protection follows var.db_deletion_protection, whose default is true."
  }

  assert {
    condition = (
      aws_db_instance.platform.skip_final_snapshot != true
      && coalesce(aws_db_instance.platform.final_snapshot_identifier, "") != ""
    )
    error_message = "A final snapshot is taken on destroy: skip_final_snapshot false and a final_snapshot_identifier."
  }

  assert {
    condition     = aws_db_instance.platform.manage_master_user_password == true
    error_message = "manage_master_user_password = true: the master password lives in an RDS-managed secret."
  }

  assert {
    condition     = aws_db_instance.platform.password == null && aws_db_instance.platform.password_wo == null
    error_message = "No master password in Terraform (neither password nor password_wo)."
  }

  assert {
    condition     = aws_db_instance.platform.username == "loom_owner"
    error_message = "The master user is loom_owner."
  }

  assert {
    condition     = aws_db_instance.platform.auto_minor_version_upgrade != false
    error_message = "Auto minor version upgrades stay on."
  }

  assert {
    condition     = aws_db_instance.platform.publicly_accessible == false
    error_message = "RDS is not publicly accessible."
  }
}

run "rds_parameter_group_forces_ssl_and_keepalives" {
  command = plan

  assert {
    condition     = aws_db_instance.platform.parameter_group_name == aws_db_parameter_group.platform.name
    error_message = "The instance uses aws_db_parameter_group.platform."
  }

  assert {
    condition     = aws_db_parameter_group.platform.family == "postgres16"
    error_message = "The parameter group family is postgres16."
  }

  assert {
    condition = jsonencode({
      for p in aws_db_parameter_group.platform.parameter : p.name => tostring(p.value)
      if contains(["rds.force_ssl", "tcp_keepalives_idle", "tcp_keepalives_interval", "tcp_keepalives_count"], p.name)
      }) == jsonencode({
      "rds.force_ssl"           = "1"
      "tcp_keepalives_idle"     = "60"
      "tcp_keepalives_interval" = "10"
      "tcp_keepalives_count"    = "3"
    })
    error_message = "Parameters: rds.force_ssl = 1, tcp_keepalives_idle 60, tcp_keepalives_interval 10, tcp_keepalives_count 3 (§5.2, §8.3); got ${jsonencode([for p in aws_db_parameter_group.platform.parameter : { name = p.name, value = p.value }])}."
  }
}

run "deletion_protection_follows_the_variable" {
  command = plan

  variables {
    db_deletion_protection = false
  }

  assert {
    condition     = aws_db_instance.platform.deletion_protection == false
    error_message = "deletion_protection must be var.db_deletion_protection."
  }
}
