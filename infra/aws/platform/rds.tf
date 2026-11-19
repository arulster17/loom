# PostgreSQL 16 on db.t4g.micro in the private subnets (docs/phase1-m3-spec.md §5.2).
# The master user's secret is made and rotated by RDS (manage_master_user_password), so
# no database credential ever enters the configuration or the state.

resource "aws_db_subnet_group" "platform" {
  name       = "loom-platform"
  subnet_ids = [aws_subnet.private_a.id, aws_subnet.private_b.id]

  tags = { Name = "loom-platform" }
}

# rds.force_ssl: the services connect with sslmode=verify-full (§6.4). The server-side
# TCP keepalives keep idle pooled connections from being dropped silently (§8.3).
resource "aws_db_parameter_group" "platform" {
  name   = "loom-platform-postgres16"
  family = "postgres16"

  parameter {
    name  = "rds.force_ssl"
    value = "1"
  }

  parameter {
    name  = "tcp_keepalives_idle"
    value = "60"
  }

  parameter {
    name  = "tcp_keepalives_interval"
    value = "10"
  }

  parameter {
    name  = "tcp_keepalives_count"
    value = "3"
  }

  tags = { Name = "loom-platform-postgres16" }
}

resource "aws_db_instance" "platform" {
  identifier = "loom-platform"

  engine                      = "postgres"
  engine_version              = "16"
  auto_minor_version_upgrade  = true
  allow_major_version_upgrade = false
  instance_class              = "db.t4g.micro"

  allocated_storage = 20
  storage_type      = "gp3"
  storage_encrypted = true

  db_name                     = "loom_platform"
  username                    = "loom_owner"
  manage_master_user_password = true

  multi_az               = false
  publicly_accessible    = false
  db_subnet_group_name   = aws_db_subnet_group.platform.name
  vpc_security_group_ids = [aws_security_group.rds.id]
  parameter_group_name   = aws_db_parameter_group.platform.name

  iam_database_authentication_enabled = true

  backup_retention_period   = 7
  copy_tags_to_snapshot     = true
  deletion_protection       = var.db_deletion_protection
  skip_final_snapshot       = false
  final_snapshot_identifier = "loom-platform-final"

  tags = { Name = "loom-platform" }
}
