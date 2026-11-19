# The platform box: t4g.medium on Amazon Linux 2023 arm64, SSM only (no key pair, no
# port 22), IMDSv2 with hop limit 1 so containers get no instance credentials
# (docs/phase1-m3-spec.md §4, §5.2).

data "aws_ssm_parameter" "al2023_arm64" {
  name = "/aws/service/ami-amazon-linux-latest/al2023-ami-kernel-default-arm64"
}

resource "aws_instance" "platform" {
  ami                         = data.aws_ssm_parameter.al2023_arm64.insecure_value
  instance_type               = var.instance_type
  subnet_id                   = aws_subnet.public.id
  vpc_security_group_ids      = [aws_security_group.instance.id]
  iam_instance_profile        = aws_iam_instance_profile.instance.name
  associate_public_ip_address = true # replaced by the Elastic IP once associated
  disable_api_termination     = true
  monitoring                  = false

  metadata_options {
    http_endpoint               = "enabled"
    http_tokens                 = "required"
    http_put_response_hop_limit = 1
  }

  root_block_device {
    volume_type           = "gp3"
    volume_size           = 30
    encrypted             = true
    delete_on_termination = true
  }

  # CRLF-proof: a Windows checkout must not hand cloud-init "#!/bin/bash\r".
  user_data = replace(templatefile("${path.module}/user_data.sh.tftpl", {
    data_volume_serial = replace(aws_ebs_volume.data.id, "-", "")
  }), "\r\n", "\n")

  tags = { Name = "loom-platform" }

  lifecycle {
    # A newer AMI or an edited boot script must not replace the box (its data volume is
    # separate, but a replacement still means downtime and a new instance id).
    ignore_changes = [ami, user_data]
  }
}

# Redis AOF, the settle spool, Caddy's certificates, the pool manager's state and the
# Prometheus TSDB live here, so the volume outlives the instance (prevent_destroy; the
# runbook's teardown removes it explicitly, §6.10). Daily snapshots: dlm.tf.
resource "aws_ebs_volume" "data" {
  availability_zone = aws_subnet.public.availability_zone
  type              = "gp3"
  size              = 20
  encrypted         = true

  tags = {
    Name            = "loom-platform-data"
    "loom:snapshot" = "daily"
  }

  lifecycle {
    prevent_destroy = true
  }
}

resource "aws_volume_attachment" "data" {
  device_name = "/dev/sdf"
  volume_id   = aws_ebs_volume.data.id
  instance_id = aws_instance.platform.id
}

resource "aws_eip" "platform" {
  domain = "vpc"

  tags = { Name = "loom-platform" }
}

resource "aws_eip_association" "platform" {
  allocation_id = aws_eip.platform.id
  instance_id   = aws_instance.platform.id
}

# The services' log group (the awslogs driver and the CloudWatch agent write here).
resource "aws_cloudwatch_log_group" "platform" {
  name              = "/loom/platform"
  retention_in_days = 30
}

# The CloudWatch agent's Prometheus-metrics log group (the agent writes the metric events here).
resource "aws_cloudwatch_log_group" "prometheus" {
  name              = "/loom/platform/prometheus"
  retention_in_days = 30
}
