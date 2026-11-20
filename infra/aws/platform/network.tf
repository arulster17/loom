# VPC 10.42.0.0/16: one public subnet for the instance, two private subnets (two AZs) for
# the RDS subnet group. No NAT gateway: RDS needs no egress (docs/phase1-m3-spec.md §5.2).

resource "aws_vpc" "platform" {
  cidr_block           = "10.42.0.0/16"
  enable_dns_support   = true
  enable_dns_hostnames = true

  tags = { Name = "loom-platform" }
}

# The VPC's default security group keeps no rules, so nothing can use it by accident.
resource "aws_default_security_group" "default" {
  vpc_id = aws_vpc.platform.id

  tags = { Name = "loom-platform-default-unused" }
}

resource "aws_internet_gateway" "platform" {
  vpc_id = aws_vpc.platform.id

  tags = { Name = "loom-platform" }
}

# The instance takes its public address from the Elastic IP (ec2.tf), not the subnet.
resource "aws_subnet" "public" {
  vpc_id                  = aws_vpc.platform.id
  cidr_block              = "10.42.0.0/24"
  availability_zone       = data.aws_availability_zones.available.names[0]
  map_public_ip_on_launch = false

  tags = { Name = "loom-platform-public-a" }
}

resource "aws_subnet" "private_a" {
  vpc_id            = aws_vpc.platform.id
  cidr_block        = "10.42.10.0/24"
  availability_zone = data.aws_availability_zones.available.names[0]

  tags = { Name = "loom-platform-private-a" }
}

resource "aws_subnet" "private_b" {
  vpc_id            = aws_vpc.platform.id
  cidr_block        = "10.42.11.0/24"
  availability_zone = data.aws_availability_zones.available.names[1]

  tags = { Name = "loom-platform-private-b" }
}

resource "aws_route_table" "public" {
  vpc_id = aws_vpc.platform.id

  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.platform.id
  }

  tags = { Name = "loom-platform-public" }
}

# No routes beyond the VPC's own: the private subnets cannot reach the internet.
resource "aws_route_table" "private" {
  vpc_id = aws_vpc.platform.id

  tags = { Name = "loom-platform-private" }
}

resource "aws_route_table_association" "public" {
  subnet_id      = aws_subnet.public.id
  route_table_id = aws_route_table.public.id
}

resource "aws_route_table_association" "private_a" {
  subnet_id      = aws_subnet.private_a.id
  route_table_id = aws_route_table.private.id
}

resource "aws_route_table_association" "private_b" {
  subnet_id      = aws_subnet.private_b.id
  route_table_id = aws_route_table.private.id
}

# Only 80 and 443 are open to the internet (80 redirects and answers ACME). Ingress is
# inline so a plan shows every rule. No port 22: the box is reached through SSM.
resource "aws_security_group" "instance" {
  name        = "loom-platform-instance"
  description = "Loom platform box: HTTP and HTTPS in, anything out"
  vpc_id      = aws_vpc.platform.id

  ingress {
    description = "HTTP: redirect and ACME"
    protocol    = "tcp"
    from_port   = 80
    to_port     = 80
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    description = "HTTPS"
    protocol    = "tcp"
    from_port   = 443
    to_port     = 443
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    description = "Package mirrors, ECR, S3, Stripe, SES, ACME, RDS"
    protocol    = "-1"
    from_port   = 0
    to_port     = 0
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = { Name = "loom-platform-instance" }
}

# RDS accepts the instance's security group on 5432 and nothing else, and has no egress.
resource "aws_security_group" "rds" {
  name        = "loom-platform-rds"
  description = "Loom platform database: PostgreSQL from the box only"
  vpc_id      = aws_vpc.platform.id

  ingress {
    description     = "PostgreSQL from the instance"
    protocol        = "tcp"
    from_port       = 5432
    to_port         = 5432
    security_groups = [aws_security_group.instance.id]
  }

  tags = { Name = "loom-platform-rds" }
}
