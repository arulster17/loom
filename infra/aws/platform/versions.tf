terraform {
  required_version = ">= 1.10" # S3 native state locking (backend.tf)

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 6.0"
    }
    archive = {
      source  = "hashicorp/archive"
      version = "~> 2.4"
    }
  }
}

provider "aws" {
  region = var.region

  default_tags {
    tags = {
      "loom:stack" = "platform"
    }
  }
}

# Route 53 publishes health-check metrics only in us-east-1, so alarm H3 is created
# through this alias (healthcheck.tf, docs/phase1-m3-spec.md §8.5).
provider "aws" {
  alias  = "us_east_1"
  region = "us-east-1"

  default_tags {
    tags = {
      "loom:stack" = "platform"
    }
  }
}
