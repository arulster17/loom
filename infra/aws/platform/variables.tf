variable "domain" {
  description = "The site's domain: the apex serves the web app, api. the gateway, hooks. the Stripe webhook. No default: terraform plan asks for it (docs/phase1-m3-spec.md §6.11)."
  type        = string

  validation {
    condition     = can(regex("^([a-z0-9]([a-z0-9-]{0,61}[a-z0-9])?[.])+[a-z]{2,}$", var.domain))
    error_message = "domain must be a lower-case DNS name such as a registered domain, without a scheme, a path or a trailing dot."
  }
}

variable "brand_name" {
  description = "The product name users see (web app, sign-in mail sender, Checkout line item). No default: terraform plan asks for it (§6.11)."
  type        = string

  validation {
    condition     = length(trimspace(var.brand_name)) > 0 && length(var.brand_name) <= 60
    error_message = "brand_name must be 1 to 60 characters."
  }

  # loom-deploy refuses these too (§6.11): they would pass apply and then fail every deploy.
  validation {
    condition     = !can(regex("[\\x00-\\x1f,;:@()\\[\\]<>\"\\\\]", var.brand_name))
    error_message = "brand_name must not contain control characters or any of , ; : @ ( ) [ ] < > \" \\ (§6.11)."
  }
}

variable "releases_bucket" {
  description = "Name of the releases S3 bucket (releases/<sha>.json and releases/<sha>/). S3 names are global, so there is no default (§5.6)."
  type        = string

  validation {
    condition     = can(regex("^[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]$", var.releases_bucket))
    error_message = "releases_bucket must be a valid S3 bucket name (3 to 63 lower-case letters, digits, dots or hyphens)."
  }
}

variable "ops_email" {
  description = "Where the alarm topic and the budget send mail. The SNS subscription must be confirmed from this mailbox."
  type        = string

  validation {
    condition     = can(regex("^[^@[:space:]]+@[^@[:space:]]+$", var.ops_email))
    error_message = "ops_email must be an email address."
  }
}

variable "budget_usd" {
  description = "Monthly AWS cost budget in USD (§15 D12); mail at 50%, 80% and 100% actual and 100% forecast."
  type        = number

  validation {
    condition     = var.budget_usd > 0
    error_message = "budget_usd must be greater than zero."
  }
}

variable "route53_zone_id" {
  description = "Route 53 hosted zone of the domain. When set, Terraform writes the DNS records; when null, the same records are the dns_records output for the user's DNS host."
  type        = string
  default     = null
}

variable "serving_reaper_enabled" {
  description = "Turns on the serving reaper's 15-minute schedule and the P1 alarms' actions. Stays false until M4; the reaper runs in dry-run mode either way."
  type        = bool
  default     = false
}

variable "instance_type" {
  description = "EC2 instance type of the platform box (Graviton: the images are arm64)."
  type        = string
  default     = "t4g.medium"
}

variable "db_deletion_protection" {
  description = "RDS deletion protection. Set false and apply before terraform destroy (§6.10)."
  type        = bool
  default     = true
}

variable "create_github_oidc_provider" {
  description = "Create the account's GitHub OIDC identity provider. Set false when the account already has one (an account can hold only one per URL); it is then looked up."
  type        = bool
  default     = true
}

variable "region" {
  description = "Region of the stack. Must be us-east-1: the H3 alarm's metrics exist only there and its action must be in the same region (§8.5; healthcheck.tf checks it)."
  type        = string
  default     = "us-east-1"
}
