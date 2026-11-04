variable "region" {
  description = "Primary region: the bucket, the reaper Lambdas and the first GPU host region (IAM, so the instance role, is global)."
  type        = string
  default     = "us-east-1"
}

variable "extra_host_regions" {
  description = "More regions GPU hosts may launch in when var.region has no capacity, in sweep order. Each gets a no-ingress security group in its default VPC, and the reaper and runner policy cover it. Supported: us-east-2, us-west-2 (regions.tf)."
  type        = list(string)
  default     = ["us-east-2", "us-west-2"]

  validation {
    condition = alltrue([
      for r in var.extra_host_regions : contains(["us-east-2", "us-west-2"], r)
    ]) && length(distinct(var.extra_host_regions)) == length(var.extra_host_regions)
    error_message = "extra_host_regions may list us-east-2 and us-west-2, each at most once (add a provider alias and module in regions.tf for another)."
  }
}

variable "name_prefix" {
  description = "Prefix for every resource name."
  type        = string
  default     = "loom-bench"

  validation {
    condition     = can(regex("^[a-z0-9][a-z0-9-]{1,30}$", var.name_prefix))
    error_message = "name_prefix must be 2-31 lowercase letters, digits or hyphens."
  }
}

variable "owner" {
  description = "Value for the loom:owner tag (who to ask about these resources)."
  type        = string

  validation {
    condition     = can(regex("^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$", var.owner))
    error_message = "owner must be letters, digits, '.', '_' or '-'."
  }
}

variable "hf_token_secret_name" {
  description = "Name of the existing Secrets Manager secret holding the Hugging Face token (plain string)."
  type        = string
  default     = "loom/hf-token"
}

variable "allowed_instance_types" {
  description = "Instance types the runner may launch."
  type        = list(string)
  default = [
    "g5.xlarge",
    "g6.xlarge",
    "g6e.xlarge",
    "g6e.2xlarge",
    "g6e.12xlarge",
  ]
}

variable "runs_expiry_days" {
  description = "Days after which objects under runs/ and ssm/ in the bucket expire."
  type        = number
  default     = 30
}

variable "reaper_max_age_hours" {
  description = "Age after which a managed resource with a missing or unreadable loom:ttl is reaped."
  type        = number
  default     = 24
}

variable "reaper_dry_run" {
  description = "Log what the reaper would delete without deleting it."
  type        = bool
  default     = false
}

variable "runpod_reaper_secret_name" {
  description = "Name of the Secrets Manager secret Terraform creates (empty) for the RunPod reaper's API key."
  type        = string
  default     = "loom/runpod-reaper-api-key"
}

variable "runpod_reaper_dry_run" {
  description = "Log which Loom RunPod pods and volumes the RunPod reaper would delete, without deleting them. Starts true; set false once a dry-run invoke looks right."
  type        = bool
  default     = true
}

variable "runpod_reaper_max_per_run" {
  description = "When more Loom RunPod resources than this are expired in one run, the RunPod reaper deletes none and fails (a circuit breaker)."
  type        = number
  default     = 10

  validation {
    condition     = var.runpod_reaper_max_per_run >= 1 && floor(var.runpod_reaper_max_per_run) == var.runpod_reaper_max_per_run
    error_message = "runpod_reaper_max_per_run must be a positive whole number."
  }
}
