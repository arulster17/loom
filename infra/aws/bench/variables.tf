variable "region" {
  description = "Region for the bucket, instance role, reaper and GPU hosts."
  type        = string
  default     = "us-east-1"
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
