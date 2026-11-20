variable "state_bucket" {
  description = "Name of the remote-state bucket. S3 names are global, so there is no default."
  type        = string

  validation {
    condition     = can(regex("^[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]$", var.state_bucket))
    error_message = "state_bucket must be a valid S3 bucket name (3 to 63 lower-case letters, digits, dots or hyphens)."
  }
}

variable "region" {
  description = "Region of the bucket."
  type        = string
  default     = "us-east-1"
}
