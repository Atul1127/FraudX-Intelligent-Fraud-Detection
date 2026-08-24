variable "aws_region" {
  description = "AWS region for FraudX"
  type        = string
  default     = "ap-south-1"
}

variable "mongodb_uri" {
  description = "MongoDB connection URI for the production deployment"
  type        = string
  sensitive   = true
}
