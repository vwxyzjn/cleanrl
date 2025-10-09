variable "max_vcpus" {
  description = "The maximum number of vcpus in each computing environment"
  type        = number
  default     = 2000
}

variable "on_demand_allocation_strategy" {
  description = "The allocation strateg for the on-demand computing environment (e.g. `BEST_FIT`, `BEST_FIT_PROGRESSIVE`; see https://docs.aws.amazon.com/batch/latest/userguide/allocation-strategies.html)"
  type        = string
  default     = "BEST_FIT"
}

variable "spot_allocation_strategy" {
  description = "The allocation strateg for the on-demand computing environment (e.g. `BEST_FIT`, `BEST_FIT_PROGRESSIVE`, or `SPOT_CAPACITY_OPTIMIZED`; see https://docs.aws.amazon.com/batch/latest/userguide/allocation-strategies.html)"
  type        = string
  default     = "BEST_FIT"
}

variable "spot_bid_percentage" {
  description = "The spot bid percentage"
  type        = string
  default     = "50"
}

variable "instance_types" {
  type = list(string)
  default = [
    "g4dn.4xlarge", # 16 vCPU, 64GB, $1.204, GPU
    "g4dn.xlarge",  # 4 vCPU, 16GB, $0.526, GPU
    "r5ad.large",   # 2 vCPU, 16GB, $0.131
    "c6g.medium",   # 1 vCPU, 2GB, $0.034
    # ARM-based
    "a1.medium",   # 1 vCPU, 2GB, $0.0255
    "m6gd.medium", # 1 vCPU, 4GB, $0.0452
  ]
}
