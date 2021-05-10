variable "max_vcpus" {
  description = "The maximum number of vcpus in each computing environment"
  type = number
  default = 2000
}

variable "on_demand_allocation_strategy" {
  description = "The allocation strateg for the on-demand computing environment (e.g. `BEST_FIT`, `BEST_FIT_PROGRESSIVE`; see https://docs.aws.amazon.com/batch/latest/userguide/allocation-strategies.html)"
  type = string
  default = "BEST_FIT"
}

variable "spot_allocation_strategy" {
  description = "The allocation strateg for the on-demand computing environment (e.g. `BEST_FIT`, `BEST_FIT_PROGRESSIVE`, or `SPOT_CAPACITY_OPTIMIZED`; see https://docs.aws.amazon.com/batch/latest/userguide/allocation-strategies.html)"
  type = string
  default = "BEST_FIT"
}

variable "spot_bid_percentage" {
  description = "The spot bid percentage"
  type = string
  default = "50"
}