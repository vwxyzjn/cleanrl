############
# On-demand resources
############

resource "aws_batch_compute_environment" "on_demand" {
  count                    = length(var.instance_types)
  compute_environment_name = replace(var.instance_types[count.index], ".", "-")
  compute_resources {
    instance_role = aws_iam_instance_profile.ecs_instance_role.arn
    instance_type = [
      var.instance_types[count.index],
    ]
    max_vcpus = var.max_vcpus
    min_vcpus = 0
    security_group_ids = [
      aws_security_group.sample.id,
    ]
    subnets             = data.aws_subnet_ids.all_default_subnets.ids
    type                = "EC2"
    allocation_strategy = var.on_demand_allocation_strategy
  }
  service_role = aws_iam_role.aws_batch_service_role.arn
  type         = "MANAGED"
  depends_on   = [aws_iam_role_policy_attachment.aws_batch_service_role]
}

resource "aws_batch_job_queue" "on_demand" {
  count    = length(var.instance_types)
  name     = replace(var.instance_types[count.index], ".", "-")
  state    = "ENABLED"
  priority = 100
  compute_environments = [
    aws_batch_compute_environment.on_demand[count.index].arn,
  ]
}

############
# Spot resources
############

resource "aws_batch_compute_environment" "spot" {
  count                    = length(var.instance_types)
  compute_environment_name = replace("${var.instance_types[count.index]}-spot", ".", "-")
  compute_resources {
    instance_role = aws_iam_instance_profile.ecs_instance_role.arn
    instance_type = [
      var.instance_types[count.index],
    ]
    max_vcpus = var.max_vcpus
    min_vcpus = 0
    security_group_ids = [
      aws_security_group.sample.id,
    ]
    subnets             = data.aws_subnet_ids.all_default_subnets.ids
    type                = "SPOT"
    bid_percentage      = var.spot_bid_percentage
    allocation_strategy = var.spot_allocation_strategy
    spot_iam_fleet_role = aws_iam_role.AWS_EC2_spot_fleet_role.arn
  }
  service_role = aws_iam_role.aws_batch_service_role.arn
  type         = "MANAGED"
  depends_on   = [aws_iam_role_policy_attachment.aws_batch_service_role]
}

resource "aws_batch_job_queue" "spot" {
  count    = length(var.instance_types)
  name     = replace("${var.instance_types[count.index]}-spot", ".", "-")
  state    = "ENABLED"
  priority = 100
  compute_environments = [
    aws_batch_compute_environment.spot[count.index].arn,
  ]
}
