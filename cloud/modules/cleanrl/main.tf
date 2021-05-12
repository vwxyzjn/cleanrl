resource "aws_batch_job_definition" "cleanrl" {
  name = "cleanrl"
  type = "container"
  container_properties = <<CONTAINER_PROPERTIES
{
    "image": "vwxyzjn/cleanrl:latest",
    "memory": 1024,
    "vcpus": 1
}
CONTAINER_PROPERTIES
}


############
# On-demand resources
############

resource "aws_batch_compute_environment" "gpu_on_demand" {
  compute_environment_name = "gpu_on_demand"
  compute_resources {
    instance_role = aws_iam_instance_profile.ecs_instance_role.arn
    instance_type = [
      "g4dn",
    ]
    max_vcpus = var.max_vcpus
    min_vcpus = 0
    security_group_ids = [
      aws_security_group.sample.id,
    ]
    subnets = data.aws_subnet_ids.all_default_subnets.ids
    type = "EC2"
    allocation_strategy = var.on_demand_allocation_strategy
  }
  service_role = aws_iam_role.aws_batch_service_role.arn
  type         = "MANAGED"
  depends_on   = [aws_iam_role_policy_attachment.aws_batch_service_role]
}

resource "aws_batch_job_queue" "gpu_on_demand" {
  name     = "gpu_on_demand"
  state    = "ENABLED"
  priority = 100
  compute_environments = [
    aws_batch_compute_environment.gpu_on_demand.arn,
  ]
}

resource "aws_batch_compute_environment" "cpu_on_demand" {
  compute_environment_name = "cpu_on_demand"
  compute_resources {
    instance_role = aws_iam_instance_profile.ecs_instance_role.arn
    instance_type = [
      "r5ad",
    ]
    max_vcpus = var.max_vcpus
    min_vcpus = 0
    security_group_ids = [
      aws_security_group.sample.id,
    ]
    subnets = data.aws_subnet_ids.all_default_subnets.ids
    type = "EC2"
    allocation_strategy = var.on_demand_allocation_strategy
  }
  service_role = aws_iam_role.aws_batch_service_role.arn
  type         = "MANAGED"
  depends_on   = [aws_iam_role_policy_attachment.aws_batch_service_role]
}

resource "aws_batch_job_queue" "cpu_on_demand" {
  name     = "cpu_on_demand"
  state    = "ENABLED"
  priority = 100
  compute_environments = [
    aws_batch_compute_environment.cpu_on_demand.arn,
  ]
}

############
# Spot resources
############

resource "aws_batch_compute_environment" "gpu_spot" {
  compute_environment_name = "gpu_spot"
  compute_resources {
    instance_role = aws_iam_instance_profile.ecs_instance_role.arn
    instance_type = [
      "g4dn",
    ]
    max_vcpus = var.max_vcpus
    min_vcpus = 0
    security_group_ids = [
      aws_security_group.sample.id,
    ]
    subnets = data.aws_subnet_ids.all_default_subnets.ids
    type = "SPOT"
    bid_percentage=var.spot_bid_percentage
    allocation_strategy = var.spot_allocation_strategy
    spot_iam_fleet_role = aws_iam_role.AWS_EC2_spot_fleet_role.arn
  }
  service_role = aws_iam_role.aws_batch_service_role.arn
  type         = "MANAGED"
  depends_on   = [aws_iam_role_policy_attachment.aws_batch_service_role]
}

resource "aws_batch_job_queue" "gpu_spot" {
  name     = "gpu_spot"
  state    = "ENABLED"
  priority = 100
  compute_environments = [
    aws_batch_compute_environment.gpu_spot.arn,
  ]
}

resource "aws_batch_compute_environment" "cpu_spot" {
  compute_environment_name = "cpu_spot"
  compute_resources {
    instance_role = aws_iam_instance_profile.ecs_instance_role.arn
    instance_type = [
      "r5ad",
    ]
    max_vcpus = var.max_vcpus
    min_vcpus = 0
    security_group_ids = [
      aws_security_group.sample.id,
    ]
    subnets = data.aws_subnet_ids.all_default_subnets.ids
    type = "SPOT"
    bid_percentage=var.spot_bid_percentage
    allocation_strategy = var.spot_allocation_strategy
    spot_iam_fleet_role = aws_iam_role.AWS_EC2_spot_fleet_role.arn
  }
  service_role = aws_iam_role.aws_batch_service_role.arn
  type         = "MANAGED"
  depends_on   = [aws_iam_role_policy_attachment.aws_batch_service_role]
}

resource "aws_batch_job_queue" "cpu_spot" {
  name     = "cpu_spot"
  state    = "ENABLED"
  priority = 100
  compute_environments = [
    aws_batch_compute_environment.cpu_spot.arn,
  ]
}
