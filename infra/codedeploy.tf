# App & Deployment Group
resource "aws_codedeploy_app" "pli" {
  name             = local.app_prefix
  compute_platform = "Server"
  tags             = local.tags
}

# Service role for CodeDeploy (managed policy)
resource "aws_iam_role" "pli_codedeploy_service" {
  name = "${local.app_prefix}-codedeploy-svc"
  assume_role_policy = jsonencode({
    Version = "2012-10-17",
    Statement = [
      {
        Effect    = "Allow",
        Principal = { Service = "codedeploy.amazonaws.com" },
        Action    = "sts:AssumeRole"
      }
    ]
  })
  tags = local.tags
}

resource "aws_iam_role_policy_attachment" "pli_codedeploy_managed" {
  role       = aws_iam_role.pli_codedeploy_service.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSCodeDeployRole"
}

resource "aws_codedeploy_deployment_group" "pli_deploy_grp" {
  app_name              = aws_codedeploy_app.pli.name
  deployment_group_name = "${local.app_prefix}-deploy-grp"
  service_role_arn      = aws_iam_role.pli_codedeploy_service.arn

  # target EC2 instances by tags
  ec2_tag_set {
    ec2_tag_filter {
      key   = "CodeDeploy:App"
      type  = "KEY_AND_VALUE"
      value = local.app_prefix
    }

    ec2_tag_filter {
      key   = "CodeDeploy:Group"
      type  = "KEY_AND_VALUE"
      value = "${local.app_prefix}-deploy-grp"
    }
  }

  deployment_style {
    deployment_option = "WITHOUT_TRAFFIC_CONTROL"
    deployment_type   = "IN_PLACE"
  }

  auto_rollback_configuration {
    enabled = true
    events  = ["DEPLOYMENT_FAILURE"]
  }

  tags = local.tags
}
