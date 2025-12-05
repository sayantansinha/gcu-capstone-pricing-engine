resource "aws_cloudwatch_log_group" "pli-app-lg" {
  name              = "${local.app_prefix}-app-lg"
  retention_in_days = 1
  tags              = local.tags
}
