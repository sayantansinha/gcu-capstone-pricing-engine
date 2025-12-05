# # Minimal runtime params; add more whenever you need
# resource "aws_ssm_parameter" "aws_region" {
#   name  = "/${local.app_prefix}/AWS_REGION"
#   type  = "String"
#   value = local.aws_region
#   tags  = local.tags
# }
#
# resource "aws_ssm_parameter" "buckets_csv" {
#   name  = "/${local.app_prefix}/S3_BUCKETS"
#   type  = "String"
#   value = join(",", local.buckets_all)
#   tags  = local.tags
# }

# resource "aws_ssm_parameter" "ppe_admin_password" {
#   name  = "${local.app_prefix}_admin_password"
#   type  = "SecureString"
#   value = var.ppe_admin_password
# }
