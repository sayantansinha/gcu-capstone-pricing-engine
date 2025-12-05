# # Allow SSM to run a one-time install command after apply
# resource "aws_ssm_document" "install_codedeploy_agent" {
#   name          = "${local.app_prefix}-install-codedeploy"
#   document_type = "Command"
#   content = jsonencode({
#     schemaVersion = "2.2",
#     description   = "Install CodeDeploy agent on AL2023",
#     mainSteps = [
#       {
#         action = "aws:runShellScript",
#         name   = "install",
#         inputs = {
#           runCommand = [
#             "dnf -y install ruby wget || true",
#             "cd /tmp",
#             "wget https://aws-codedeploy-${local.aws_region}.s3.${local.aws_region}.amazonaws.com/latest/install",
#             "chmod +x ./install",
#             "./install auto",
#             "systemctl enable codedeploy-agent",
#             "systemctl restart codedeploy-agent",
#             "systemctl status codedeploy-agent || true"
#           ]
#         }
#       }
#     ]
#   })
#   tags = local.tags
# }
#
# resource "aws_ssm_association" "install_codedeploy_to_instance" {
#   name = aws_ssm_document.install_codedeploy_agent.name
#
#   targets {
#     key    = "InstanceIds"
#     values = [aws_instance.pli_ec2.id]
#   }
# }
