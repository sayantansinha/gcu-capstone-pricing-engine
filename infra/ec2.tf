resource "aws_instance" "pli_ec2" {
  ami                         = local.app_ami
  instance_type               = "t3.small"
  subnet_id                   = data.aws_subnets.default.ids[0]
  associate_public_ip_address = true
  iam_instance_profile        = aws_iam_instance_profile.pli_ec2_profile.name
  vpc_security_group_ids      = [aws_security_group.pli_web_sg.id]

  tags = merge(local.tags, {
    "CodeDeploy:App"   = local.app_prefix
    "CodeDeploy:Group" = "${local.app_prefix}-deploy-grp"
    Name               = "${local.app_prefix}-ec2"
  })
}
