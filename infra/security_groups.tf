# ALB Security Group
resource "aws_security_group" "pli_alb_sg" {
  name        = "${local.app_prefix}-alb-sg"
  description = "ALB for PPE Streamlit"
  vpc_id      = data.aws_vpc.default.id

  # HTTP from the internet
  ingress {
    description = "HTTP from internet"
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  # Outbound to anywhere (for health checks, etc.)
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "${local.app_prefix}-alb-sg"
  }
}

# EC2 Security Group (no direct public access)
resource "aws_security_group" "pli_web_sg" {
  name        = "${local.app_prefix}-web-sg"
  description = "Security group for PPE EC2 behind ALB"
  vpc_id      = data.aws_vpc.default.id

  # App traffic ONLY from the ALB security group on 8501
  ingress {
    description     = "App traffic from ALB"
    from_port       = 8501
    to_port         = 8501
    protocol        = "tcp"
    security_groups = [aws_security_group.pli_alb_sg.id]
  }

  # Outbound to anywhere (S3, CodeDeploy, OS updates, etc.)
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "${local.app_prefix}-web-sg"
  }
}
