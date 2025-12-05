# Application Load Balancer
resource "aws_lb" "pli_alb" {
  name               = "${local.app_prefix}-alb"
  internal           = false
  load_balancer_type = "application"
  security_groups    = [aws_security_group.pli_alb_sg.id]

  # Public subnets for the ALB
  subnets = data.aws_subnets.default.ids

  tags = {
    Name = "${local.app_prefix}-alb"
  }
}

# Target Group (PPE app on port 8501)
resource "aws_lb_target_group" "pli_tg" {
  name        = "${local.app_prefix}-tg"
  port        = 8501
  protocol    = "HTTP"
  vpc_id      = data.aws_vpc.default.id
  target_type = "instance"

  health_check {
    protocol            = "HTTP"
    port                = "8501"
    path                = "/" # change to /health later if you add a health endpoint
    healthy_threshold   = 2
    unhealthy_threshold = 2
    timeout             = 5
    interval            = 30
    matcher             = "200-399"
  }

  tags = {
    Name = "${local.app_prefix}-tg"
  }
}

# Attach EC2 instance to Target Group
resource "aws_lb_target_group_attachment" "pli_tg_attachment" {
  target_group_arn = aws_lb_target_group.pli_tg.arn
  target_id        = aws_instance.pli_ec2.id
  port             = 8501
}

# HTTP Listener (port 80)
resource "aws_lb_listener" "pli_http" {
  load_balancer_arn = aws_lb.pli_alb.arn
  port              = 80
  protocol          = "HTTP"

  default_action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.pli_tg.arn
  }
}

output "pli_alb_dns_name" {
  description = "Public DNS name of the PLI app ALB"
  value       = aws_lb.pli_alb.dns_name
}