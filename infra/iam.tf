# EC2 assume-role trust
data "aws_iam_policy_document" "pli_ec2_trust" {
  statement {
    actions = ["sts:AssumeRole"]

    principals {
      type        = "Service"
      identifiers = ["ec2.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "pli_ec2_role" {
  name               = "${local.app_prefix}-ec2-role"
  assume_role_policy = data.aws_iam_policy_document.pli_ec2_trust.json
  tags               = local.tags
}

resource "aws_iam_instance_profile" "pli_ec2_profile" {
  name = "${local.app_prefix}-ec2-instance-profile"
  role = aws_iam_role.pli_ec2_role.name
}

# Tight S3 RW policy for just your buckets
data "aws_iam_policy_document" "pli_s3_rw" {
  statement {
    actions   = ["s3:ListAllMyBuckets"]
    resources = ["*"]
  }

  statement {
    actions   = ["s3:GetBucketLocation", "s3:ListBucket"]
    resources = [for b in local.buckets_all : "arn:aws:s3:::${b}"]
  }

  statement {
    actions = [
      "s3:GetObject",
      "s3:PutObject",
      "s3:DeleteObject",
      "s3:AbortMultipartUpload",
      "s3:ListBucketMultipartUploads"
    ]
    resources = [for b in local.buckets_all : "arn:aws:s3:::${b}/*"]
  }
}

resource "aws_iam_policy" "pli_s3_rw" {
  name   = "${local.app_prefix}-s3-rw-policy"
  policy = data.aws_iam_policy_document.pli_s3_rw.json
  tags   = local.tags
}

# CloudWatch IAM Policy
resource "aws_iam_policy" "pli_cloudwatch_logs_policy" {
  name        = "${local.app_prefix}-cw-logs-policy"
  description = "Allow EC2 to send PLI app logs to CloudWatch Logs"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "logs:CreateLogStream",
          "logs:PutLogEvents",
          "logs:DescribeLogStreams"
        ]
        Resource = "${aws_cloudwatch_log_group.pli-app-lg.arn}:*"
      },
      {
        Effect = "Allow"
        Action = [
          "logs:DescribeLogGroups"
        ]
        Resource = "*"
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "pli_s3_rw_attach" {
  role       = aws_iam_role.pli_ec2_role.name
  policy_arn = aws_iam_policy.pli_s3_rw.arn
}

# Managed helper policies
resource "aws_iam_role_policy_attachment" "pli_ssm_core" {
  role       = aws_iam_role.pli_ec2_role.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore"
}

resource "aws_iam_role_policy_attachment" "pli_ssm_read" {
  role       = aws_iam_role.pli_ec2_role.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSSMReadOnlyAccess"
}

resource "aws_iam_role_policy_attachment" "pli_cloudwatch_agent" {
  role       = aws_iam_role.pli_ec2_role.name
  policy_arn = "arn:aws:iam::aws:policy/CloudWatchAgentServerPolicy"
}

resource "aws_iam_role_policy_attachment" "pli_cloudwatch_logs" {
  role       = aws_iam_role.pli_ec2_role.name
  policy_arn = aws_iam_policy.pli_cloudwatch_logs_policy.arn
}