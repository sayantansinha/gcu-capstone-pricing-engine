# OIDC provider for GitHub Actions (once per account)
resource "aws_iam_openid_connect_provider" "github" {
  url             = "https://token.actions.githubusercontent.com"
  client_id_list  = ["sts.amazonaws.com"]
  thumbprint_list = ["6938fd4d98bab03faadb97b34396831e3780aea1"]
}

# Role that GitHub Actions will assume to deploy
resource "aws_iam_role" "gha_deployer" {
  name = "${local.app_prefix}-gha-deployer"
  assume_role_policy = jsonencode({
    Version = "2012-10-17",
    Statement = [
      {
        Effect    = "Allow",
        Principal = { Federated = aws_iam_openid_connect_provider.github.arn },
        Action    = "sts:AssumeRoleWithWebIdentity",
        Condition = {
          StringEquals = {
            "token.actions.githubusercontent.com:aud" = "sts.amazonaws.com"
          },
          StringLike = {
            # limit to your repo and default branches/tags
            "token.actions.githubusercontent.com:sub" = [
              "repo:sayantansinha/gcu-capstone-pricing-engine:ref:refs/heads/main",
              "repo:sayantansinha/gcu-capstone-pricing-engine:ref:refs/tags/*"
            ]
          }
        }
      }
    ]
  })
  tags = local.tags
}

# Policy for uploading artifact to S3 and triggering CodeDeploy
resource "aws_iam_policy" "gha_deploy_policy" {
  name = "${local.app_prefix}-gha-deploy-policy"
  policy = jsonencode({
    Version = "2012-10-17",
    Statement = [
      {
        Sid    = "PutArtifact"
        Effect = "Allow"
        Action = [
          "s3:PutObject",
          "s3:PutObjectAcl"
        ]
        Resource = [
          "arn:aws:s3:::${local.deploy_artifact_bucket}/releases/*"
        ]
      },
      {
        Sid    = "CodeDeployDeploy"
        Effect = "Allow"
        Action = [
          "codedeploy:CreateDeployment",
          "codedeploy:RegisterApplicationRevision",
          "codedeploy:GetDeployment",
          "codedeploy:GetDeploymentConfig",
          "codedeploy:GetApplication",
          "codedeploy:GetDeploymentGroup"
        ]
        Resource = "*"
      }
    ]
  })
  tags = local.tags
}


resource "aws_iam_role_policy_attachment" "gha_attach" {
  role       = aws_iam_role.gha_deployer.name
  policy_arn = aws_iam_policy.gha_deploy_policy.arn
}
