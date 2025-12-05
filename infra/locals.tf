#############################################
# identity
#############################################
data "aws_caller_identity" "current" {}

#############################################
# local constants
#############################################
locals {
  # AWS region
  aws_region = "us-west-2"

  # Project naming prefix
  app_prefix = "pli"

  # AMI
  app_ami = "ami-030899d9b6df3345c"

  # Buckets (Free-Tier eligible)
  raw_bucket             = "cap-${local.app_prefix}-raw"
  processed_bucket       = "cap-${local.app_prefix}-processed"
  profiles_bucket        = "cap-${local.app_prefix}-profiles"
  models_bucket          = "cap-${local.app_prefix}-models"
  figures_bucket         = "cap-${local.app_prefix}-figures"
  reports_bucket         = "cap-${local.app_prefix}-reports"
  security_bucket        = "cap-${local.app_prefix}-security"
  predictions_bucket     = "cap-${local.app_prefix}-preds"
  deploy_artifact_bucket = "cap-${local.app_prefix}-deploy-artifacts"

  buckets_all = [
    local.raw_bucket,
    local.processed_bucket,
    local.profiles_bucket,
    local.models_bucket,
    local.figures_bucket,
    local.reports_bucket,
    local.security_bucket,
    local.predictions_bucket,
    local.deploy_artifact_bucket
  ]

  tags = {
    Project = local.app_prefix
    Owner   = data.aws_caller_identity.current.account_id
  }

  # CloudWatch agent configuration (metrics + logs)
  cwagent_config = {
    metrics = {
      metrics_collected = {
        mem = {
          measurement                 = ["mem_used_percent"]
          metrics_collection_interval = 60
        }
        swap = {
          measurement                 = ["swap_used_percent"]
          metrics_collection_interval = 60
        }
      }
      append_dimensions = {
        InstanceId = "$${aws:InstanceId}"
      }
    }

    logs = {
      logs_collected = {
        files = {
          collect_list = [
            {
              file_path       = "/var/log/app/app.log"
              log_group_name  = aws_cloudwatch_log_group.pli-app-lg.name
              log_stream_name = "{instance_id}"
              timezone        = "LOCAL"
            }
          ]
        }
      }
    }
  }

  # Base64 so we can write the JSON in one clean shell command
  cwagent_config_b64 = base64encode(jsonencode(local.cwagent_config))
}

