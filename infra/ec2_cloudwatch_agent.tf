# Allow CloudWatch to run a one-time install command after apply
resource "aws_ssm_document" "install_cloudwatch_agent" {
  name          = "${local.app_prefix}-install-cwagent"
  document_type = "Command"

  content = jsonencode({
    schemaVersion = "2.2"
    description   = "Configure CloudWatch Agent on PLI app EC2"
    mainSteps = [
      {
        action = "aws:runShellScript"
        name   = "installAndConfigure"
        inputs = {
          runCommand = [
            # Turn on strict + verbose bash
            "set -euxo pipefail",

            # Log everything we do to a debug log
            "exec > >(tee -a /var/log/cwagent_setup.log) 2>&1",

            "echo '--- Starting CloudWatch Agent config ---'",

            "echo 'Creating config directory...'",
            "mkdir -p /opt/aws/amazon-cloudwatch-agent/etc",

            "echo 'Writing JSON config from base64...'",
            # This is where the file should be created/overwritten
            "echo '${local.cwagent_config_b64}' | base64 -d > /opt/aws/amazon-cloudwatch-agent/etc/amazon-cloudwatch-agent.json",

            "echo 'Listing config directory contents:'",
            "ls -l /opt/aws/amazon-cloudwatch-agent/etc",

            "echo 'Config file contents:'",
            "cat /opt/aws/amazon-cloudwatch-agent/etc/amazon-cloudwatch-agent.json || echo '*** FAILED TO READ CONFIG FILE ***'",

            "echo 'Starting CloudWatch Agent with JSON config...'",
            "/opt/aws/amazon-cloudwatch-agent/bin/amazon-cloudwatch-agent-ctl -a fetch-config -m ec2 -c file:/opt/aws/amazon-cloudwatch-agent/etc/amazon-cloudwatch-agent.json -s",

            "echo '--- CloudWatch Agent setup complete ---'"
          ]
        }
      }
    ]
  })

  tags = local.tags
}

resource "aws_ssm_association" "install_cloudwatch_agent_to_instance" {
  name = aws_ssm_document.install_cloudwatch_agent.name

  targets {
    key    = "InstanceIds"
    values = [aws_instance.pli_ec2.id]
  }
}

