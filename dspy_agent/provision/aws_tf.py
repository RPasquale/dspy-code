from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Optional


def write_terraform_bundle(out_dir: Path, *, project: str, region: str, instance_type: str, volume_size_gb: int) -> Dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)

    versions_tf = """
terraform {
  required_version = ">= 1.3.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = ">= 5.0"
    }
  }
}
"""

    provider_tf = f"""
provider "aws" {{
  region = "{region}"
}}
"""

    variables_tf = f"""
variable "project" {{
  type    = string
  default = "{project}"
}}

variable "instance_type" {{
  type    = string
  default = "{instance_type}"
}}

variable "volume_size" {{
  type    = number
  default = {int(volume_size_gb)}
}}
"""

    # Simple EC2 + S3 for artifacts; user_data installs Docker
    main_tf = """
data "aws_ami" "ubuntu" {
  most_recent = true
  owners      = ["099720109477"] # Canonical
  filter {
    name   = "name"
    values = ["ubuntu/images/hvm-ssd/ubuntu-jammy-22.04-amd64-server-*"]
  }
}

resource "aws_s3_bucket" "agent_logs" {
  bucket = "${var.project}-dspy-agent-logs"
}

resource "aws_security_group" "agent_sg" {
  name        = "${var.project}-agent-sg"
  description = "Allow SSH and dashboard"
  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }
  ingress {
    from_port   = 8081
    to_port     = 8081
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }
  ingress {
    from_port   = 8765
    to_port     = 8765
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

resource "aws_instance" "agent" {
  ami                    = data.aws_ami.ubuntu.id
  instance_type          = var.instance_type
  vpc_security_group_ids = [aws_security_group.agent_sg.id]

  root_block_device {
    volume_size = var.volume_size
  }

  user_data = <<-EOT
    #!/bin/bash
    set -eux
    apt-get update -y
    apt-get install -y docker.io docker-compose git
    systemctl enable docker
    systemctl start docker
    # Placeholder: clone repo and run lightweight stack (user completes setup)
  EOT

  tags = {
    Name = "${var.project}-dspy-agent"
  }
}

output "agent_public_ip" {
  value = aws_instance.agent.public_ip
}

output "logs_bucket" {
  value = aws_s3_bucket.agent_logs.id
}
"""

    files = {
        "versions.tf": versions_tf,
        "provider.tf": provider_tf,
        "variables.tf": variables_tf,
        "main.tf": main_tf,
    }

    for name, content in files.items():
        (out_dir / name).write_text(content)

    readme = f"""
    AWS Terraform bundle for {project}
    
    Steps:
      1) Ensure AWS credentials are configured (AWS_PROFILE/AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY)
      2) terraform -chdir={out_dir} init
      3) terraform -chdir={out_dir} apply
      4) Note the 'agent_public_ip' output, then SSH and run the lightweight stack.
    
    Security note: tighten security groups in production.
    """
    (out_dir / "README.txt").write_text(readme)
    return {"dir": str(out_dir)}

