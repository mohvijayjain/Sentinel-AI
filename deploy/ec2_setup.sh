#!/usr/bin/env bash
# One-time server setup on a fresh Ubuntu 24.04 EC2 instance:
#     bash deploy/ec2_setup.sh
# Installs Docker Engine + the Compose plugin from Docker's official apt
# repository and adds 4 GB of swap. Safe to re-run.

set -euo pipefail

echo "==> Docker Engine + Compose plugin (download.docker.com)"
sudo apt-get update
sudo apt-get install -y ca-certificates curl git
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] \
https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "$VERSION_CODENAME") stable" \
    | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io \
    docker-buildx-plugin docker-compose-plugin
sudo usermod -aG docker "$USER"

echo "==> 4 GB swap (headroom for monitoring runs and image builds)"
if ! swapon --show | grep -q /swapfile; then
    sudo fallocate -l 4G /swapfile
    sudo chmod 600 /swapfile
    sudo mkswap /swapfile
    sudo swapon /swapfile
    echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab > /dev/null
fi

docker_version=$(sudo docker compose version --short)
echo
echo "Done. Docker Compose ${docker_version}."
echo "Log out and SSH in again so 'docker' works without sudo."
