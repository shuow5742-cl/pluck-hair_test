#!/usr/bin/env bash

set -euo pipefail

INSTALL_NVIDIA_DRIVER=1
TARGET_PROJECT_TORCH_CUDA="12.8"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-nvidia-driver)
      INSTALL_NVIDIA_DRIVER=0
      shift
      ;;
    *)
      echo "Unknown argument: $1" >&2
      echo "Usage: sudo bash scripts/bootstrap_host_phase1.sh [--skip-nvidia-driver]" >&2
      exit 1
      ;;
  esac
done

if [[ "$(id -u)" -ne 0 ]]; then
  echo "Please run as root: sudo bash scripts/bootstrap_host_phase1.sh" >&2
  exit 1
fi

if [[ ! -f /etc/os-release ]]; then
  echo "Unsupported OS: /etc/os-release not found" >&2
  exit 1
fi

source /etc/os-release

if [[ "${ID:-}" != "ubuntu" ]]; then
  echo "This script currently targets Ubuntu. Detected: ${ID:-unknown}" >&2
  exit 1
fi

echo "[phase1] Installing base packages..."
apt-get update
apt-get install -y \
  ca-certificates \
  curl \
  gnupg \
  lsb-release \
  software-properties-common \
  apt-transport-https \
  git \
  jq \
  unzip \
  tar \
  build-essential \
  python3 \
  python3-pip \
  python3-venv

echo "[phase1] Installing Docker Engine (if needed)..."
if ! command -v docker >/dev/null 2>&1; then
  install -m 0755 -d /etc/apt/keyrings
  curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /etc/apt/keyrings/docker.gpg
  chmod a+r /etc/apt/keyrings/docker.gpg

  echo \
    "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
    ${VERSION_CODENAME} stable" > /etc/apt/sources.list.d/docker.list

  apt-get update
  apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
else
  echo "[phase1] Docker already installed, skipping"
fi

systemctl enable docker
systemctl restart docker

TARGET_USER="${SUDO_USER:-}"
if [[ -n "${TARGET_USER}" ]]; then
  usermod -aG docker "${TARGET_USER}" || true
  echo "[phase1] Added ${TARGET_USER} to docker group (re-login required)."
fi

REBOOT_REQUIRED=0

if [[ "${INSTALL_NVIDIA_DRIVER}" -eq 1 ]]; then
  echo "[phase1] Checking NVIDIA driver..."
  if command -v nvidia-smi >/dev/null 2>&1; then
    echo "[phase1] NVIDIA driver already available: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n1)"
  else
    echo "[phase1] Installing NVIDIA driver via ubuntu-drivers..."
    apt-get install -y ubuntu-drivers-common
    ubuntu-drivers autoinstall
    REBOOT_REQUIRED=1
  fi
else
  echo "[phase1] Skipping NVIDIA driver installation by request."
fi

echo "[phase1] Done."
echo "[phase1] Project baseline requires Torch CUDA runtime: ${TARGET_PROJECT_TORCH_CUDA} (from torch cu128)."
if [[ "${REBOOT_REQUIRED}" -eq 1 ]]; then
  echo "[phase1] Reboot is required before running phase2."
else
  echo "[phase1] You can continue with phase2 now."
fi
