#!/usr/bin/env bash

set -euo pipefail

# Strong-bound Daheng Linux SDK x86 URL (user-requested strategy)
DEFAULT_DAHENG_SDK_URL="https://hs.va-imaging.com/hubfs/Supportshare/Daheng%20SDK/Galaxy_Linux-x86_Gige-U3_32bits-64bits_2.4.2507.9231.zip"

DAHENG_SDK_URL="${DEFAULT_DAHENG_SDK_URL}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --daheng-sdk-url)
      DAHENG_SDK_URL="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      echo "Usage: sudo bash scripts/bootstrap_host_phase2.sh [--daheng-sdk-url URL]" >&2
      exit 1
      ;;
  esac
done

if [[ "$(id -u)" -ne 0 ]]; then
  echo "Please run as root: sudo bash scripts/bootstrap_host_phase2.sh" >&2
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

echo "[phase2] Installing runtime dependencies..."
apt-get update
apt-get install -y \
  ca-certificates \
  curl \
  wget \
  tar \
  unzip \
  usbutils \
  libusb-1.0-0 \
  libglib2.0-0

echo "[phase2] Checking NVIDIA driver..."
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi not found. Run phase1 first and reboot if needed." >&2
  exit 1
fi

echo "[phase2] Installing NVIDIA container toolkit..."
if ! dpkg -s nvidia-container-toolkit >/dev/null 2>&1; then
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
    | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

  curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
    | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
    > /etc/apt/sources.list.d/nvidia-container-toolkit.list

  apt-get update
  apt-get install -y nvidia-container-toolkit
fi

nvidia-ctk runtime configure --runtime=docker
systemctl restart docker

echo "[phase2] Downloading Daheng SDK from fixed URL..."
WORK_DIR="/tmp/daheng-sdk-install"
rm -rf "${WORK_DIR}"
mkdir -p "${WORK_DIR}"

SDK_ARCHIVE="${WORK_DIR}/daheng_sdk.tar.gz"

if [[ "${DAHENG_SDK_URL}" == *"/pages/customerdownloads"* ]]; then
  echo "Current Daheng URL points to the landing page, not direct SDK file:" >&2
  echo "  ${DAHENG_SDK_URL}" >&2
  echo "Please set --daheng-sdk-url to the direct Linux SDK x86 file URL." >&2
  exit 1
fi

if ! curl -fL --retry 3 --connect-timeout 10 --max-time 300 -o "${SDK_ARCHIVE}" "${DAHENG_SDK_URL}"; then
  echo "Failed to download Daheng SDK from URL: ${DAHENG_SDK_URL}" >&2
  echo "Please update --daheng-sdk-url to direct Linux SDK x86 file URL." >&2
  exit 1
fi

echo "[phase2] Extracting Daheng SDK..."
if [[ "${DAHENG_SDK_URL}" == *.zip* ]]; then
  if ! unzip -o "${SDK_ARCHIVE}" -d "${WORK_DIR}" >/dev/null; then
    echo "Downloaded file is not a valid zip archive. URL may not be SDK binary." >&2
    exit 1
  fi
else
  if ! tar -xf "${SDK_ARCHIVE}" -C "${WORK_DIR}"; then
    echo "Downloaded file is not a valid tar archive. URL may not be SDK binary." >&2
    exit 1
  fi
fi

echo "[phase2] Installing Daheng SDK..."
INSTALL_SCRIPT="$(find "${WORK_DIR}" -maxdepth 6 -type f \( -name "Galaxy_camera.run" -o -name "*.run" -o -name "install.sh" \) | head -n1 || true)"

if [[ -z "${INSTALL_SCRIPT}" ]]; then
  echo "Cannot find Daheng SDK installer script in extracted files." >&2
  exit 1
fi

chmod +x "${INSTALL_SCRIPT}"

if [[ "${INSTALL_SCRIPT}" == *.run ]]; then
  "${INSTALL_SCRIPT}" --mode unattended || "${INSTALL_SCRIPT}"
else
  bash "${INSTALL_SCRIPT}"
fi

echo "[phase2] Copying SDK libraries to system paths..."
SDK_LIB_DIR="$(find "${WORK_DIR}" -maxdepth 6 -type d -name "x86_64" | head -n1 || true)"
if [[ -n "${SDK_LIB_DIR}" ]]; then
  for f in "${SDK_LIB_DIR}"/*.cti; do
    if [[ -f "${f}" ]]; then
      cp -v "${f}" /usr/lib/
    fi
  done
fi

echo "[phase2] Refreshing dynamic linker cache..."
ldconfig

echo "[phase2] Verifying SDK libraries..."
for lib in libgxiapi.so liblog4cplus_gx.so; do
  if [[ -f "/usr/lib/${lib}" ]]; then
    echo "[phase2] OK: /usr/lib/${lib}"
  else
    echo "[phase2] WARN: /usr/lib/${lib} not found"
  fi
done
for cti in GxU3VTL.cti GxGVTL.cti; do
  if [[ -f "/usr/lib/${cti}" ]]; then
    echo "[phase2] OK: /usr/lib/${cti}"
  else
    echo "[phase2] WARN: /usr/lib/${cti} not found"
  fi
done

echo "[phase2] Writing udev rule for USB camera access..."
cat >/etc/udev/rules.d/99-daheng-camera.rules <<'EOF'
SUBSYSTEM=="usb", ATTR{idVendor}=="054d", MODE="0666"
SUBSYSTEM=="usb", ATTR{idVendor}=="04b4", MODE="0666"
EOF

udevadm control --reload-rules
udevadm trigger || true

echo "[phase2] Running checks..."
nvidia-smi >/dev/null

if ! docker info --format '{{json .Runtimes}}' | grep -q 'nvidia'; then
  echo "GPU docker check failed: 'nvidia' runtime not found in docker runtimes." >&2
  echo "Verify nvidia-container-toolkit and Docker runtime config (nvidia-ctk runtime configure)." >&2
  exit 1
fi

echo "[phase2] Done. Host is ready for backend deployment."
