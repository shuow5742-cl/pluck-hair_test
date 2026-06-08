#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

ROS_DISTRO_NAME="${ROS_DISTRO:-}"
UBUNTU_CODENAME="${UBUNTU_CODENAME:-}"
SKIP_APT_UPDATE=0
SKIP_PYTHON=0
CHECK_ONLY=0
REPO_CHANGED=0

if [[ "${EUID}" -eq 0 ]]; then
  APT=(apt-get)
  RUN_ROOT=()
else
  APT=(sudo apt-get)
  RUN_ROOT=(sudo)
fi

run_root() {
  "${RUN_ROOT[@]}" "$@"
}

usage() {
  cat <<'EOF'
Usage:
  bash scripts/setup_calibration_env.sh [options]

Options:
  --ros-distro <name>   ROS distro (example: humble/jazzy). Default: $ROS_DISTRO
  --skip-apt-update     Skip the initial apt-get update (post-repo-change update still runs)
  --skip-python         Skip Python dependency setup
  --check-only          Only run environment checks, do not install anything
  -h, --help            Show this help

Notes:
  - This script auto-configures ROS2 apt source on Ubuntu (idempotent).
  - It is safe to run repeatedly (idempotent).
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --ros-distro)
      ROS_DISTRO_NAME="${2:-}"
      shift 2
      ;;
    --skip-apt-update)
      SKIP_APT_UPDATE=1
      shift
      ;;
    --skip-python)
      SKIP_PYTHON=1
      shift
      ;;
    --check-only)
      CHECK_ONLY=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 2
      ;;
  esac
done

if [[ -z "${UBUNTU_CODENAME}" && -f /etc/os-release ]]; then
  # shellcheck source=/dev/null
  source /etc/os-release
  UBUNTU_CODENAME="${UBUNTU_CODENAME:-${VERSION_CODENAME:-}}"
fi

if [[ -z "${UBUNTU_CODENAME}" ]]; then
  echo "[setup] Cannot detect Ubuntu codename from /etc/os-release." >&2
  echo "[setup] Set UBUNTU_CODENAME manually before running this script." >&2
  exit 1
fi

if [[ -z "${ROS_DISTRO_NAME}" ]]; then
  if [[ -d /opt/ros ]]; then
    mapfile -t distros < <(find /opt/ros -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort)
    if [[ "${#distros[@]}" -eq 1 ]]; then
      ROS_DISTRO_NAME="${distros[0]}"
      echo "[setup] Auto-detected ROS distro: ${ROS_DISTRO_NAME}"
    elif [[ "${#distros[@]}" -gt 1 ]]; then
      echo "[setup] Multiple ROS distros found: ${distros[*]}" >&2
      echo "[setup] Please specify one with --ros-distro <name>." >&2
      exit 1
    fi
  fi
fi

if [[ -z "${ROS_DISTRO_NAME}" ]]; then
  case "${UBUNTU_CODENAME}" in
    jammy)
      ROS_DISTRO_NAME="humble"
      ;;
    noble)
      ROS_DISTRO_NAME="jazzy"
      ;;
    *)
      ROS_DISTRO_NAME=""
      ;;
  esac
  if [[ -n "${ROS_DISTRO_NAME}" ]]; then
    echo "[setup] Auto-selected ROS distro ${ROS_DISTRO_NAME} for Ubuntu ${UBUNTU_CODENAME}."
  fi
fi

if [[ -z "${ROS_DISTRO_NAME}" ]]; then
  echo "[setup] ROS distro not set. Export ROS_DISTRO or pass --ros-distro." >&2
  exit 1
fi

ROS_SETUP="/opt/ros/${ROS_DISTRO_NAME}/setup.bash"

PACKAGES=(
  "ros-${ROS_DISTRO_NAME}-ros-base"
  "ros-${ROS_DISTRO_NAME}-v4l2-camera"
  "ros-${ROS_DISTRO_NAME}-camera-calibration"
  "v4l-utils"
)

ensure_ros2_apt_source() {
  local keyring_path="/usr/share/keyrings/ros-archive-keyring.gpg"
  local source_list="/etc/apt/sources.list.d/ros2.list"
  local arch repo_line

  arch="$(dpkg --print-architecture)"
  repo_line="deb [arch=${arch} signed-by=${keyring_path}] http://packages.ros.org/ros2/ubuntu ${UBUNTU_CODENAME} main"

  if [[ ! -f "${keyring_path}" ]]; then
    local tmp_key
    tmp_key="$(mktemp)"
    echo "[setup] Downloading ROS GPG key"
    curl -fsSL "https://raw.githubusercontent.com/ros/rosdistro/master/ros.key" -o "${tmp_key}"
    run_root gpg --dearmor -o "${keyring_path}" "${tmp_key}"
    rm -f "${tmp_key}"
    REPO_CHANGED=1
  fi

  if [[ ! -f "${source_list}" ]] || ! grep -Fqx "${repo_line}" "${source_list}"; then
    echo "[setup] Writing ROS apt source: ${source_list}"
    echo "${repo_line}" | run_root tee "${source_list}" >/dev/null
    REPO_CHANGED=1
  fi
}

if [[ "${CHECK_ONLY}" -eq 0 ]]; then
  if ! command -v apt-get >/dev/null 2>&1; then
    echo "[setup] apt-get not found. This script currently supports Debian/Ubuntu only." >&2
    exit 1
  fi

  if [[ "${EUID}" -ne 0 ]] && ! command -v sudo >/dev/null 2>&1; then
    echo "[setup] sudo is required when not running as root." >&2
    exit 1
  fi

  if [[ "${SKIP_APT_UPDATE}" -eq 0 ]]; then
    echo "[setup] apt-get update (base repos)"
    "${APT[@]}" update
  fi

  echo "[setup] Installing apt prerequisites"
  "${APT[@]}" install -y ca-certificates curl gnupg lsb-release

  ensure_ros2_apt_source

  if [[ "${SKIP_APT_UPDATE}" -eq 0 || "${REPO_CHANGED}" -eq 1 ]]; then
    echo "[setup] apt-get update (with ROS repo)"
    "${APT[@]}" update
  fi

  echo "[setup] Installing ROS calibration dependencies for ${ROS_DISTRO_NAME}"
  "${APT[@]}" install -y "${PACKAGES[@]}"

  if [[ "${SKIP_PYTHON}" -eq 0 ]]; then
    echo "[setup] Installing Python dependencies (project: ${ROOT_DIR})"
    if command -v uv >/dev/null 2>&1; then
      (cd "${ROOT_DIR}" && uv sync --locked)
    else
      if [[ ! -x "${ROOT_DIR}/.venv/bin/python" ]]; then
        python3 -m venv "${ROOT_DIR}/.venv"
      fi
      "${ROOT_DIR}/.venv/bin/python" -m pip install --upgrade pip
      (cd "${ROOT_DIR}" && "${ROOT_DIR}/.venv/bin/pip" install -e .)
    fi
  fi
fi

if [[ ! -f "${ROS_SETUP}" ]]; then
  echo "[check] ROS setup not found after install: ${ROS_SETUP}" >&2
  exit 1
fi

echo "[check] Sourcing ${ROS_SETUP}"
set +u
# shellcheck source=/dev/null
source "${ROS_SETUP}"
set -u

if ! command -v ros2 >/dev/null 2>&1; then
  echo "[check] ros2 command not found after sourcing ROS setup." >&2
  exit 1
fi

if ! ros2 pkg executables camera_calibration 2>/dev/null | grep -q "cameracalibrator"; then
  echo "[check] camera_calibration/cameracalibrator not found." >&2
  exit 1
fi

echo "[check] camera_calibration is ready."
echo "[next] Run intrinsic calibration:"
echo "  bash tools/calibrate/ros2_intrinsic_calibrate.sh tools/calibrate/ros2_intrinsic.example.yaml"
