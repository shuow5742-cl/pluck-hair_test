#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-tools/calibrate/ros2_intrinsic.example.yaml}"
shift || true

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

PY_BIN="${PY_BIN:-}"
if [[ -z "${PY_BIN}" ]]; then
  if [[ -x "${ROOT_DIR}/.venv/bin/python" ]]; then
    PY_BIN="${ROOT_DIR}/.venv/bin/python"
  else
    PY_BIN="python3"
  fi
fi

"${PY_BIN}" -m tools.calibrate.ros2_intrinsic --config "$CONFIG" "$@"
