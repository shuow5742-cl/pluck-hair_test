#!/usr/bin/env bash
set -euo pipefail

# Detectron2 bootstrap helper.
# - Ensures .venv exists
# - Exports CUDA/gcc hints (defaults to CUDA 12.1 + gcc-12)
# - Runs `uv sync --preview-features extra-build-dependencies`

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_DIR}"

if [ ! -d ".venv" ]; then
  echo "[detectron2-bootstrap] Creating virtualenv at .venv"
  uv venv .venv
fi

CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.1}"
if [ ! -d "${CUDA_HOME}" ]; then
  echo "[detectron2-bootstrap] WARNING: CUDA_HOME (${CUDA_HOME}) not found. Set CUDA_HOME before running this script." >&2
else
  export CUDA_HOME
  export PATH="${CUDA_HOME}/bin:${PATH}"
  export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
fi

if command -v gcc-12 >/dev/null 2>&1 && command -v g++-12 >/dev/null 2>&1; then
  export CC="${CC:-/usr/bin/gcc-12}"
  export CXX="${CXX:-/usr/bin/g++-12}"
fi

echo "[detectron2-bootstrap] Syncing dependencies (torch/cu121 wheels + detectron2 git)..."
uv sync --preview-features extra-build-dependencies

echo "[detectron2-bootstrap] Done. Activate via: source .venv/bin/activate"
