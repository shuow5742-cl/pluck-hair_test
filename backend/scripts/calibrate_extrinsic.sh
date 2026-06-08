#!/usr/bin/env bash
# Extrinsic calibration: live alignment assist
#
# Opens camera with a red crosshair at the principal point.
# Jog the arm until the crosshair aligns with the cross-hair mark,
# then press SPACE and enter the arm position.
#
# Usage:
#   ./scripts/calibrate_extrinsic.sh
#   ./scripts/calibrate_extrinsic.sh --preview-scale 0.4

set -euo pipefail
cd "$(dirname "$0")/.."

python -m tools.calibrate.extrinsic_align \
    --config config/settings.yaml \
    --intrinsic config/calibration/camera_intrinsic.yaml \
    --mm-per-pixel 0.009857 \
    "$@"
