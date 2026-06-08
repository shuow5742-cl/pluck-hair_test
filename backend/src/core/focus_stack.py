"""Focus-stack fusion: N frames at different Z → all-in-focus RGB + height map.

Algorithm:
1. ECC translation alignment to the middle slice (compensates nova5 XY drift).
2. Per-pixel Laplacian-variance sharpness in a local window.
3. argmax across slices → fused RGB (sharpest color) + height_map (mm above tray).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Sequence

import cv2
import numpy as np

logger = logging.getLogger(__name__)

_ECC_CRITERIA = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 1e-4)


@dataclass(frozen=True)
class FocusStackResult:
    fused_rgb: np.ndarray
    height_map_mm: np.ndarray
    sharpness_map: np.ndarray
    aligned_count: int


def fuse_focus_stack(
    frames: Sequence[np.ndarray],
    z_values_mm: Sequence[float],
    z_tray_flange_mm: float,
    *,
    sharpness_kernel: int = 31,
) -> FocusStackResult:
    """Fuse a Z-stack into an all-in-focus image and a height map.

    Parameters
    ----------
    frames : sequence of BGR images (all same shape)
    z_values_mm : nova5 flange Z at each capture (same length as frames)
    z_tray_flange_mm : flange Z when touching tray (baseline)
    sharpness_kernel : local window size for Laplacian variance
    """
    n = len(frames)
    if n == 0:
        raise ValueError("empty frame stack")
    if n != len(z_values_mm):
        raise ValueError("frames and z_values_mm length mismatch")

    if n == 1:
        h_above = abs(z_values_mm[0] - z_tray_flange_mm)
        hmap = np.full(frames[0].shape[:2], h_above, dtype=np.float32)
        sharp = np.ones(frames[0].shape[:2], dtype=np.float32)
        return FocusStackResult(
            fused_rgb=frames[0].copy(),
            height_map_mm=hmap,
            sharpness_map=sharp,
            aligned_count=1,
        )

    ref_idx = n // 2
    ref_gray = cv2.cvtColor(frames[ref_idx], cv2.COLOR_BGR2GRAY)
    h, w = ref_gray.shape

    aligned_frames = []
    aligned_count = 0
    for i, frame in enumerate(frames):
        if i == ref_idx:
            aligned_frames.append(frame)
            aligned_count += 1
            continue
        aligned = _align_to_ref(frame, ref_gray, w, h)
        if aligned is not None:
            aligned_frames.append(aligned)
            aligned_count += 1
        else:
            aligned_frames.append(frame)
            aligned_count += 1
            logger.warning("ECC alignment failed for slice %d, using unaligned", i)

    sharpness_stack = _compute_sharpness_stack(aligned_frames, sharpness_kernel)
    best_idx = np.argmax(sharpness_stack, axis=0)

    fused = np.empty((h, w, 3), dtype=np.uint8)
    height_map = np.empty((h, w), dtype=np.float32)
    sharpness_out = np.empty((h, w), dtype=np.float32)

    for i in range(len(aligned_frames)):
        mask = best_idx == i
        fused[mask] = aligned_frames[i][mask]
        height_map[mask] = abs(z_values_mm[i] - z_tray_flange_mm)
        sharpness_out[mask] = sharpness_stack[i][mask]

    return FocusStackResult(
        fused_rgb=fused,
        height_map_mm=height_map,
        sharpness_map=sharpness_out,
        aligned_count=aligned_count,
    )


def _align_to_ref(
    frame: np.ndarray,
    ref_gray: np.ndarray,
    w: int,
    h: int,
) -> np.ndarray | None:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    warp_matrix = np.eye(2, 3, dtype=np.float32)
    try:
        _, warp_matrix = cv2.findTransformECC(
            ref_gray, gray, warp_matrix, cv2.MOTION_TRANSLATION, _ECC_CRITERIA,
        )
    except cv2.error:
        return None
    aligned = cv2.warpAffine(
        frame, warp_matrix, (w, h), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
    )
    return aligned


def _compute_sharpness_stack(
    frames: list[np.ndarray],
    kernel_size: int,
) -> np.ndarray:
    """Return (N, H, W) float32 array of per-pixel sharpness scores."""
    n = len(frames)
    h, w = frames[0].shape[:2]
    stack = np.empty((n, h, w), dtype=np.float32)
    ksize = kernel_size | 1  # ensure odd
    for i, frame in enumerate(frames):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        lap = cv2.Laplacian(gray, cv2.CV_32F)
        lap_sq = lap * lap
        stack[i] = cv2.blur(lap_sq, (ksize, ksize))
    return stack
