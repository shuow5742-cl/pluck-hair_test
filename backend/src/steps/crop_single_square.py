"""Crop the single bright cell ROI by template-matching a known cell shape.

Why template matching
---------------------
The rig uses a telecentric lens with mm_per_pixel = 0.009857. A grid cell is
physically 10 mm × 10 mm with small corner chamfers, so on-pixel the cell is
~1015 × 1015 px every single time, regardless of which cell we're looking at
or how the lighting falls. That's a very strong prior — we know the target
shape exactly, we just need to find *where* it is in the frame.

So:
1. Binarize gray < `metal_threshold` -> metal=1, cell=0  (just for the score
   image; we still match against the idealized template, not the binarization).
2. Build an idealized binary template: a 1015×1015 white square with the four
   corners chamfered.
3. cv2.matchTemplate(cell_mask, template, TM_CCOEFF_NORMED) gives a response
   map. The maximum response is where the template best aligns with the
   binarized image.
4. Bias the response by distance from the image center (the camera always
   frames its target near center) and pick the argmax.
5. Crop with `border_px/2` padding clipped to image bounds.

Failure handling
----------------
If the best response falls below `min_match_score`, fall back to a centered
1100 px box so downstream segmentation still gets something reasonable.

Contract
--------
- ``ctx.original_image`` stays the full frame.
- ``ctx.processed_image`` becomes the cropped ROI.
- ``ctx.metadata[PROCESSED_ORIGIN_METADATA_KEY]`` = top-left in original-image px.
- ``ctx.metadata[SQUARE_CROP_METADATA_KEY]['source']`` = ``template_match`` or
  ``geometric_fallback``.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from autoweaver.pipeline import PipelineContext, ProcessStep, register_step

logger = logging.getLogger(__name__)

PROCESSED_ORIGIN_METADATA_KEY = "processed_origin_xy"
SQUARE_CROP_METADATA_KEY = "crop_single_square"


@dataclass(frozen=True)
class CropBox:
    left: int
    top: int
    right: int
    bottom: int

    @property
    def width(self) -> int:
        return self.right - self.left

    @property
    def height(self) -> int:
        return self.bottom - self.top


class CropSingleSquareStep(ProcessStep):
    """Find the single cell by matching a known 10×10 mm chamfered template."""

    def __init__(self, params: dict | None = None):
        super().__init__(params)
        # Pixel scale comes from the telecentric lens calibration.
        self.mm_per_pixel = float(self.params.get("mm_per_pixel", 0.009857))
        # Physical cell size, corner chamfer, and metal-frame width (in mm).
        # The metal frame around the cell is included in the template so the
        # match score has both bright-interior and dark-border evidence.
        # Without the dark border, the template is nearly constant and
        # TM_CCOEFF_NORMED scores collapse toward zero.
        self.cell_size_mm = float(self.params.get("cell_size_mm", 10.0))
        self.chamfer_mm = float(self.params.get("chamfer_mm", 0.5))
        self.frame_mm = float(self.params.get("frame_mm", 0.5))
        # Anything darker than this is metal. Calibrated from the real images:
        # metal pixels sit ~30-60, cells >120, so 100 is a comfortable margin
        # that also tolerates dim metal without bridging cells.
        self.metal_threshold = int(self.params.get("metal_threshold", 100))
        # Center-bias: penalize candidates far from frame center, in units of
        # "fraction of the diagonal". 0 disables the bias entirely. The
        # camera always physically frames its target near center, so a mild
        # bias breaks ties between equally good template matches.
        self.center_bias = float(self.params.get("center_bias", 0.15))
        # Minimum normalized match score (TM_CCOEFF_NORMED, range [-1, 1])
        # required to accept a match. Below this we fall back to a centered
        # box. 0.4 is empirical — confident matches typically score >0.6.
        self.min_match_score = float(self.params.get("min_match_score", 0.40))
        # Padding kept around the matched cell; clipped to image bounds.
        self.border_px = int(self.params.get("border_px", 176))
        # Centered fallback box side (px) when no match clears the threshold.
        self.fallback_side_px = int(self.params.get("fallback_side_px", 1100))
        # Cache the template since it only depends on calibration.
        self._template: Optional[np.ndarray] = None

    @property
    def name(self) -> str:
        return self._custom_name or "crop_single_square"

    @property
    def cell_side_px(self) -> int:
        """Side length of the bright cell, in pixels."""
        return int(round(self.cell_size_mm / self.mm_per_pixel))

    @property
    def chamfer_px(self) -> int:
        return max(0, int(round(self.chamfer_mm / self.mm_per_pixel)))

    @property
    def frame_px(self) -> int:
        """Width of the dark metal collar baked into the template, in pixels."""
        return max(0, int(round(self.frame_mm / self.mm_per_pixel)))

    @property
    def template_side_px(self) -> int:
        """Total template side, including the metal collar on both sides."""
        return self.cell_side_px + 2 * self.frame_px

    def build_template(self) -> np.ndarray:
        """Idealized template: white chamfered cell on a black metal collar.

        The collar gives the matcher dark-border evidence so the score
        actually discriminates. Cached after first call.
        """
        if self._template is not None:
            return self._template
        side = self.cell_side_px
        chamfer = self.chamfer_px
        frame = self.frame_px
        total = side + 2 * frame
        tpl = np.zeros((total, total), dtype=np.uint8)
        # Paint the bright cell interior in the middle of the canvas.
        cv2.rectangle(
            tpl,
            (frame, frame),
            (frame + side - 1, frame + side - 1),
            color=255, thickness=-1,
        )
        if chamfer > 0:
            # Cut each corner of the bright square back to dark.
            x0, y0 = frame, frame
            x1, y1 = frame + side - 1, frame + side - 1
            tris = [
                np.array([[x0, y0], [x0 + chamfer, y0], [x0, y0 + chamfer]],
                         np.int32),
                np.array([[x1, y0], [x1 - chamfer, y0], [x1, y0 + chamfer]],
                         np.int32),
                np.array([[x0, y1], [x0 + chamfer, y1], [x0, y1 - chamfer]],
                         np.int32),
                np.array([[x1, y1], [x1 - chamfer, y1], [x1, y1 - chamfer]],
                         np.int32),
            ]
            for tri in tris:
                cv2.fillPoly(tpl, [tri], 0)
        self._template = tpl
        return tpl

    def build_cell_mask(self, image: np.ndarray) -> np.ndarray:
        """Binarize: cells = 255, metal = 0. Public for the probe."""
        gray = _to_gray(image)
        # gray > metal_threshold  -> 255 (cell), else 0 (metal).
        _, mask = cv2.threshold(
            gray, self.metal_threshold, 255, cv2.THRESH_BINARY,
        )
        return mask

    def compute_score_map(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (cell_mask, raw_match_score). Score has shape
        (H - side + 1, W - side + 1), float32, range [-1, 1]."""
        mask = self.build_cell_mask(image)
        template = self.build_template()
        score = cv2.matchTemplate(mask, template, cv2.TM_CCOEFF_NORMED)
        return mask, score

    def process(self, ctx: PipelineContext) -> PipelineContext:
        step_start = time.perf_counter()
        image = ctx.processed_image if ctx.processed_image is not None else ctx.original_image
        if image is None or image.size == 0:
            return ctx

        h, w = image.shape[:2]
        side = self.cell_side_px
        tpl_side = self.template_side_px

        if h < tpl_side or w < tpl_side:
            # Frame smaller than the template — nothing meaningful to match.
            crop_box = self._fallback_centered_box(w, h)
            self._write_metadata(
                ctx, crop_box, source="geometric_fallback",
                square_xyxy=[crop_box.left, crop_box.top, crop_box.right, crop_box.bottom],
                match_score=None,
            )
            _store_step_timing(ctx.metadata, self.name, (time.perf_counter() - step_start) * 1000.0)
            return ctx

        mask, score = self.compute_score_map(image)
        biased = self._apply_center_bias(score, w, h)
        _, max_val, _, max_loc = cv2.minMaxLoc(biased)
        # max_val here is the *biased* score; we report the raw one.
        raw_score = float(score[max_loc[1], max_loc[0]])

        if raw_score < self.min_match_score:
            crop_box = self._fallback_centered_box(w, h)
            source = "geometric_fallback"
            square_xyxy = [crop_box.left, crop_box.top, crop_box.right, crop_box.bottom]
            logger.warning(
                "%s: best match %.3f below threshold %.3f; using centered fallback",
                self.name, raw_score, self.min_match_score,
            )
        else:
            # max_loc is the template's top-left in the image. The cell
            # interior starts `frame_px` inside the template, so shift.
            tx = max_loc[0] + self.frame_px
            ty = max_loc[1] + self.frame_px
            crop_box = self._pad_and_clip(tx, ty, side, side, w, h)
            source = "template_match"
            square_xyxy = [tx, ty, tx + side, ty + side]

        self._write_metadata(
            ctx, crop_box, source=source, square_xyxy=square_xyxy,
            match_score=raw_score,
        )
        _store_step_timing(ctx.metadata, self.name, (time.perf_counter() - step_start) * 1000.0)
        return ctx

    def _apply_center_bias(
        self, score: np.ndarray, image_w: int, image_h: int,
    ) -> np.ndarray:
        """Subtract a Gaussian-ish penalty for being far from frame center."""
        if self.center_bias <= 0.0:
            return score
        tpl_side = self.template_side_px
        sh, sw = score.shape
        # Coordinates of each candidate template's *center* in the full frame.
        ys = np.arange(sh, dtype=np.float32)[:, None] + tpl_side / 2.0
        xs = np.arange(sw, dtype=np.float32)[None, :] + tpl_side / 2.0
        cx, cy = image_w / 2.0, image_h / 2.0
        diag = float(np.hypot(image_w, image_h))
        dist = np.hypot(xs - cx, ys - cy) / diag  # [0, ~0.5]
        return score - self.center_bias * dist.astype(np.float32)

    def _pad_and_clip(
        self, x: int, y: int, cw: int, ch: int, frame_w: int, frame_h: int,
    ) -> CropBox:
        half = self.border_px // 2
        left = max(0, x - half)
        top = max(0, y - half)
        right = min(frame_w, x + cw + half)
        bottom = min(frame_h, y + ch + half)
        if right <= left or bottom <= top:
            right = min(frame_w, left + 1)
            bottom = min(frame_h, top + 1)
        return CropBox(left=left, top=top, right=right, bottom=bottom)

    def _fallback_centered_box(self, frame_w: int, frame_h: int) -> CropBox:
        side = min(self.fallback_side_px, frame_w, frame_h)
        cx, cy = frame_w // 2, frame_h // 2
        left = max(0, cx - side // 2)
        top = max(0, cy - side // 2)
        right = min(frame_w, left + side)
        bottom = min(frame_h, top + side)
        return CropBox(left=left, top=top, right=right, bottom=bottom)

    def _write_metadata(
        self, ctx: PipelineContext, crop_box: CropBox, *,
        source: str, square_xyxy: list[int], match_score: Optional[float],
    ) -> None:
        image = ctx.processed_image if ctx.processed_image is not None else ctx.original_image
        cropped = image[crop_box.top:crop_box.bottom, crop_box.left:crop_box.right].copy()
        base_origin = _get_processed_origin_xy(ctx.metadata)
        origin_x = base_origin[0] + crop_box.left
        origin_y = base_origin[1] + crop_box.top
        ctx.processed_image = cropped
        ctx.metadata[PROCESSED_ORIGIN_METADATA_KEY] = [origin_x, origin_y]
        ctx.metadata[SQUARE_CROP_METADATA_KEY] = {
            "applied": True,
            "source": source,
            "match_score": match_score,
            "square_xyxy_in_input": square_xyxy,
            "square_xyxy_in_original": [
                origin_x + (square_xyxy[0] - crop_box.left),
                origin_y + (square_xyxy[1] - crop_box.top),
                origin_x + (square_xyxy[2] - crop_box.left),
                origin_y + (square_xyxy[3] - crop_box.top),
            ],
            "box_xyxy_in_input": [
                crop_box.left, crop_box.top, crop_box.right, crop_box.bottom,
            ],
            "box_xyxy_in_original": [
                origin_x, origin_y,
                origin_x + crop_box.width, origin_y + crop_box.height,
            ],
            "cropped_size": [crop_box.width, crop_box.height],
            "border_px": self.border_px,
            "frame_px": self.frame_px,
            "cell_side_px": self.cell_side_px,
        }


def _to_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def _store_step_timing(metadata: dict, step_name: str, elapsed_ms: float) -> None:
    timings = metadata.setdefault("step_timing_ms", {})
    if isinstance(timings, dict):
        timings[step_name] = round(float(elapsed_ms), 2)


def _get_processed_origin_xy(metadata: dict) -> tuple[int, int]:
    raw = metadata.get(PROCESSED_ORIGIN_METADATA_KEY)
    if isinstance(raw, (list, tuple)) and len(raw) >= 2:
        return int(raw[0]), int(raw[1])
    return 0, 0


def register() -> None:
    register_step("crop_single_square", CropSingleSquareStep)


register()
