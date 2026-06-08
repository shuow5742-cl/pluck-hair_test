"""AbstainNearMetalStep — drop detections whose pick point is too close to metal.

This is an independent filter step in the segmentation pipeline. After
``yolo_seg`` populates each ``SegDetection`` with ``distance_to_metal_px``,
this step removes any detection whose pick point sits within
``safety_margin_px`` pixels of the metal pressing plate. Downstream tasks
see a clean detection list — no half-state "kept but flagged unsafe" entries.

Composability rationale: each filter (size, confidence, ROI, near-metal,
…) is a separate ProcessStep that runs in the configured order. Adding /
removing / reordering filters is a yaml change, not a code change.
"""

from __future__ import annotations

import cv2
import numpy as np
import time
from dataclasses import replace

from autoweaver.pipeline import PipelineContext, ProcessStep, register_step

from src.steps.crop_single_square import SQUARE_CROP_METADATA_KEY
from src.types import SegDetection


class AbstainNearMetalStep(ProcessStep):
    """Filter out SegDetections whose pick point is within safety_margin_px of metal.

    Params:
        safety_margin_px: minimum required distance from pick point to nearest
            metal pixel. Defaults to 26 to match the picker's internal
            ``metal_safety_margin_px`` so the abstain decision is consistent
            with what the picker considers "safe".
    """

    def __init__(self, params: dict | None = None):
        super().__init__(params)
        self.safety_margin_px = float(self.params.get("safety_margin_px", 26))
        self.enforce_safe_box = bool(self.params.get("enforce_safe_box", True))
        self.require_trusted_crop = bool(self.params.get("require_trusted_crop", True))
        self.trusted_match_score = float(self.params.get("trusted_match_score", 0.48))
        self.edge_strip_px = int(self.params.get("edge_strip_px", 18))
        self.edge_dark_max_gray = float(self.params.get("edge_dark_max_gray", 110.0))
        self.edge_contrast_min = float(self.params.get("edge_contrast_min", 18.0))
        self.min_dark_edge_pass_count = int(self.params.get("min_dark_edge_pass_count", 3))

    @property
    def name(self) -> str:
        return self._custom_name or "abstain_near_metal"

    def process(self, ctx: PipelineContext) -> PipelineContext:
        step_start = time.perf_counter()
        crop_check = _assess_crop_safety(
            ctx,
            safety_margin_px=self.safety_margin_px,
            require_trusted_crop=self.require_trusted_crop,
            trusted_match_score=self.trusted_match_score,
            edge_strip_px=self.edge_strip_px,
            edge_dark_max_gray=self.edge_dark_max_gray,
            edge_contrast_min=self.edge_contrast_min,
            min_dark_edge_pass_count=self.min_dark_edge_pass_count,
        )
        kept: list = []
        abstained_ids: list[str] = []
        out_of_safe_box_ids: list[str] = []
        too_close_to_metal_ids: list[str] = []
        preview_only_detections: list[SegDetection] = []
        crop_guard_rejected = False
        safe_box_xyxy = crop_check.get("safe_box_xyxy")
        if crop_check.get("crop_valid") is False:
            crop_guard_rejected = True
            for d in ctx.detections:
                if isinstance(d, SegDetection):
                    abstained_ids.append(d.detection_id or "")
            ctx.detections[:] = []
            ctx.metadata["abstain_near_metal"] = {
                "safety_margin_px": float(self.safety_margin_px),
                "abstained_count": len(abstained_ids),
                "safe_box_xyxy": safe_box_xyxy,
                **crop_check,
            }
            if abstained_ids:
                ctx.metadata["abstained_ids"] = abstained_ids
                ctx.metadata["abstained_count"] = len(abstained_ids)
            _store_step_timing(ctx.metadata, self.name, (time.perf_counter() - step_start) * 1000.0)
            return ctx
        for d in ctx.detections:
            if isinstance(d, SegDetection) and self.enforce_safe_box and safe_box_xyxy is not None:
                if not _pick_point_inside_safe_box(d, safe_box_xyxy):
                    abstained_ids.append(d.detection_id or "")
                    out_of_safe_box_ids.append(d.detection_id or "")
                    preview_only_detections.append(_preview_only_detection(d))
                    continue
            if isinstance(d, SegDetection) and d.distance_to_metal_px is not None:
                if d.distance_to_metal_px < self.safety_margin_px:
                    abstained_ids.append(d.detection_id or "")
                    too_close_to_metal_ids.append(d.detection_id or "")
                    continue
            kept.append(d)
        ctx.detections[:] = kept
        # Surface the live safety margin for downstream renderers / audits.
        # Keeps the preview overlay consistent with whatever yaml override
        # is in effect, without having the renderer re-read configs.
        ctx.metadata["abstain_near_metal"] = {
            "safety_margin_px": float(self.safety_margin_px),
            "abstained_count": len(abstained_ids),
            "out_of_safe_box_count": len(out_of_safe_box_ids),
            "too_close_to_metal_count": len(too_close_to_metal_ids),
            "safe_box_xyxy": safe_box_xyxy,
            "crop_guard_rejected": crop_guard_rejected,
            **crop_check,
        }
        if abstained_ids:
            ctx.metadata["abstained_ids"] = abstained_ids
            ctx.metadata["abstained_count"] = len(abstained_ids)
        if out_of_safe_box_ids:
            ctx.metadata["out_of_safe_box_ids"] = out_of_safe_box_ids
        if too_close_to_metal_ids:
            ctx.metadata["too_close_to_metal_ids"] = too_close_to_metal_ids
        if preview_only_detections:
            ctx.metadata["preview_only_detections"] = preview_only_detections
        _store_step_timing(ctx.metadata, self.name, (time.perf_counter() - step_start) * 1000.0)
        return ctx


def _preview_only_detection(det: SegDetection) -> SegDetection:
    return replace(
        det,
        pick_point_xy=None,
        pick_angle_deg=None,
        pick_method=None,
        pick_score=None,
        world_xy=None,
    )


def _store_step_timing(metadata: dict, step_name: str, elapsed_ms: float) -> None:
    timings = metadata.setdefault("step_timing_ms", {})
    if isinstance(timings, dict):
        timings[step_name] = round(float(elapsed_ms), 2)


def _assess_crop_safety(
    ctx: PipelineContext,
    *,
    safety_margin_px: float,
    require_trusted_crop: bool,
    trusted_match_score: float,
    edge_strip_px: int,
    edge_dark_max_gray: float,
    edge_contrast_min: float,
    min_dark_edge_pass_count: int,
) -> dict:
    crop_meta = ctx.metadata.get(SQUARE_CROP_METADATA_KEY)
    if not isinstance(crop_meta, dict):
        return {
            "crop_guard_applied": False,
            "crop_valid": True,
            "crop_guard_reason": "no_crop_metadata",
            "safe_box_xyxy": None,
        }

    box = _parse_box(crop_meta.get("box_xyxy_in_original"))
    safe_box = _inset_box(box, safety_margin_px) if box is not None else None
    source = str(crop_meta.get("source") or "")
    match_score = crop_meta.get("match_score")
    frame_px = int(crop_meta.get("frame_px") or 0)
    square = _parse_box(crop_meta.get("square_xyxy_in_original"))

    result = {
        "crop_guard_applied": True,
        "crop_valid": True,
        "crop_guard_reason": "ok",
        "safe_box_xyxy": list(safe_box) if safe_box is not None else None,
        "crop_source": source,
        "crop_match_score": float(match_score) if match_score is not None else None,
        "crop_dark_edge_pass_count": None,
    }

    if safe_box is None:
        result["crop_valid"] = False
        result["crop_guard_reason"] = "invalid_safe_box"
        return result

    if source != "template_match":
        result["crop_valid"] = not require_trusted_crop
        result["crop_guard_reason"] = "geometric_fallback"
        return result

    dark_edge_pass_count = _count_dark_edges(
        image=ctx.original_image,
        square_xyxy=square,
        frame_px=frame_px,
        edge_strip_px=edge_strip_px,
        edge_dark_max_gray=edge_dark_max_gray,
        edge_contrast_min=edge_contrast_min,
    )
    result["crop_dark_edge_pass_count"] = dark_edge_pass_count

    score_ok = match_score is not None and float(match_score) >= float(trusted_match_score)
    edges_ok = dark_edge_pass_count >= int(min_dark_edge_pass_count)
    if require_trusted_crop and not (score_ok or edges_ok):
        result["crop_valid"] = False
        result["crop_guard_reason"] = "weak_crop_match"
    elif not score_ok and edges_ok:
        result["crop_guard_reason"] = "trusted_by_edge_check"
    elif score_ok and not edges_ok:
        result["crop_guard_reason"] = "trusted_by_match_score"
    return result


def _parse_box(raw) -> tuple[int, int, int, int] | None:
    if not isinstance(raw, (list, tuple)) or len(raw) < 4:
        return None
    try:
        x1, y1, x2, y2 = (int(round(float(v))) for v in raw[:4])
    except (TypeError, ValueError):
        return None
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _inset_box(
    box: tuple[int, int, int, int] | None,
    inset_px: float,
) -> tuple[int, int, int, int] | None:
    if box is None:
        return None
    inset = int(round(float(inset_px)))
    x1, y1, x2, y2 = box
    sx1, sy1 = x1 + inset, y1 + inset
    sx2, sy2 = x2 - inset, y2 - inset
    if sx2 <= sx1 or sy2 <= sy1:
        return None
    return sx1, sy1, sx2, sy2


def _pick_point_inside_safe_box(
    det: SegDetection,
    safe_box_xyxy: list[int] | None,
) -> bool:
    if safe_box_xyxy is None:
        return True
    pick_xy = det.pick_point_xy
    if not pick_xy or len(pick_xy) < 2:
        return False
    px = float(pick_xy[0])
    py = float(pick_xy[1])
    x1, y1, x2, y2 = safe_box_xyxy
    return x1 <= px <= x2 and y1 <= py <= y2


def _count_dark_edges(
    *,
    image,
    square_xyxy: tuple[int, int, int, int] | None,
    frame_px: int,
    edge_strip_px: int,
    edge_dark_max_gray: float,
    edge_contrast_min: float,
) -> int:
    if image is None or getattr(image, "size", 0) == 0 or square_xyxy is None:
        return 0
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    x1, y1, x2, y2 = square_xyxy
    strip = max(6, min(int(edge_strip_px), max(frame_px, 6)))
    passes = 0
    for inner_box, outer_box in _edge_strip_boxes(x1, y1, x2, y2, strip, w, h):
        if inner_box is None or outer_box is None:
            continue
        ix1, iy1, ix2, iy2 = inner_box
        ox1, oy1, ox2, oy2 = outer_box
        inner = gray[iy1:iy2, ix1:ix2]
        outer = gray[oy1:oy2, ox1:ox2]
        if inner.size == 0 or outer.size == 0:
            continue
        inner_mean = float(np.mean(inner))
        outer_mean = float(np.mean(outer))
        if outer_mean <= edge_dark_max_gray and (inner_mean - outer_mean) >= edge_contrast_min:
            passes += 1
    return passes


def _edge_strip_boxes(
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    strip: int,
    width: int,
    height: int,
):
    yield (
        _clip_box(x1, y1, x2, min(y1 + strip, y2), width, height),
        _clip_box(x1, max(0, y1 - strip), x2, y1, width, height),
    )
    yield (
        _clip_box(x1, max(y2 - strip, y1), x2, y2, width, height),
        _clip_box(x1, y2, x2, min(height, y2 + strip), width, height),
    )
    yield (
        _clip_box(x1, y1, min(x1 + strip, x2), y2, width, height),
        _clip_box(max(0, x1 - strip), y1, x1, y2, width, height),
    )
    yield (
        _clip_box(max(x2 - strip, x1), y1, x2, y2, width, height),
        _clip_box(x2, y1, min(width, x2 + strip), y2, width, height),
    )


def _clip_box(
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    width: int,
    height: int,
) -> tuple[int, int, int, int] | None:
    x1 = max(0, min(width, x1))
    y1 = max(0, min(height, y1))
    x2 = max(0, min(width, x2))
    y2 = max(0, min(height, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def register() -> None:
    register_step("abstain_near_metal", AbstainNearMetalStep)


register()
