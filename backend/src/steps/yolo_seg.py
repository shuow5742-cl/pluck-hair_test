"""YOLO instance-segmentation pipeline step.

Wraps ``src.core.instance_segmenter.YoloSegmentationRuntime`` so it plugs into
autoweaver 0.4.0's VisionPipeline. The step:

- Pushes ``SegDetection`` instances into ``ctx.detections``. Downstream tasks
  iterate the standard way and can promote-isinstance to read pick fields
  (pick_point_xy / pick_angle_deg / shape_class / mask paths).
- Surfaces the annotated visualization path on ``ctx.metadata`` for preview.
- Reprojects mask + annotated artifacts back to original-frame coordinates when
  the upstream crop step set a non-zero origin, so every geometric field on the
  emitted SegDetection (and the saved overlay PNG) lives in the same coordinate
  system.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np

from autoweaver.pipeline import (
    BoundingBox,
    PipelineContext,
    ProcessStep,
    register_step,
)

from src.steps.crop_single_square import PROCESSED_ORIGIN_METADATA_KEY
from src.types import SegDetection

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from src.core.instance_segmenter import SegmentDetection, YoloSegmentationRuntime  # noqa: F401


class YOLOSegStep(ProcessStep):
    """Run YOLOv8m-seg on ctx.processed_image, emit SegDetection per target."""

    def __init__(self, params: dict | None = None):
        super().__init__(params)

        self.model_path = self.params.get("model", "assets/best_foreigh_segment_yolov8m_seg.pt")
        self.conf = float(self.params.get("conf", 0.25))
        self.iou = float(self.params.get("iou", 0.45))
        self.imgsz = int(self.params.get("imgsz", 640))
        self.device = self.params.get("device", "auto")
        self.output_dir = self.params.get("output_dir", "data/segmentation_outputs")
        self.save_artifacts = bool(self.params.get("save_artifacts", False))
        self.tool_assignment_long_edge_px = float(
            self.params.get("tool_assignment_long_edge_px", 50.0)
        )

        self._runtime: Any = None
        self._frame_counter = 0

    @property
    def name(self) -> str:
        return self._custom_name or "yolo_seg"

    @property
    def runtime(self):
        if self._runtime is None:
            from src.core.instance_segmenter import YoloSegmentationRuntime

            self._runtime = YoloSegmentationRuntime(
                model_path=self.model_path,
                conf=self.conf,
                iou=self.iou,
                imgsz=self.imgsz,
                device=self.device,
                output_dir=self.output_dir,
                save_artifacts=self.save_artifacts,
                tool_assignment_long_edge_px=self.tool_assignment_long_edge_px,
            )
        return self._runtime

    def process(self, ctx: PipelineContext) -> PipelineContext:
        step_start = time.perf_counter()
        image = ctx.processed_image if ctx.processed_image is not None else ctx.original_image
        if image is None or image.size == 0:
            return ctx

        origin_xy = _get_processed_origin_xy(ctx.metadata)
        self._frame_counter += 1
        frame_id = f"{int(time.time())}_{self._frame_counter:06d}"

        result = self.runtime.process_frame(image, frame_id=frame_id)

        full_frame = ctx.original_image if ctx.original_image is not None else image
        if origin_xy != (0, 0) and result.detections:
            _reproject_mask_artifacts_to_full_frame(
                result.detections,
                origin_xy=origin_xy,
                full_frame_shape=full_frame.shape[:2],
            )

        seg_detections = _to_seg_detections(result.detections, origin_xy=origin_xy)

        if origin_xy != (0, 0) and result.annotated_path:
            _rewrite_annotated_on_full_frame(
                annotated_path=result.annotated_path,
                full_frame_bgr=full_frame,
                seg_detections=seg_detections,
            )

        ctx.detections.extend(seg_detections)
        ctx.metadata["seg_status"] = result.status
        ctx.metadata["seg_frame_id"] = result.frame_id
        ctx.metadata["seg_input_origin_xy"] = [origin_xy[0], origin_xy[1]]
        ctx.metadata["seg_timing_ms"] = dict(result.timing_ms or {})
        if result.annotated_path:
            ctx.metadata["seg_annotated_path"] = result.annotated_path
        if result.json_path:
            ctx.metadata["seg_json_path"] = result.json_path
        _store_step_timing(
            ctx.metadata,
            self.name,
            (time.perf_counter() - step_start) * 1000.0,
        )

        return ctx


def _store_step_timing(metadata: dict, step_name: str, elapsed_ms: float) -> None:
    timings = metadata.setdefault("step_timing_ms", {})
    if isinstance(timings, dict):
        timings[step_name] = round(float(elapsed_ms), 2)


def _to_seg_detections(
    seg_dets: list[Any],
    *,
    origin_xy: tuple[int, int] = (0, 0),
) -> list[SegDetection]:
    out: list[SegDetection] = []
    offset_x, offset_y = origin_xy
    for sd in seg_dets:
        x1, y1, x2, y2 = sd.bbox_xyxy
        out.append(
            SegDetection(
                bbox=BoundingBox(
                    x1=float(x1 + offset_x),
                    y1=float(y1 + offset_y),
                    x2=float(x2 + offset_x),
                    y2=float(y2 + offset_y),
                ),
                object_type=sd.class_name,
                confidence=float(sd.confidence),
                detection_id=f"seg_{sd.index:03d}",
                center_xy=_shift_xy(sd.center_xy, origin_xy),
                polygon_xy=[_shift_xy(p, origin_xy) for p in sd.polygon_xy],
                mask_bbox_xyxy=_shift_bbox(sd.mask_bbox_xyxy, origin_xy),
                mask_area=int(sd.mask_area),
                mask_path=sd.mask_path,
                mask_crop_path=sd.mask_crop_path,
                bbox_crop_path=sd.bbox_crop_path,
                masked_crop_path=sd.masked_crop_path,
                pick_point_xy=_shift_optional_xy(sd.pick_point_xy, origin_xy),
                pick_angle_deg=sd.pick_angle_deg,
                pick_method=sd.pick_method,
                pick_score=sd.pick_score,
                distance_to_metal_px=sd.distance_to_metal_px,
                distance_to_edge_px=sd.distance_to_edge_px,
                preferred_epson_tool=sd.preferred_epson_tool,
                shape_class=sd.shape_class,
                object_length_px=sd.object_length_px,
                object_width_px=sd.object_width_px,
                object_aspect_ratio=sd.object_aspect_ratio,
                extent=sd.extent,
                solidity=sd.solidity,
                hair_candidate_area=sd.hair_candidate_area,
            )
        )
    return out


def _get_processed_origin_xy(metadata: dict) -> tuple[int, int]:
    raw = metadata.get(PROCESSED_ORIGIN_METADATA_KEY)
    if isinstance(raw, (list, tuple)) and len(raw) >= 2:
        return int(raw[0]), int(raw[1])
    return 0, 0


def _shift_xy(values: list[float], origin_xy: tuple[int, int]) -> list[float]:
    return [float(values[0] + origin_xy[0]), float(values[1] + origin_xy[1])]


def _shift_optional_xy(
    values: list[float] | None,
    origin_xy: tuple[int, int],
) -> list[float] | None:
    if values is None:
        return None
    return _shift_xy(values, origin_xy)


def _shift_bbox(
    values: list[int] | None,
    origin_xy: tuple[int, int],
) -> list[int] | None:
    if values is None:
        return None
    return [
        int(values[0] + origin_xy[0]),
        int(values[1] + origin_xy[1]),
        int(values[2] + origin_xy[0]),
        int(values[3] + origin_xy[1]),
    ]


def _reproject_mask_artifacts_to_full_frame(
    seg_dets: list[Any],
    *,
    origin_xy: tuple[int, int],
    full_frame_shape: tuple[int, int],
) -> None:
    """Rewrite mask_path PNGs on disk to full-frame size with the mask pasted at origin_xy.

    Without this, the saved mask is at crop-image size while the SegDetection's
    mask_bbox / polygon / bbox have already been shifted to full-frame
    coordinates — using them together would mis-align by `origin_xy`.
    """
    full_h, full_w = full_frame_shape
    ox, oy = origin_xy
    for sd in seg_dets:
        mask_path = getattr(sd, "mask_path", None)
        if not mask_path:
            continue
        path = Path(mask_path)
        if not path.exists():
            continue
        crop_mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if crop_mask is None:
            continue
        canvas = np.zeros((full_h, full_w), dtype=np.uint8)
        ch, cw = crop_mask.shape[:2]
        x1 = max(0, ox)
        y1 = max(0, oy)
        x2 = min(full_w, ox + cw)
        y2 = min(full_h, oy + ch)
        if x2 <= x1 or y2 <= y1:
            continue
        sx1 = x1 - ox
        sy1 = y1 - oy
        sx2 = sx1 + (x2 - x1)
        sy2 = sy1 + (y2 - y1)
        canvas[y1:y2, x1:x2] = crop_mask[sy1:sy2, sx1:sx2]
        cv2.imwrite(str(path), canvas)


def _rewrite_annotated_on_full_frame(
    *,
    annotated_path: str,
    full_frame_bgr: np.ndarray,
    seg_detections: list[SegDetection],
) -> None:
    """Replace the runtime-saved crop-coord overlay with one drawn on the full frame.

    Uses the same rendering recipe as ``tools.machine_result_snapshot`` so the
    audit image and the live machine_result snapshot stay visually consistent.
    """
    try:
        from tools.machine_result_snapshot import render_annotated
    except Exception as exc:  # noqa: BLE001
        logger.warning("annotated reproject skipped: %s", exc)
        return
    overlay = render_annotated(full_frame_bgr, seg_detections)
    cv2.imwrite(str(annotated_path), overlay)


def register() -> None:
    """Register YOLOSegStep with autoweaver under type name ``yolo_seg``."""
    register_step("yolo_seg", YOLOSegStep)


# Auto-register on import so ``from src.steps import yolo_seg`` is enough.
register()
