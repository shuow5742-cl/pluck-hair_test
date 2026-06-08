"""YOLO instance segmentation runtime utilities.

This module is intentionally independent from the existing AutoWeaver detection
pipeline.  It loads an Ultralytics YOLO segmentation model, runs inference on a
BGR OpenCV frame, and writes per-target artifacts:

- original image
- annotated image with bbox + mask overlay
- bbox crop
- binary mask crop
- transparent masked crop (BGRA)
- JSON report
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
from ultralytics import YOLO

from src.core.pick_point_estimator import (
    PICK_U_ZERO_OFFSET_DEG,
    build_metal_safety_context,
    estimate_pick_point,
)

logger = logging.getLogger(__name__)


def _normalize_preview_axis_deg(angle_deg: float) -> float:
    while angle_deg >= 90.0:
        angle_deg -= 180.0
    while angle_deg < -90.0:
        angle_deg += 180.0
    return angle_deg


def _pick_u_to_preview_axis_deg(u_deg: float) -> float:
    return _normalize_preview_axis_deg(float(u_deg) - PICK_U_ZERO_OFFSET_DEG)


@dataclass
class SegmentDetection:
    index: int
    class_id: int
    class_name: str
    confidence: float
    bbox_xyxy: list[float]
    center_xy: list[float]
    mask_area: int
    mask_bbox_xyxy: list[int] | None = None
    polygon_xy: list[list[float]] = field(default_factory=list)
    bbox_crop_path: str | None = None
    mask_path: str | None = None
    mask_crop_path: str | None = None
    masked_crop_path: str | None = None
    pick_point_xy: list[float] | None = None
    pick_angle_deg: float | None = None
    pick_method: str | None = None
    pick_score: float | None = None
    object_length_px: float | None = None
    object_width_px: float | None = None
    object_aspect_ratio: float | None = None
    distance_to_edge_px: float | None = None
    distance_to_metal_px: float | None = None
    hair_candidate_area: int | None = None
    extent: float | None = None
    solidity: float | None = None
    shape_class: str | None = None
    preferred_epson_tool: int | None = None


@dataclass
class SegmentFrameResult:
    status: str
    frame_id: str
    image_path: str | None
    annotated_path: str | None
    json_path: str | None
    image_width: int
    image_height: int
    detections: list[SegmentDetection]
    timing_ms: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "frame_id": self.frame_id,
            "image_path": self.image_path,
            "annotated_path": self.annotated_path,
            "json_path": self.json_path,
            "image_width": self.image_width,
            "image_height": self.image_height,
            "timing_ms": self.timing_ms,
            "detections": [asdict(d) for d in self.detections],
        }


class YoloSegmentationRuntime:
    """Runtime wrapper for a trained Ultralytics YOLO segmentation model."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        conf: float = 0.25,
        iou: float = 0.45,
        imgsz: int = 640,
        device: str | None = "auto",
        output_dir: str | Path = "data/segmentation_outputs",
        save_artifacts: bool = True,
        tool_assignment_long_edge_px: float = 50.0,
    ) -> None:
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"YOLO segmentation model not found: {self.model_path}")

        self.conf = float(conf)
        self.iou = float(iou)
        self.imgsz = int(imgsz)
        self.device = None if device in (None, "", "auto") else str(device)
        self.output_dir = Path(output_dir)
        self.save_artifacts = bool(save_artifacts)
        self.tool_assignment_long_edge_px = float(tool_assignment_long_edge_px)

        # task="segment" avoids Ultralytics guessing the wrong task from a custom file name.
        self.model = YOLO(str(self.model_path), task="segment")
        self.names = getattr(self.model, "names", {}) or {}
        logger.info("Loaded YOLO segmentation model: %s", self.model_path)

    def process_frame(
        self,
        frame_bgr: np.ndarray,
        *,
        frame_id: str | None = None,
    ) -> SegmentFrameResult:
        total_start = time.perf_counter()
        if frame_bgr is None or frame_bgr.size == 0:
            raise ValueError("frame_bgr is empty")
        if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
            raise ValueError("frame_bgr must be a BGR image shaped [H, W, 3]")

        frame_id = frame_id or time.strftime("%Y%m%d_%H%M%S") + f"_{int(time.time() * 1000) % 1000:03d}"
        h, w = frame_bgr.shape[:2]

        run_dir = self.output_dir / frame_id
        image_path = annotated_path = json_path = None
        if self.save_artifacts:
            (run_dir / "crops").mkdir(parents=True, exist_ok=True)
            (run_dir / "masks").mkdir(parents=True, exist_ok=True)
            image_path = str(run_dir / "original.png")
            annotated_path = str(run_dir / "annotated.png")
            json_path = str(run_dir / "result.json")
            cv2.imwrite(image_path, frame_bgr)

        predict_kwargs: dict[str, Any] = {
            "source": frame_bgr,
            "conf": self.conf,
            "iou": self.iou,
            "imgsz": self.imgsz,
            "verbose": False,
        }
        if self.device is not None:
            predict_kwargs["device"] = self.device

        predict_start = time.perf_counter()
        results = self.model.predict(**predict_kwargs)
        predict_end = time.perf_counter()
        if not results:
            result = SegmentFrameResult(
                status="no_result",
                frame_id=frame_id,
                image_path=image_path,
                annotated_path=annotated_path,
                json_path=json_path,
                image_width=w,
                image_height=h,
                detections=[],
                timing_ms={
                    "yolo_infer_ms": round((predict_end - predict_start) * 1000.0, 2),
                    "postprocess_ms": 0.0,
                    "artifact_save_ms": 0.0,
                    "seg_total_ms": round((time.perf_counter() - total_start) * 1000.0, 2),
                },
            )
            self._save_empty_outputs(frame_bgr, result)
            return result

        yolo_result = results[0]
        parse_start = time.perf_counter()
        detections, parse_timing = self._parse_result(yolo_result, frame_bgr, run_dir if self.save_artifacts else None)
        parse_end = time.perf_counter()
        status = "ok" if detections else "no_detection"
        result = SegmentFrameResult(
            status=status,
            frame_id=frame_id,
            image_path=image_path,
            annotated_path=annotated_path,
            json_path=json_path,
            image_width=w,
            image_height=h,
            detections=detections,
            timing_ms={},
        )

        artifact_start = time.perf_counter()
        if self.save_artifacts:
            annotated = self.draw_annotated(frame_bgr, detections)
            cv2.imwrite(str(annotated_path), annotated)
            Path(json_path).write_text(
                json.dumps(result.to_dict(), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        artifact_end = time.perf_counter()

        result.timing_ms = {
            "yolo_infer_ms": round((predict_end - predict_start) * 1000.0, 2),
            "postprocess_ms": round((parse_end - parse_start) * 1000.0, 2),
            "artifact_save_ms": round((artifact_end - artifact_start) * 1000.0, 2),
            "seg_total_ms": round((artifact_end - total_start) * 1000.0, 2),
        }
        result.timing_ms.update(parse_timing)

        return result

    def _parse_result(
        self,
        yolo_result: Any,
        frame_bgr: np.ndarray,
        run_dir: Path | None,
    ) -> tuple[list[SegmentDetection], dict[str, Any]]:
        h, w = frame_bgr.shape[:2]
        boxes_obj = getattr(yolo_result, "boxes", None)
        masks_obj = getattr(yolo_result, "masks", None)

        if boxes_obj is None or getattr(boxes_obj, "xyxy", None) is None:
            return [], {}
        if masks_obj is None:
            logger.warning("YOLO result has boxes but no masks; check that the model is a *-seg model")
            return [], {}

        boxes = boxes_obj.xyxy.cpu().numpy()
        scores = boxes_obj.conf.cpu().numpy() if boxes_obj.conf is not None else np.zeros((len(boxes),), dtype=float)
        labels = boxes_obj.cls.cpu().numpy().astype(int) if boxes_obj.cls is not None else np.zeros((len(boxes),), dtype=int)

        mask_data = getattr(masks_obj, "data", None)
        if mask_data is None:
            return [], {}
        masks = mask_data.cpu().numpy()

        polygons_raw: Iterable[np.ndarray] | None = getattr(masks_obj, "xy", None)
        polygons = list(polygons_raw) if polygons_raw is not None else [np.empty((0, 2)) for _ in range(len(boxes))]

        detections: list[SegmentDetection] = []
        pick_timing_totals: dict[str, float] = {}
        pick_timing_rows: list[dict[str, Any]] = []
        count = min(len(boxes), len(masks))

        # Union of all YOLO masks in this frame, full image size. Passed into
        # estimate_pick_point so metal-plate detection can exclude pixels
        # already claimed by YOLO as foreign matter.
        yolo_mask_union = np.zeros((h, w), dtype=np.uint8)
        for i in range(count):
            mfull = self._mask_to_original_size(masks[i], w, h)
            yolo_mask_union[mfull > 0.5] = 1

        metal_safety_context = build_metal_safety_context(
            frame_bgr=frame_bgr,
            yolo_mask_union=yolo_mask_union,
        )

        for i in range(count):
            x1, y1, x2, y2 = self._clip_xyxy(boxes[i], w, h)
            if x2 <= x1 or y2 <= y1:
                continue

            mask_full = self._mask_to_original_size(masks[i], w, h)
            mask_bin = (mask_full > 0.5).astype(np.uint8) * 255
            mask_area = int(np.count_nonzero(mask_bin))
            if mask_area <= 0:
                continue

            mask_bbox = self._mask_bbox(mask_bin)
            class_id = int(labels[i])
            class_name = self._class_name(class_id)
            conf = float(scores[i])
            cx = float((x1 + x2) / 2.0)
            cy = float((y1 + y2) / 2.0)
            pick_result = estimate_pick_point(
                mask_bin,
                [float(x1), float(y1), float(x2), float(y2)],
                frame_bgr=frame_bgr,
                yolo_mask_union=yolo_mask_union,
                metal_safety_context=metal_safety_context,
            )
            pick_row: dict[str, Any] = {
                "det": int(i),
                "class": class_name,
                "shape_class": pick_result.shape_class,
                "pick_method": pick_result.pick_method,
            }
            for key, value in (pick_result.timing_ms or {}).items():
                numeric = round(float(value or 0.0), 2)
                pick_timing_totals[key] = round(float(pick_timing_totals.get(key, 0.0)) + numeric, 2)
                pick_row[key] = numeric
            pick_timing_rows.append(pick_row)

            polygon_xy = []
            if i < len(polygons):
                poly = np.asarray(polygons[i], dtype=float)
                polygon_xy = [[float(px), float(py)] for px, py in poly.reshape(-1, 2)] if poly.size else []

            det = SegmentDetection(
                index=len(detections),
                class_id=class_id,
                class_name=class_name,
                confidence=conf,
                bbox_xyxy=[float(x1), float(y1), float(x2), float(y2)],
                center_xy=[cx, cy],
                mask_area=mask_area,
                mask_bbox_xyxy=mask_bbox,
                polygon_xy=polygon_xy,
                pick_point_xy=pick_result.pick_point_xy,
                pick_angle_deg=pick_result.pick_angle_deg,
                pick_method=pick_result.pick_method,
                pick_score=pick_result.pick_score,
                object_length_px=pick_result.length_px,
                object_width_px=pick_result.width_px,
                object_aspect_ratio=pick_result.aspect_ratio,
                distance_to_edge_px=pick_result.distance_to_edge_px,
                distance_to_metal_px=pick_result.distance_to_metal_px,
                hair_candidate_area=pick_result.hair_candidate_area,
                extent=pick_result.extent,
                solidity=pick_result.solidity,
                shape_class=pick_result.shape_class,
                preferred_epson_tool=_preferred_epson_tool_for_box(
                    x1, y1, x2, y2, self.tool_assignment_long_edge_px
                ),
            )

            if run_dir is not None:
                self._save_detection_artifacts(frame_bgr, mask_bin, x1, y1, x2, y2, det, run_dir)

            detections.append(det)

        parse_timing: dict[str, Any] = {}
        for key, value in (metal_safety_context.timing_ms or {}).items():
            parse_timing[f"shared_{key}"] = round(float(value or 0.0), 2)
        for key, value in pick_timing_totals.items():
            prefixed_key = key if key.startswith("pick_") else f"pick_{key}"
            parse_timing[f"{prefixed_key}_sum"] = round(float(value), 2)
        if pick_timing_rows:
            parse_timing["pick_timing_rows"] = pick_timing_rows
        return detections, parse_timing

    def _save_detection_artifacts(
        self,
        frame_bgr: np.ndarray,
        mask_bin: np.ndarray,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        det: SegmentDetection,
        run_dir: Path,
    ) -> None:
        name = f"target_{det.index:03d}_{det.class_name}_{det.confidence:.3f}"
        bbox_crop = frame_bgr[y1:y2, x1:x2]
        mask_crop = mask_bin[y1:y2, x1:x2]

        # BGRA transparent cutout: target pixels keep BGR, background alpha=0.
        masked_bgra = cv2.cvtColor(bbox_crop, cv2.COLOR_BGR2BGRA)
        masked_bgra[:, :, 3] = mask_crop

        bbox_crop_path = run_dir / "crops" / f"{name}_bbox.png"
        mask_path = run_dir / "masks" / f"{name}_mask_full.png"
        mask_crop_path = run_dir / "masks" / f"{name}_mask_crop.png"
        masked_crop_path = run_dir / "crops" / f"{name}_masked_crop.png"

        cv2.imwrite(str(bbox_crop_path), bbox_crop)
        cv2.imwrite(str(mask_path), mask_bin)
        cv2.imwrite(str(mask_crop_path), mask_crop)
        cv2.imwrite(str(masked_crop_path), masked_bgra)

        det.bbox_crop_path = str(bbox_crop_path)
        det.mask_path = str(mask_path)
        det.mask_crop_path = str(mask_crop_path)
        det.masked_crop_path = str(masked_crop_path)

    def draw_annotated(self, frame_bgr: np.ndarray, detections: list[SegmentDetection]) -> np.ndarray:
        out = frame_bgr.copy()
        overlay = out.copy()
        for det in detections:
            x1, y1, x2, y2 = [int(round(v)) for v in det.bbox_xyxy]
            if det.mask_path and Path(det.mask_path).exists():
                mask = cv2.imread(det.mask_path, cv2.IMREAD_GRAYSCALE)
                if mask is not None:
                    overlay[mask > 0] = (0, 128, 255)
            box_color, tool_label = _tool_style(det)
            cv2.rectangle(out, (x1, y1), (x2, y2), box_color, 2)
            label = f"{det.class_name} {det.confidence:.2f} area={det.mask_area}"
            if tool_label:
                label = f"{tool_label} {label}"
            cv2.putText(out, label, (x1, max(0, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 1, cv2.LINE_AA)
            self._draw_pick_point(out, det)
        if detections:
            out = cv2.addWeighted(overlay, 0.35, out, 0.65, 0)
            for det in detections:
                x1, y1, x2, y2 = [int(round(v)) for v in det.bbox_xyxy]
                box_color, _tool_label = _tool_style(det)
                cv2.rectangle(out, (x1, y1), (x2, y2), box_color, 2)
                self._draw_pick_point(out, det)
        return out

    @staticmethod
    def _draw_pick_point(out: np.ndarray, det: SegmentDetection) -> None:
        if not det.pick_point_xy:
            return
        px, py = [int(round(v)) for v in det.pick_point_xy]
        color = (0, 0, 255)  # red — abstain filtering happens upstream now
        cv2.drawMarker(out, (px, py), color, markerType=cv2.MARKER_CROSS, markerSize=18, thickness=2)
        cv2.circle(out, (px, py), 5, color, thickness=2)
        if det.pick_angle_deg is not None:
            theta = np.deg2rad(_pick_u_to_preview_axis_deg(float(det.pick_angle_deg)))
            dx = int(round(np.cos(theta) * 30))
            # Match the validated field chart used by the Epson U probe:
            # right is 50 deg and larger angles rotate toward image-up.
            dy = int(round(-np.sin(theta) * 30))
            cv2.line(out, (px - dx, py - dy), (px + dx, py + dy), (255, 0, 255), 2, cv2.LINE_AA)
        text = det.pick_method or "pick"
        if det.pick_score is not None:
            text += f" {det.pick_score:.2f}"
        if det.distance_to_metal_px is not None:
            text += f" metal={det.distance_to_metal_px:.0f}px"
        cv2.putText(out, text, (px + 8, max(0, py - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
        coord_text = f"xy=({px},{py})"
        if det.pick_angle_deg is not None:
            coord_text += f" u={float(det.pick_angle_deg):.1f}deg"
        cv2.putText(out, coord_text, (px + 8, py + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    def _save_empty_outputs(self, frame_bgr: np.ndarray, result: SegmentFrameResult) -> None:
        if result.annotated_path:
            cv2.imwrite(result.annotated_path, frame_bgr)
        if result.json_path:
            Path(result.json_path).write_text(
                json.dumps(result.to_dict(), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

    def _class_name(self, class_id: int) -> str:
        if isinstance(self.names, dict):
            return str(self.names.get(class_id, class_id))
        if isinstance(self.names, (list, tuple)) and 0 <= class_id < len(self.names):
            return str(self.names[class_id])
        return str(class_id)

    @staticmethod
    def _clip_xyxy(box: np.ndarray, width: int, height: int) -> tuple[int, int, int, int]:
        x1, y1, x2, y2 = [float(v) for v in box[:4]]
        x1 = max(0, min(width, int(np.floor(x1))))
        y1 = max(0, min(height, int(np.floor(y1))))
        x2 = max(0, min(width, int(np.ceil(x2))))
        y2 = max(0, min(height, int(np.ceil(y2))))
        return x1, y1, x2, y2

    @staticmethod
    def _mask_to_original_size(mask: np.ndarray, width: int, height: int) -> np.ndarray:
        if mask.shape[:2] == (height, width):
            return mask
        return cv2.resize(mask.astype(np.float32), (width, height), interpolation=cv2.INTER_LINEAR)

    @staticmethod
    def _mask_bbox(mask_bin: np.ndarray) -> list[int] | None:
        ys, xs = np.where(mask_bin > 0)
        if len(xs) == 0 or len(ys) == 0:
            return None
        return [int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)]


def _preferred_epson_tool_for_box(
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    long_edge_threshold_px: float,
) -> int:
    long_edge_px = max(float(x2 - x1), float(y2 - y1))
    return 2 if long_edge_px < float(long_edge_threshold_px) else 1


def _tool_style(det: SegmentDetection) -> tuple[tuple[int, int, int], str]:
    tool_code = getattr(det, "preferred_epson_tool", None)
    if tool_code == 2:
        return (0, 0, 255), "suck"
    return (0, 255, 0), "tweezers"
