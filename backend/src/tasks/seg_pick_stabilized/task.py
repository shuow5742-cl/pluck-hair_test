"""SegPickStabilizedTask — seg_pick with multi-frame hit gating."""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any, Optional

from autoweaver.pipeline import VisionPipeline
from autoweaver.tasks import AlwaysFalseCondition, DoneCondition, TaskBase

from src.tasks.stats import TaskStats
from src.tasks.stabilized_detection.stabilizer import StabilizerConfig
from src.types import SegDetection

from .stabilizer import SegDetectionStabilizer

logger = logging.getLogger(__name__)


class SegPickStabilizedTask(TaskBase):
    """Run segmentation pipeline and emit only multi-frame-stable SegDetections."""

    def __init__(
        self,
        pipeline: VisionPipeline,
        *,
        stabilizer_config: Optional[StabilizerConfig] = None,
        done_condition: Optional[DoneCondition] = None,
        stats: Optional[TaskStats] = None,
        name: str = "seg_pick_stabilized",
    ):
        super().__init__()
        self.pipeline = pipeline
        self.done_condition = done_condition or AlwaysFalseCondition()
        self.stats = stats or TaskStats()
        self._name = name
        self._frame_count = 0
        self.stabilizer = SegDetectionStabilizer(config=stabilizer_config)
        self._cycle_frame_index = 0
        self._cycle_frames_total = self.frames_per_resume
        self._cycle_timing_rows: list[dict[str, Any]] = []

    @property
    def name(self) -> str:
        return self._name

    @property
    def frames_per_resume(self) -> int:
        cfg = self.stabilizer.config
        return max(
            1,
            int(max(cfg.window_size, cfg.min_frames_to_stable)),
        )

    def on_resume(self, pipeline_runs: int) -> None:
        # Each parked photo position is its own temporal window. Reset the
        # stabilizer so hits from the previous photo cannot leak into the
        # next one, then consume exactly the run budget frame_loop armed.
        self.stabilizer.reset()
        self._cycle_frame_index = 0
        self._cycle_frames_total = max(1, int(pipeline_runs))
        self._cycle_timing_rows = []

    def run(self, data: Any) -> None:
        image = data
        pipeline_result = self.pipeline.run(image)
        self._frame_count += 1
        self._cycle_frame_index += 1

        raw_seg_dets = [
            d for d in pipeline_result.detections if isinstance(d, SegDetection)
        ]
        stable_seg_dets = self.stabilizer.update(raw_seg_dets)
        picks = [_pick_payload(d) for d in stable_seg_dets]
        timing_row = _build_timing_row(
            pipeline_time_ms=pipeline_result.processing_time_ms,
            step_timing_ms=pipeline_result.metadata.get("step_timing_ms") or {},
            seg_timing_ms=pipeline_result.metadata.get("seg_timing_ms") or {},
            frame_index=self._cycle_frame_index,
        )
        self._cycle_timing_rows.append(timing_row)

        metadata = {
            "pipeline_time_ms": pipeline_result.processing_time_ms,
            "detection_count": len(stable_seg_dets),
            "raw_detection_count": len(pipeline_result.detections),
            "seg_detection_count": len(stable_seg_dets),
            "raw_seg_detection_count": len(raw_seg_dets),
            "stable_seg_detection_count": len(stable_seg_dets),
            "cluster_count": self.stabilizer.get_cluster_count(),
            "frame_in_task": self._frame_count,
            "resume_cycle_frame": self._cycle_frame_index,
            "resume_cycle_total": self._cycle_frames_total,
            "resume_cycle_final": self._cycle_frame_index >= self._cycle_frames_total,
            "seg_frame_id": pipeline_result.metadata.get("seg_frame_id"),
            "step_timing_ms": pipeline_result.metadata.get("step_timing_ms") or {},
            "seg_timing_ms": pipeline_result.metadata.get("seg_timing_ms") or {},
            "crop_single_square": pipeline_result.metadata.get("crop_single_square"),
            "abstain_near_metal": pipeline_result.metadata.get("abstain_near_metal"),
            "preview_only_detections": pipeline_result.metadata.get("preview_only_detections") or [],
            "cycle_timing_rows": list(self._cycle_timing_rows),
            "cycle_pipeline_total_ms": _sum_timing(self._cycle_timing_rows, "pipeline_time_ms"),
            "cycle_yolo_total_ms": _sum_timing(self._cycle_timing_rows, "yolo_infer_ms"),
            "cycle_postprocess_total_ms": _sum_timing(self._cycle_timing_rows, "postprocess_ms"),
            "cycle_crop_total_ms": _sum_timing(self._cycle_timing_rows, "crop_square_ms"),
            "cycle_abstain_total_ms": _sum_timing(self._cycle_timing_rows, "abstain_ms"),
            "cycle_pick_total_ms": _sum_timing(self._cycle_timing_rows, "pick_total_ms"),
            "cycle_pick_metal_detect_total_ms": _sum_timing(self._cycle_timing_rows, "pick_metal_detect_ms"),
            "cycle_pick_dark_line_total_ms": _sum_timing(self._cycle_timing_rows, "pick_dark_line_ms"),
            "cycle_pick_density_total_ms": _sum_timing(self._cycle_timing_rows, "pick_density_thickness_ms"),
            "cycle_pick_straight_total_ms": _sum_timing(self._cycle_timing_rows, "pick_straight_thin_ms"),
            "cycle_pick_curved_total_ms": _sum_timing(self._cycle_timing_rows, "pick_curved_ms"),
            "cycle_pick_distance_total_ms": _sum_timing(self._cycle_timing_rows, "pick_distance_transform_ms"),
        }
        result = SimpleNamespace(
            detections=stable_seg_dets,
            stable_targets=[],
            tracked_targets=[],
            picks=picks,
            is_done=False,
            metadata=metadata,
        )

        is_final_frame = self._cycle_frame_index >= self._cycle_frames_total
        if not is_final_frame:
            return

        logger.info(
            "%s timing window frame_count=%d pipeline_total=%.2fms yolo_total=%.2fms postprocess_total=%.2fms crop_total=%.2fms abstain_total=%.2fms pick_total=%.2fms pick_metal_detect=%.2fms pick_dark_line=%.2fms pick_density=%.2fms pick_straight=%.2fms pick_curved=%.2fms pick_distance=%.2fms rows=%s",
            self._name,
            self._cycle_frames_total,
            float(metadata["cycle_pipeline_total_ms"]),
            float(metadata["cycle_yolo_total_ms"]),
            float(metadata["cycle_postprocess_total_ms"]),
            float(metadata["cycle_crop_total_ms"]),
            float(metadata["cycle_abstain_total_ms"]),
            float(metadata["cycle_pick_total_ms"]),
            float(metadata["cycle_pick_metal_detect_total_ms"]),
            float(metadata["cycle_pick_dark_line_total_ms"]),
            float(metadata["cycle_pick_density_total_ms"]),
            float(metadata["cycle_pick_straight_total_ms"]),
            float(metadata["cycle_pick_curved_total_ms"]),
            float(metadata["cycle_pick_distance_total_ms"]),
            metadata["cycle_timing_rows"],
        )

        self.stats.record(result)
        is_done = self.done_condition.check(self.stats, result)

        self.broadcast(
            "TASK:ITERATION",
            {
                "source": self._name,
                "payload": {
                    "detections": stable_seg_dets,
                    "raw_detections": pipeline_result.detections,
                    "picks": picks,
                    "metadata": metadata,
                },
            },
        )
        if is_done:
            self.broadcast(
                "TASK:DONE",
                {
                    "source": self._name,
                    "payload": {
                        "task_name": self._name,
                        "stats": self.stats.summary(),
                    },
                },
            )

    def reset(self) -> None:
        self._frame_count = 0
        self._cycle_frame_index = 0
        self._cycle_frames_total = self.frames_per_resume
        self._cycle_timing_rows = []
        self.stabilizer.reset()
        self.stats.reset()
        self.done_condition.reset()

    def close(self) -> None:
        super().close()

    def get_stats_summary(self) -> dict:
        return self.stats.summary()


def _pick_payload(d: SegDetection) -> dict[str, Any]:
    return {
        "detection_id": d.detection_id,
        "object_type": d.object_type,
        "confidence": d.confidence,
        "pick_point_xy": d.pick_point_xy,
        "pick_angle_deg": d.pick_angle_deg,
        "pick_method": d.pick_method,
        "pick_score": d.pick_score,
        "shape_class": d.shape_class,
        "distance_to_metal_px": d.distance_to_metal_px,
        "mask_area": d.mask_area,
    }


def _build_timing_row(
    *,
    pipeline_time_ms: float,
    step_timing_ms: dict[str, Any],
    seg_timing_ms: dict[str, Any],
    frame_index: int,
) -> dict[str, Any]:
    return {
        "frame": int(frame_index),
        "pipeline_time_ms": round(float(pipeline_time_ms or 0.0), 2),
        "crop_square_ms": round(float(step_timing_ms.get("crop_square", 0.0) or 0.0), 2),
        "yolo_step_ms": round(float(step_timing_ms.get("segment", 0.0) or 0.0), 2),
        "yolo_infer_ms": round(float(seg_timing_ms.get("yolo_infer_ms", 0.0) or 0.0), 2),
        "postprocess_ms": round(float(seg_timing_ms.get("postprocess_ms", 0.0) or 0.0), 2),
        "seg_total_ms": round(float(seg_timing_ms.get("seg_total_ms", 0.0) or 0.0), 2),
        "abstain_ms": round(float(step_timing_ms.get("abstain", 0.0) or 0.0), 2),
        "pick_total_ms": round(float(seg_timing_ms.get("pick_total_ms_sum", 0.0) or 0.0), 2),
        "pick_metal_detect_ms": round(float(seg_timing_ms.get("pick_metal_detect_ms_sum", 0.0) or 0.0), 2),
        "pick_dark_line_ms": round(float(seg_timing_ms.get("pick_dark_line_ms_sum", 0.0) or 0.0), 2),
        "pick_density_thickness_ms": round(float(seg_timing_ms.get("pick_density_thickness_ms_sum", 0.0) or 0.0), 2),
        "pick_straight_thin_ms": round(float(seg_timing_ms.get("pick_straight_thin_ms_sum", 0.0) or 0.0), 2),
        "pick_curved_ms": round(float(seg_timing_ms.get("pick_curved_ms_sum", 0.0) or 0.0), 2),
        "pick_distance_transform_ms": round(float(seg_timing_ms.get("pick_distance_transform_ms_sum", 0.0) or 0.0), 2),
        "pick_timing_rows": list(seg_timing_ms.get("pick_timing_rows") or []),
    }


def _sum_timing(rows: list[dict[str, Any]], key: str) -> float:
    return round(sum(float(row.get(key, 0.0) or 0.0) for row in rows), 2)
