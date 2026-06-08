"""SegPickTask — consumes SegDetection from the pipeline, broadcasts pick payload.

This is the minimal task that bridges the segmentation pipeline to the rest
of the system. It runs the configured VisionPipeline (currently expected to
contain a single yolo_seg step), iterates ``pipeline_result.detections``,
isolates the ``SegDetection`` entries, and broadcasts one TASK:ITERATION
event per frame carrying the picked points + angles.

Coordinate transform (pixel → world) is intentionally NOT done here. That's
the responsibility of a separate downstream task in the chain, configured
later. SegPickTask only emits image-space picks so the chain stays composable.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Optional

from autoweaver.pipeline import VisionPipeline
from autoweaver.tasks import AlwaysFalseCondition, DoneCondition, TaskBase

from src.tasks.stats import TaskStats
from src.types import SegDetection


class SegPickTask(TaskBase):
    """Run segmentation pipeline and broadcast per-target pick info."""

    def __init__(
        self,
        pipeline: VisionPipeline,
        *,
        done_condition: Optional[DoneCondition] = None,
        stats: Optional[TaskStats] = None,
        name: str = "seg_pick",
    ):
        super().__init__()
        self.pipeline = pipeline
        self.done_condition = done_condition or AlwaysFalseCondition()
        self.stats = stats or TaskStats()
        self._name = name
        self._frame_count = 0

    @property
    def name(self) -> str:
        return self._name

    def run(self, data: Any) -> None:
        image = data
        pipeline_result = self.pipeline.run(image)
        self._frame_count += 1

        seg_dets = [d for d in pipeline_result.detections if isinstance(d, SegDetection)]
        picks = [_pick_payload(d) for d in seg_dets]

        metadata = {
            "pipeline_time_ms": pipeline_result.processing_time_ms,
            "detection_count": len(pipeline_result.detections),
            "seg_detection_count": len(seg_dets),
            "frame_in_task": self._frame_count,
            "seg_frame_id": pipeline_result.metadata.get("seg_frame_id"),
            "crop_single_square": pipeline_result.metadata.get("crop_single_square"),
            # Surface abstain's live margin so frame_loop preview / machine_result
            # snapshot can render the inset safety band consistently with the
            # filter that was actually applied to detections.
            "abstain_near_metal": pipeline_result.metadata.get("abstain_near_metal"),
        }
        result = SimpleNamespace(
            detections=pipeline_result.detections,
            stable_targets=[],
            tracked_targets=[],
            picks=picks,
            is_done=False,
            metadata=metadata,
        )

        self.stats.record(result)
        is_done = self.done_condition.check(self.stats, result)

        self.broadcast(
            "TASK:ITERATION",
            {
                "source": self._name,
                "payload": {
                    "detections": pipeline_result.detections,
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
        self.stats.reset()
        self.done_condition.reset()

    def close(self) -> None:
        super().close()

    def get_stats_summary(self) -> dict:
        return self.stats.summary()


def _pick_payload(d: SegDetection) -> dict[str, Any]:
    """Flatten the pick-relevant fields off a SegDetection for downstream tasks."""
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
