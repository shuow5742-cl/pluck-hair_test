"""Detection task with multi-frame stabilization."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Optional

from autoweaver.pipeline import VisionPipeline
from autoweaver.reactive import EventBus
from autoweaver.tasks import AlwaysFalseCondition, DoneCondition, TaskBase
from ..stats import TaskStats
from .pick_process import PickProcess, PickProcessConfig
from .stabilizer import Stabilizer, StabilizerConfig

from src.core.pick_orientation import DetectedBBox, estimate_image_axis_yaw_from_bbox
from src.core.target_converter import TargetConverter


class StabilizedDetectionTask(TaskBase):
    """Detection task with Stabilizer integration.

    Runs Pipeline for single-frame detection, then uses Stabilizer
    to output stable targets with reduced jitter for display.

    Satisfies the ``Task`` Protocol via structural subtyping.
    """

    def __init__(
        self,
        pipeline: VisionPipeline,
        *,
        stabilizer_config: Optional[StabilizerConfig] = None,
        pick_process_config: Optional[PickProcessConfig] = None,
        target_converter: Optional[TargetConverter] = None,
        done_condition: Optional[DoneCondition] = None,
        stats: Optional[TaskStats] = None,
        name: str = "stabilized_detection",
    ):
        super().__init__()
        self.pipeline = pipeline
        self.done_condition = done_condition or AlwaysFalseCondition()
        self.stats = stats or TaskStats()
        self._name = name
        self._frame_count = 0
        self.stabilizer = Stabilizer(config=stabilizer_config)
        self.pick_process = PickProcess(config=pick_process_config)
        self._target_converter = target_converter

    @property
    def name(self) -> str:
        return self._name

    def attach(self, event_bus: EventBus) -> None:
        super().attach(event_bus)

    def subscribe(self) -> None:
        if self._event_bus is None:
            return
        self._event_bus.subscribe("COMM:REQUEST_TARGET", self._on_comm_request_target)
        self._event_bus.subscribe("COMM:PICK_DONE", self._on_comm_pick_done)
        self._event_bus.subscribe("COMM:RESET", self._on_comm_reset)

    def run(self, data: Any) -> None:
        image = data
        # Step 1: Single-frame detection
        pipeline_result = self.pipeline.run(image)
        self._frame_count += 1

        # Step 2: Multi-frame stabilization (technical domain)
        stable_targets = self.stabilizer.update(pipeline_result.detections)
        self._attach_pick_orientation(image, stable_targets)

        # Step 2.5: Attach world coordinates if converter available
        if self._target_converter is not None:
            for t in stable_targets:
                wp = self._target_converter._transformer.pixel_to_world(
                    t.x, t.y, 0.0, 0.0,
                )
                t.world_x = wp.x
                t.world_y = wp.y

        # Step 3: Track ID management (business domain)
        self.pick_process.update(stable_targets)
        tracked_targets = self.pick_process.get_all_targets()

        if self._target_converter is not None:
            world_targets = self._target_converter.convert(tracked_targets)
            by_track_id = {wt.track_id: wt for wt in world_targets}
            for tracked in tracked_targets:
                wt = by_track_id.get(tracked.track_id)
                if wt is None:
                    continue
                tracked.world_x = wt.x
                tracked.world_y = wt.y

        track_stats = self.pick_process.get_stats()

        metadata = {
            "pipeline_time_ms": pipeline_result.processing_time_ms,
            "detection_count": len(pipeline_result.detections),
            "stable_count": len(stable_targets),
            "cluster_count": self.stabilizer.get_cluster_count(),
            "frame_in_task": self._frame_count,
            # PickProcess statistics
            "region_picked": track_stats.region_picked,
            "current_pending": track_stats.current_pending,
            "phase": self.pick_process.phase.value,
        }
        result = SimpleNamespace(
            detections=pipeline_result.detections,
            stable_targets=stable_targets,
            tracked_targets=tracked_targets,
            is_done=False,
            metadata=metadata,
        )

        self.stats.record(result)
        is_done = self.done_condition.check(self.stats, result)

        # Publish iteration event
        self.broadcast(
            "TASK:ITERATION",
            {
                "source": self._name,
                "payload": {
                    "detections": pipeline_result.detections,
                    "stable_targets": stable_targets,
                    "tracked_targets": tracked_targets,
                    "metadata": metadata,
                },
            },
        )

        # Publish pick result if available
        pick_result = self.pick_process.get_last_pick_result()
        if pick_result is not None:
            self.broadcast(
                "TASK:PICK_RESULT",
                {
                    "source": self._name,
                    "payload": {
                        "pick_result": pick_result,
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

    # ------------------------------------------------------------------
    # COMM event handlers (called synchronously by EventBus)
    # ------------------------------------------------------------------

    def _on_comm_request_target(self, _event: str, _data: dict) -> None:
        """Handle PLC request for next pick target."""
        target = self.pick_process.get_next_target()
        if target is None:
            return  # No response → CommTask sees None → returns error to PLC

        dispatch_state = self.pick_process.get_dispatch_state(target)
        yaw = 0.0 if target.u is None else round(target.u, 4)

        if self._target_converter is not None:
            wt = self._target_converter.convert_one(
                track_id=target.track_id,
                x=target.x,
                y=target.y,
                width=target.width,
                height=target.height,
                confidence=target.confidence,
                object_type=target.object_type,
            )
            payload = {
                "type": "target",
                "track_id": wt.track_id,
                "x": round(wt.x, 4),
                "y": round(wt.y, 4),
                "world_x_mm": round(wt.x, 4),
                "world_y_mm": round(wt.y, 4),
                "pixel_x": round(target.x, 2),
                "pixel_y": round(target.y, 2),
                "width": round(wt.width, 4),
                "height": round(wt.height, 4),
                "width_mm": round(wt.width, 4),
                "height_mm": round(wt.height, 4),
                "confidence": wt.confidence,
                "object_type": wt.object_type,
                "state": target.state.value,
                "dispatch_state": dispatch_state,
                "u": yaw,
                "pick_attempts": target.pick_attempts,
                "cluster_id": target.cluster_id,
            }
        else:
            payload = {
                "type": "target",
                "track_id": target.track_id,
                "x": target.x,
                "y": target.y,
                "world_x_mm": None,
                "world_y_mm": None,
                "pixel_x": round(target.x, 2),
                "pixel_y": round(target.y, 2),
                "width": target.width,
                "height": target.height,
                "width_mm": None,
                "height_mm": None,
                "confidence": target.confidence,
                "object_type": target.object_type,
                "state": target.state.value,
                "dispatch_state": dispatch_state,
                "u": yaw,
                "pick_attempts": target.pick_attempts,
                "cluster_id": target.cluster_id,
            }

        self.broadcast(
            "COMM:TARGET_RESPONSE",
            {"source": self._name, "payload": payload},
        )
        self.broadcast(
            "FRAME_LOOP:PAUSE",
            {
                "source": self._name,
                "payload": {
                    "reason": "awaiting_robot_action",
                    "track_id": target.track_id,
                },
            },
        )

    def _on_comm_pick_done(self, _event: str, data: dict) -> None:
        """Handle PLC pick-done notification."""
        payload = data.get("payload", {})
        track_id = payload.get("track_id")
        if track_id is not None:
            self.pick_process.on_pick_done(target_id=track_id)
            self.broadcast(
                "FRAME_LOOP:RESUME",
                {
                    "source": self._name,
                    "payload": {
                        "reason": "pick_done_confirmation",
                        "track_id": track_id,
                    },
                },
            )

    def _on_comm_reset(self, _event: str, _data: dict) -> None:
        """Handle PLC reset request."""
        self.broadcast(
            "FRAME_LOOP:RESUME",
            {
                "source": self._name,
                "payload": {
                    "reason": "reset",
                },
            },
        )
        self.reset()

    def reset(self) -> None:
        """Reset all state for new session."""
        self._frame_count = 0
        self.stabilizer.reset()
        self.pick_process.reset()
        self.stats.reset()
        self.done_condition.reset()

    def close(self) -> None:
        super().close()

    def get_stats_summary(self) -> dict:
        return self.stats.summary()

    def get_last_pick_result(self):
        """Expose last pick result without leaking internal PickProcess."""
        return self.pick_process.get_last_pick_result()

    def _attach_pick_orientation(self, image: Any, stable_targets: list) -> None:
        """Estimate image-plane grasp yaw for each stable target."""
        for target in stable_targets:
            bbox = DetectedBBox(
                x1=target.x - target.width / 2.0,
                y1=target.y - target.height / 2.0,
                x2=target.x + target.width / 2.0,
                y2=target.y + target.height / 2.0,
            )
            target.u = estimate_image_axis_yaw_from_bbox(image, bbox=bbox)
