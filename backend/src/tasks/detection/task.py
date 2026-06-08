"""Default detection task (no tracker yet)."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Optional

from autoweaver.pipeline import VisionPipeline
from autoweaver.tasks import AlwaysFalseCondition, DoneCondition, TaskBase
from ..stats import TaskStats


class DetectionTask(TaskBase):
    """Wrap VisionPipeline into a Task without tracking.

    Satisfies the ``Task`` Protocol via structural subtyping.
    """

    def __init__(
        self,
        pipeline: VisionPipeline,
        *,
        done_condition: Optional[DoneCondition] = None,
        stats: Optional[TaskStats] = None,
        name: str = "detection",
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

        metadata = {
            "pipeline_time_ms": pipeline_result.processing_time_ms,
            "detection_count": len(pipeline_result.detections),
            "frame_in_task": self._frame_count,
        }
        result = SimpleNamespace(
            detections=pipeline_result.detections,
            stable_targets=[],
            tracked_targets=[],
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
                    "stable_targets": [],
                    "tracked_targets": [],
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
