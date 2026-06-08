"""Tests for seg_pick multi-frame hit gating."""

from __future__ import annotations

from dataclasses import dataclass
import sys
from types import SimpleNamespace
import types

_tasks_mod = types.ModuleType("autoweaver.tasks")
_pipeline_mod = types.ModuleType("autoweaver.pipeline")
_reactive_mod = types.ModuleType("autoweaver.reactive")
_reactive_event_bus_mod = types.ModuleType("autoweaver.reactive.event_bus")


class _StubTaskBase:
    def __init__(self) -> None:
        self._event_bus = None

    def broadcast(self, *_args, **_kwargs) -> None:
        return None

    def attach(self, event_bus) -> None:
        self._event_bus = event_bus

    def close(self) -> None:
        self._event_bus = None


class _StubAlwaysFalseCondition:
    def check(self, *_args, **_kwargs) -> bool:
        return False

    def reset(self) -> None:
        return None


@dataclass
class _StubBoundingBox:
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)


@dataclass
class _StubDetection:
    bbox: _StubBoundingBox
    object_type: str
    confidence: float
    detection_id: str | None = None


_tasks_mod.TaskBase = _StubTaskBase
_tasks_mod.Task = object
_tasks_mod.SideTask = object
_tasks_mod.DoneCondition = object
_tasks_mod.AlwaysFalseCondition = _StubAlwaysFalseCondition

_pipeline_mod.BoundingBox = _StubBoundingBox
_pipeline_mod.Detection = _StubDetection
_pipeline_mod.VisionPipeline = object

_reactive_mod.EventBus = object
_reactive_event_bus_mod.EventBus = object

sys.modules.setdefault("autoweaver", types.ModuleType("autoweaver"))
sys.modules["autoweaver.tasks"] = _tasks_mod
sys.modules["autoweaver.pipeline"] = _pipeline_mod
sys.modules["autoweaver.reactive"] = _reactive_mod
sys.modules["autoweaver.reactive.event_bus"] = _reactive_event_bus_mod

from src.tasks.seg_pick_stabilized.stabilizer import SegDetectionStabilizer
from src.tasks.seg_pick_stabilized.task import SegPickStabilizedTask
from src.tasks.stabilized_detection.stabilizer import StabilizerConfig
from src.types import BoundingBox, SegDetection


def _make_detection(x: float, y: float, detection_id: str) -> SegDetection:
    return SegDetection(
        bbox=BoundingBox(x1=x - 5.0, y1=y - 5.0, x2=x + 5.0, y2=y + 5.0),
        object_type="foreign_matter",
        confidence=0.95,
        detection_id=detection_id,
        center_xy=[x, y],
        pick_point_xy=[x, y],
        pick_angle_deg=50.0,
        pick_method="mock",
        pick_score=0.9,
    )


def test_seg_detection_stabilizer_enforces_5_frame_1_hit_gate() -> None:
    stabilizer = SegDetectionStabilizer(
        StabilizerConfig(
            window_size=5,
            min_occurrence_ratio=0.2,
            stable_exit_ratio=0.2,
            min_frames_to_stable=1,
            distance_threshold_px=30.0,
            jump_threshold_px=60.0,
            missing_frames_to_delete=4,
            reset_on_jump=True,
        )
    )

    frames = [
        [_make_detection(100.0, 100.0, "f1")],
        [],
        [],
        [],
        [],
    ]

    outputs = [stabilizer.update(frame) for frame in frames]

    assert [len(items) for items in outputs] == [1, 1, 1, 1, 1]
    assert outputs[0][0].detection_id == "f1"
    assert outputs[4][0].detection_id == "f1"


class _StubVisionPipeline:
    def __init__(self, frames: list[list[SegDetection]]) -> None:
        self._frames = list(frames)
        self._index = 0

    def run(self, _image):
        detections = self._frames[self._index]
        self._index += 1
        return SimpleNamespace(
            detections=detections,
            processing_time_ms=1.0,
            metadata={"seg_frame_id": f"frame-{self._index}"},
        )


def test_seg_pick_stabilized_emits_only_on_last_frame_of_resume_window() -> None:
    pipeline = _StubVisionPipeline([
        [_make_detection(100.0, 100.0, "f1")],
        [],
        [],
        [],
        [],
    ])
    task = SegPickStabilizedTask(
        pipeline,
        stabilizer_config=StabilizerConfig(
            window_size=5,
            min_occurrence_ratio=0.2,
            stable_exit_ratio=0.2,
            min_frames_to_stable=1,
            distance_threshold_px=30.0,
            jump_threshold_px=60.0,
            missing_frames_to_delete=4,
            reset_on_jump=True,
        ),
    )
    events: list[tuple[str, dict]] = []
    task.broadcast = lambda event, payload: events.append((event, payload))  # type: ignore[method-assign]

    task.on_resume(5)
    for _ in range(5):
        task.run(object())

    assert len(events) == 1
    event, payload = events[0]
    assert event == "TASK:ITERATION"
    detections = payload["payload"]["detections"]
    assert len(detections) == 1
    assert detections[0].detection_id == "f1"
    metadata = payload["payload"]["metadata"]
    assert metadata["resume_cycle_frame"] == 5
    assert metadata["resume_cycle_total"] == 5
    assert metadata["resume_cycle_final"] is True


def test_seg_detection_stabilizer_selects_highest_mean_iou_frame() -> None:
    stabilizer = SegDetectionStabilizer(
        StabilizerConfig(
            window_size=5,
            min_occurrence_ratio=0.2,
            stable_exit_ratio=0.2,
            min_frames_to_stable=1,
            distance_threshold_px=30.0,
            jump_threshold_px=60.0,
            missing_frames_to_delete=4,
            reset_on_jump=True,
        )
    )

    frames = [
        [_make_detection(100.0, 100.0, "f1")],
        [_make_detection(101.0, 100.0, "f2")],
        [_make_detection(106.0, 100.0, "f3")],
        [],
        [],
    ]

    outputs = [stabilizer.update(frame) for frame in frames]

    assert outputs[2][0].detection_id == "f2"
    assert outputs[3][0].detection_id == "f2"
