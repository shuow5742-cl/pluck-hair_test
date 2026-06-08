"""Tests for frame-loop pause/resume signaling."""

from __future__ import annotations

import time

import numpy as np
from autoweaver.reactive import EventBus

from src.tasks.frame_loop import FrameLoopConfig, FrameLoopSideTask


class _FakeCamera:
    def __init__(self) -> None:
        self.capture_count = 0
        self.is_open = False

    def open(self) -> bool:
        self.is_open = True
        return True

    def capture(self) -> np.ndarray:
        self.capture_count += 1
        return np.zeros((8, 8, 3), dtype=np.uint8)

    def close(self) -> None:
        self.is_open = False


class _FakeTask:
    name = "fake-task"

    def __init__(self) -> None:
        self.run_count = 0

    def run(self, _data) -> None:
        self.run_count += 1


def _wait_until(predicate, timeout: float = 1.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


def test_frame_loop_can_pause_and_resume_via_events():
    event_bus = EventBus()
    camera = _FakeCamera()
    task = _FakeTask()
    frame_loop = FrameLoopSideTask(
        camera=camera,
        task_map={"detect": task},
        config=FrameLoopConfig(loop_delay_ms=5, show_preview=False),
    )

    try:
        frame_loop.attach(event_bus)
        event_bus.publish("STATE:CHANGED", {"payload": {"new_state": "detect"}})

        assert _wait_until(lambda: task.run_count >= 3), "frame loop never started"

        event_bus.publish(
            "FRAME_LOOP:PAUSE",
            {"payload": {"reason": "unit_test"}},
        )
        time.sleep(0.05)
        paused_count = task.run_count
        time.sleep(0.05)

        assert task.run_count == paused_count

        event_bus.publish(
            "FRAME_LOOP:RESUME",
            {"payload": {"reason": "unit_test"}},
        )

        assert _wait_until(lambda: task.run_count > paused_count), "frame loop never resumed"
    finally:
        frame_loop.close()
