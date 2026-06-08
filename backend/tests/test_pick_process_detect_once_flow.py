"""Tests for detect-once style retry behavior in PickProcess."""

from __future__ import annotations

from src.tasks.stabilized_detection.pick_process import (
    Phase,
    PickProcess,
    PickProcessConfig,
    TargetState,
)
from src.tasks.stabilized_detection.stabilizer import StableTarget


def _stable_target() -> StableTarget:
    return StableTarget(
        x=100.0,
        y=120.0,
        width=30.0,
        height=12.0,
        confidence=0.95,
        occurrence_ratio=1.0,
        object_type="hair",
        cluster_id="cluster_1",
        u=25.0,
    )


def test_pick_process_abandons_target_after_third_failed_attempt():
    process = PickProcess(
        PickProcessConfig(
            init_stable_threshold=1,
            confirm_window_frames=1,
            max_pick_attempts=3,
        )
    )
    stable = _stable_target()

    process.update([stable])
    assert process.phase == Phase.READY

    expected_dispatch_states = ["new_target", "retry_1", "retry_2"]
    target = None

    for attempt_index, expected_state in enumerate(expected_dispatch_states, start=1):
        target = process.get_next_target()
        assert target is not None
        assert target.pick_attempts == attempt_index
        assert process.get_dispatch_state(target) == expected_state

        process.on_pick_done(target.track_id)
        process.update([stable])

        if attempt_index < 3:
            assert target.state == TargetState.PENDING
            assert process.phase == Phase.READY
            assert process.get_last_pick_result().message == "not_disappeared"
        else:
            assert target.state == TargetState.ABANDONED
            assert process.phase == Phase.DONE
            assert process.get_last_pick_result().message == "abandoned"

    assert target is not None
    assert process.get_next_target() is None
