"""Tests for post-pick confirmation / retry flow in plc_orchestrator."""

from __future__ import annotations

import sys
import types
from dataclasses import dataclass

_tasks_mod = types.ModuleType("autoweaver.tasks")
_pipeline_mod = types.ModuleType("autoweaver.pipeline")
_reactive_mod = types.ModuleType("autoweaver.reactive")
_reactive_event_bus_mod = types.ModuleType("autoweaver.reactive.event_bus")

_tasks_mod.TaskBase = object
_tasks_mod.Task = object
_tasks_mod.SideTask = object
_tasks_mod.DoneCondition = object


class _StubAlwaysFalseCondition:
    def check(self, *_args, **_kwargs) -> bool:
        return False

    def reset(self) -> None:
        return None


_tasks_mod.AlwaysFalseCondition = _StubAlwaysFalseCondition


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


_pipeline_mod.VisionPipeline = object
_pipeline_mod.Detection = _StubDetection
_pipeline_mod.BoundingBox = _StubBoundingBox
_reactive_mod.EventBus = object
_reactive_event_bus_mod.EventBus = object

sys.modules.setdefault("autoweaver", types.ModuleType("autoweaver"))
sys.modules.setdefault("autoweaver.tasks", _tasks_mod)
sys.modules.setdefault("autoweaver.pipeline", _pipeline_mod)
sys.modules.setdefault("autoweaver.reactive", _reactive_mod)
sys.modules.setdefault("autoweaver.reactive.event_bus", _reactive_event_bus_mod)

from src.config import PlcOrchestratorConfig
from src.core.arm_grid_mapper import ArmGridMapper, ArmGridPoint
from src.tasks.plc_orchestrator.points import PlcPoint
from src.tasks.plc_orchestrator.protocol import ProtocolWorker
from src.tasks.plc_orchestrator.task import PlcOrchestratorTask


def _make_point() -> PlcPoint:
    pose6 = {"x": 0.0, "y": 0.0, "z": 0.0, "u": 0.0, "v": 0.0, "w": 0.0}
    pose4 = {"x": 0.0, "y": 0.0, "z": 0.0, "u": 50.0}
    return PlcPoint(
        press_index=1,
        photo_index=1,
        nova2=pose6,
        nova5=pose6,
        epson_ls6_fallback=pose4,
        repeat=1,
    )


def _make_task() -> PlcOrchestratorTask:
    return PlcOrchestratorTask(
        PlcOrchestratorConfig(enabled=False, auto_start=False, points_path="unused"),
        points=[_make_point()],
    )


def _make_pick(
    x: float,
    y: float,
    *,
    detection_id: str,
    width: float = 20.0,
    height: float = 6.0,
    preferred_epson_tool: int | None = None,
) -> dict:
    return {
        "detection_id": detection_id,
        "world_xy_mm": [1.0, 2.0],
        "flange_target_xy_mm": [3.0, 4.0],
        "pick_point_xy_px": [x, y],
        "pick_angle_deg": 70.0,
        "bbox_center_xy_px": [x, y],
        "bbox_width_px": width,
        "bbox_height_px": height,
        "preferred_epson_tool": preferred_epson_tool,
    }


def test_next_task_ready_arms_confirmation_when_pick_was_dispatched() -> None:
    task = _make_task()
    with task._picks_lock:
        task._dispatched_pick = _make_pick(100.0, 100.0, detection_id="hair-1")
        task._dispatched_pick["pick_attempts"] = 1
        task._confirm_frames_remaining = 10
        task._awaiting_confirmation_batch = False

    assert task._next_task_ready() is False

    with task._picks_lock:
        assert task._awaiting_confirmation_batch is True
        assert task._accepting_batch is True
        assert task._confirm_frames_remaining == 10


def test_confirmation_queues_retry_when_target_still_present() -> None:
    task = _make_task()
    with task._picks_lock:
        task._dispatched_pick = _make_pick(100.0, 100.0, detection_id="hair-1")
        task._dispatched_pick["pick_attempts"] = 1
        task._confirm_frames_remaining = 1
        task._awaiting_confirmation_batch = True

        followup = [
            _make_pick(102.0, 101.0, detection_id="hair-1-new"),
            _make_pick(200.0, 200.0, detection_id="hair-2"),
        ]
        needs_more = task._handle_confirmation_batch_locked(followup, seg_frame_id="f-1")

        assert needs_more is False
        queued = list(task._world_picks)
        assert [pick["detection_id"] for pick in queued] == ["hair-1-new", "hair-2"]
        assert queued[0]["pick_attempts"] == 1
        assert task._dispatched_pick is None


def test_confirmation_abandons_target_after_third_failed_attempt() -> None:
    task = _make_task()
    with task._picks_lock:
        task._dispatched_pick = _make_pick(100.0, 100.0, detection_id="hair-1")
        task._dispatched_pick["pick_attempts"] = 3
        task._confirm_frames_remaining = 1
        task._awaiting_confirmation_batch = True

        followup = [
            _make_pick(101.0, 99.5, detection_id="hair-1-new"),
            _make_pick(200.0, 200.0, detection_id="hair-2"),
        ]
        needs_more = task._handle_confirmation_batch_locked(followup, seg_frame_id="f-2")

        assert needs_more is False
        queued = list(task._world_picks)
        assert [pick["detection_id"] for pick in queued] == ["hair-2"]
        assert task._dispatched_pick is None
        assert len(task._ignored_targets) == 1


def test_confirmation_thresholds_are_configurable_from_plc_config() -> None:
    task = PlcOrchestratorTask(
        PlcOrchestratorConfig(
            enabled=False,
            auto_start=False,
            points_path="unused",
            pick_confirm_match_distance_px=12.0,
            pick_confirm_match_size_ratio=0.1,
        ),
        points=[_make_point()],
    )

    assert task._find_matching_pick_index(
        _make_pick(100.0, 100.0, detection_id="target"),
        [_make_pick(118.0, 100.0, detection_id="drifted")],
    ) is None

    assert task._find_matching_pick_index(
        _make_pick(100.0, 100.0, detection_id="target", width=20.0, height=6.0),
        [_make_pick(105.0, 100.0, detection_id="resized", width=23.0, height=6.0)],
    ) is None


def test_ignored_target_is_not_requeued_later_in_same_photo() -> None:
    task = _make_task()
    abandoned = _make_pick(100.0, 100.0, detection_id="hair-1", width=20.0, height=6.0)
    task._ignored_targets.append(dict(abandoned))

    task._replace_buffer_locked([
        _make_pick(101.0, 100.0, detection_id="hair-1-rediscovered", width=20.0, height=6.0),
        _make_pick(200.0, 200.0, detection_id="hair-2", width=18.0, height=5.0),
    ])

    queued = list(task._world_picks)
    assert [pick["detection_id"] for pick in queued] == ["hair-2"]


def test_start_press_index_resolves_to_first_matching_press_row() -> None:
    pose6 = {"x": 0.0, "y": 0.0, "z": 0.0, "u": 0.0, "v": 0.0, "w": 0.0}
    pose4 = {"x": 0.0, "y": 0.0, "z": 0.0, "u": 50.0}
    points = [
        PlcPoint(press_index=press, photo_index=photo, nova2=pose6, nova5=pose6, epson_ls6_fallback=pose4, repeat=1)
        for press in range(1, 101)
        for photo in range(1, 3)
    ]
    task = PlcOrchestratorTask(
        PlcOrchestratorConfig(
            enabled=False,
            auto_start=False,
            points_path="unused",
            start_press_row=10,
            start_press_index=3,
        ),
        points=points,
    )

    assert task._resolve_start_point_index(3) == (30 - 1) * 2


def test_start_press_index_respects_ignored_rows_submatrix() -> None:
    pose6 = {"x": 0.0, "y": 0.0, "z": 0.0, "u": 0.0, "v": 0.0, "w": 0.0}
    pose4 = {"x": 0.0, "y": 0.0, "z": 0.0, "u": 50.0}
    points = [
        PlcPoint(press_index=press, photo_index=1, nova2=pose6, nova5=pose6, epson_ls6_fallback=pose4, repeat=1)
        for press in range(1, 101)
    ]
    task = PlcOrchestratorTask(
        PlcOrchestratorConfig(
            enabled=False,
            auto_start=False,
            points_path="unused",
            start_press_row=2,
            start_press_index=99,
        ),
        points=points,
    )

    # Row 1 ignored -> active presses are 2..10, 12..20, ... , 92..100 (90 entries).
    # start_press_index beyond active size falls back to first active press.
    assert task._resolve_start_point_index(99) == 1


def test_retry_attempts_apply_configurable_z_offsets() -> None:
    task = PlcOrchestratorTask(
        PlcOrchestratorConfig(
            enabled=False,
            auto_start=False,
            points_path="unused",
            epson_tweezer_z_mm=9.8,
            epson_tweezer_z_retry2_offset_mm=0.5,
            epson_tweezer_z_retry3_offset_mm=0.5,
        ),
        points=[_make_point()],
    )
    task._nova5_to_epson_mapper = ArmGridMapper([
        ArmGridPoint(
            row=1,
            col=1,
            nova5_x=3.0,
            nova5_y=4.0,
            epson_x=-86.0,
            epson_y=-46.0,
        )
    ])

    with task._picks_lock:
        task._world_picks.append(_make_pick(100.0, 100.0, detection_id="hair-1"))
        second = _make_pick(100.0, 100.0, detection_id="hair-1")
        second["pick_attempts"] = 1
        task._world_picks.append(second)
        third = _make_pick(100.0, 100.0, detection_id="hair-1")
        third["pick_attempts"] = 2
        task._world_picks.append(third)

    point = _make_point()
    fallback = dict(point.epson_ls6_fallback)

    first_coord = task._resolve_epson_coord(point, fallback)
    second_coord = task._resolve_epson_coord(point, fallback)
    third_coord = task._resolve_epson_coord(point, fallback)

    assert first_coord is not None
    assert second_coord is not None
    assert third_coord is not None
    assert first_coord["z"] == 9.8
    assert second_coord["z"] == 10.3
    assert third_coord["z"] == 10.3


def test_suction_retry_attempts_lower_z_and_keep_tool_offsets() -> None:
    task = PlcOrchestratorTask(
        PlcOrchestratorConfig(
            enabled=False,
            auto_start=False,
            points_path="unused",
            epson_suction_z_mm=9.8,
            epson_suction_z_retry2_offset_mm=0.5,
            epson_suction_z_retry3_offset_mm=0.5,
            epson_tool2_offset_z_mm=-1.0,
        ),
        points=[_make_point()],
    )
    task._nova5_to_epson_mapper = ArmGridMapper([
        ArmGridPoint(
            row=1,
            col=1,
            nova5_x=3.0,
            nova5_y=4.0,
            epson_x=-86.0,
            epson_y=-46.0,
        )
    ])
    task._worker.current_epson_tool = 2

    with task._picks_lock:
        task._world_picks.append(
            _make_pick(100.0, 100.0, detection_id="hair-1", preferred_epson_tool=2)
        )
        second = _make_pick(
            100.0, 100.0, detection_id="hair-1", preferred_epson_tool=2
        )
        second["pick_attempts"] = 1
        task._world_picks.append(second)
        third = _make_pick(
            100.0, 100.0, detection_id="hair-1", preferred_epson_tool=2
        )
        third["pick_attempts"] = 2
        task._world_picks.append(third)

    point = _make_point()
    fallback = dict(point.epson_ls6_fallback)

    first_coord = task._resolve_epson_coord(point, fallback)
    second_coord = task._resolve_epson_coord(point, fallback)
    third_coord = task._resolve_epson_coord(point, fallback)

    assert first_coord is not None
    assert second_coord is not None
    assert third_coord is not None
    assert first_coord["z"] == 8.8
    assert second_coord["z"] == 8.3
    assert third_coord["z"] == 8.3


def test_protocol_worker_restarts_from_start_index_on_nova5_func2_for_tool1() -> None:
    worker = ProtocolWorker(
        PlcOrchestratorConfig(
            enabled=False,
            auto_start=True,
            points_path="unused",
        ),
        points=[_make_point(), _make_point()],
    )
    worker.start_index = 1
    worker.auto_running = False
    worker.completed_cycle = True
    worker.current_index = 0
    worker.pick_sent_count = 5
    worker.active_press_index = 100
    worker.active_photo_key = "100-7"
    worker.current_epson_tool = 1
    worker.requests["nova5"].flag = 1.0
    worker.requests["nova5"].func = 2.0
    worker.requests["nova5"].handled = False

    worker._maybe_restart_completed_cycle()

    assert worker.auto_running is True
    assert worker.completed_cycle is False
    assert worker.current_index == 1
    assert worker.pick_sent_count == 0
    assert worker.active_press_index is None
    assert worker.active_photo_key is None


def test_protocol_worker_restarts_on_nova2_func1_for_tool2_without_waiting_for_func2() -> None:
    worker = ProtocolWorker(
        PlcOrchestratorConfig(
            enabled=False,
            auto_start=True,
            points_path="unused",
        ),
        points=[_make_point(), _make_point()],
    )
    worker.start_index = 1
    worker.auto_running = False
    worker.completed_cycle = True
    worker.pick_sent_count = 5
    worker.active_press_index = 100
    worker.active_photo_key = "100-7"
    worker.current_epson_tool = 2
    worker.requests["nova2"].flag = 1.0
    worker.requests["nova2"].func = 1.0
    worker.requests["nova2"].handled = False

    worker._maybe_restart_completed_cycle()

    assert worker.auto_running is True
    assert worker.completed_cycle is False
    assert worker.current_index == 1
    assert worker.current_route_pos == 1
    assert worker.pick_sent_count == 0
    assert worker.active_press_index is None
    assert worker.active_photo_key is None


def test_protocol_worker_does_not_restart_on_nova2_func1_for_tool1() -> None:
    worker = ProtocolWorker(
        PlcOrchestratorConfig(
            enabled=False,
            auto_start=True,
            points_path="unused",
        ),
        points=[_make_point(), _make_point()],
    )
    worker.start_index = 1
    worker.auto_running = False
    worker.completed_cycle = True
    worker.current_index = 0
    worker.pick_sent_count = 5
    worker.active_press_index = 100
    worker.active_photo_key = "100-7"
    worker.current_epson_tool = 1
    worker.requests["nova2"].flag = 1.0
    worker.requests["nova2"].func = 1.0
    worker.requests["nova2"].handled = False

    worker._maybe_restart_completed_cycle()

    assert worker.auto_running is False
    assert worker.completed_cycle is True
    assert worker.current_index == 0


def test_protocol_worker_advances_within_filtered_route_only() -> None:
    pose6 = {"x": 0.0, "y": 0.0, "z": 0.0, "u": 0.0, "v": 0.0, "w": 0.0}
    pose4 = {"x": 0.0, "y": 0.0, "z": 0.0, "u": 50.0}
    points = [
        PlcPoint(press_index=press, photo_index=1, nova2=pose6, nova5=pose6, epson_ls6_fallback=pose4, repeat=1)
        for press in range(1, 21)
    ]
    worker = ProtocolWorker(
        PlcOrchestratorConfig(
            enabled=False,
            auto_start=False,
            points_path="unused",
        ),
        points=points,
    )
    # start_press_row=5 semantics on first two columns -> keep 5..10 and 15..20.
    worker.route_indices = [4, 5, 6, 7, 8, 9, 14, 15, 16, 17, 18, 19]
    worker.start_route_pos = 0
    worker.current_route_pos = 5
    worker.current_index = worker.route_indices[worker.current_route_pos]  # press 10
    worker.auto_running = True
    worker.active_press_index = 10
    worker.active_photo_key = "10-1"

    sent: list[tuple[str, int, str]] = []

    def _capture(robot: str, _func_addr: int, _flag_addr: int, code: int, desc: str) -> bool:
        sent.append((robot, code, desc))
        return True

    worker._send_non_coord = _capture  # type: ignore[method-assign]
    worker._advance_or_finish()

    assert worker.current_route_pos == 6
    assert worker.current_index == 14
    assert worker.points[worker.current_index].press_index == 15
    assert sent[0][1] == 22


def test_tool2_applies_xyz_offsets_before_sending() -> None:
    task = PlcOrchestratorTask(
        PlcOrchestratorConfig(
            enabled=False,
            auto_start=False,
            points_path="unused",
            epson_z_mm=9.8,
            epson_tool2_offset_x_mm=0.1,
            epson_tool2_offset_y_mm=0.1,
            epson_tool2_offset_z_mm=0.1,
        ),
        points=[_make_point()],
    )
    task._nova5_to_epson_mapper = ArmGridMapper([
        ArmGridPoint(
            row=1,
            col=1,
            nova5_x=3.0,
            nova5_y=4.0,
            epson_x=-86.0,
            epson_y=-46.0,
        )
    ])
    task._worker.current_epson_tool = 2

    with task._picks_lock:
        task._world_picks.append(_make_pick(100.0, 100.0, detection_id="hair-1"))

    point = _make_point()
    fallback = dict(point.epson_ls6_fallback)
    coord = task._resolve_epson_coord(point, fallback)

    assert coord is not None
    assert coord["x"] == -85.9
    assert coord["y"] == -45.9
    assert coord["z"] == 9.9


def test_resolve_epson_coord_filters_picks_by_current_tool() -> None:
    task = PlcOrchestratorTask(
        PlcOrchestratorConfig(
            enabled=False,
            auto_start=False,
            points_path="unused",
            epson_z_mm=9.8,
        ),
        points=[_make_point()],
    )
    task._nova5_to_epson_mapper = ArmGridMapper([
        ArmGridPoint(
            row=1,
            col=1,
            nova5_x=3.0,
            nova5_y=4.0,
            epson_x=-86.0,
            epson_y=-46.0,
        )
    ])
    task._worker.current_epson_tool = 1

    with task._picks_lock:
        task._world_picks.append(
            _make_pick(100.0, 100.0, detection_id="suck-1", preferred_epson_tool=2)
        )
        task._world_picks.append(
            _make_pick(110.0, 100.0, detection_id="tweezer-1", preferred_epson_tool=1)
        )

    coord = task._resolve_epson_coord(_make_point(), dict(_make_point().epson_ls6_fallback))

    assert coord is not None
    with task._picks_lock:
        remaining = list(task._world_picks)
    assert remaining[0]["detection_id"] == "suck-1"
    assert task._dispatched_pick is not None
    assert task._dispatched_pick["detection_id"] == "tweezer-1"


def test_batch_received_empty_is_tool_aware() -> None:
    task = _make_task()
    task._worker.current_epson_tool = 1
    with task._picks_lock:
        task._batch_received = True
        task._received_was_empty = False
        task._world_picks.append(
            _make_pick(100.0, 100.0, detection_id="suck-1", preferred_epson_tool=2)
        )

    assert task._has_pending_picks() is False
    assert task._batch_received_empty() is True
