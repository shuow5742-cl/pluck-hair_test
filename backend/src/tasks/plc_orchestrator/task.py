"""PlcOrchestratorTask — SideTask wrapping ProtocolWorker.

Bridges three things:

1. Owns one ``ProtocolWorker`` thread (and its Modbus TCP connection).
2. Subscribes to ``TASK:WORLD_PICKS`` (broadcast by PixelToWorldTask) and
   maintains a buffer of the most recent vision picks so the worker can
   resolve Epson LS6 targets on demand.
3. Conforms to the autoweaver SideTask protocol (``attach`` / ``close``)
   so ``WorkflowEngine`` can manage its lifecycle alongside other side
   tasks like ``frame_loop`` and ``pixel_to_world``.

Coord-source policy for Epson LS6 (vision-driven, no YAML fallback for
the no-pick case):
- The first ``TASK:WORLD_PICKS`` event after each photo arrival is the
  canonical batch — locked into the deque until next arrival.
- Each Epson coord request pops one pick. z/u are merged from the YAML
  ``epson_ls6_fallback`` entry (vision only provides world XY).
- Three states the worker can observe via the callbacks:
    no batch yet    → silent (PLC keeps polling, we'll respond on next round)
    batch with picks → send coord
    batch was empty  → send func=21 to skip this photo position
"""

from __future__ import annotations

import logging
import math
import threading
import time
from collections import deque
from typing import Any, Callable, Deque, Dict, List, Optional

from autoweaver.reactive.event_bus import EventBus

from src.config import PlcOrchestratorConfig
from src.core.arm_grid_mapper import ArmGridMapper, ArmGridMatch
from src.pose_sources import PlcPoseSource, register_pose_source
from src.tasks.stabilized_detection.pick_process import PickProcessConfig

from .points import PlcPoint, load_points
from .protocol import EpsonCoord, EpsonStageDecision, ProtocolWorker

logger = logging.getLogger(__name__)


class PlcOrchestratorTask:
    """SideTask: PLC ModbusTCP orchestrator for nova2 / nova5 / Epson LS6."""

    def __init__(
        self,
        cfg: PlcOrchestratorConfig,
        *,
        name: str = "plc_orchestrator",
        pose_source_name: str = "plc",
        points: Optional[List[PlcPoint]] = None,
    ) -> None:
        self._name = name
        self._cfg = cfg
        self._pose_source_name = pose_source_name
        self._points: List[PlcPoint] = points if points is not None else load_points(cfg.points_path)
        mapping_cfg = cfg.nova5_to_epson_mapping
        self._nova5_to_epson_mapper: Optional[ArmGridMapper] = None
        if mapping_cfg.enabled:
            self._nova5_to_epson_mapper = ArmGridMapper.load(mapping_cfg.path)
        self._event_bus: Optional[EventBus] = None
        self._unsubscribers: List[Callable[[], None]] = []

        # Vision pick buffer (popped left-to-right by the worker).
        #
        # Lifecycle per photo position:
        #   1. nova5 acks photo arrival → buffer cleared, _accepting_batch=True
        #   2. First WORLD_PICKS event lands → buffer filled, _accepting_batch=False
        #      (further WORLD_PICKS events from the same photo position are
        #      ignored — locking the batch keeps Epson from chasing pick coords
        #      that jitter ~1-2 px between frames as YOLO re-runs)
        #   3. Epson drains the buffer pick by pick
        #   4. nova5 moves to next photo → cycle restarts
        self._picks_lock = threading.Lock()
        self._world_picks: Deque[Dict[str, Any]] = deque()
        self._last_seg_frame_id: Optional[Any] = None
        # Batch lifecycle: accepting → received. Both False means "not at a
        # photo position". `_received_was_empty` captures the original batch
        # size (0 vs. non-0) so we don't confuse "vision found nothing" with
        # "Epson drained the buffer normally" — both end up with an empty
        # deque but only the former should trigger a skip.
        self._accepting_batch: bool = False
        self._batch_received: bool = False
        self._received_was_empty: bool = False
        self._confirm_cfg = PickProcessConfig(
            match_distance_threshold=cfg.pick_confirm_match_distance_px,
            match_size_ratio_threshold=cfg.pick_confirm_match_size_ratio,
        )
        self._dispatched_pick: Optional[Dict[str, Any]] = None
        self._last_epson_coord: Optional[EpsonCoord] = None
        self._confirm_frames_remaining: int = 0
        self._awaiting_confirmation_batch: bool = False
        self._confirmation_mode: Optional[str] = None
        self._post_pick_stage_decision: Optional[EpsonStageDecision] = None
        self._ignored_targets: List[Dict[str, Any]] = []
        self._confirm_started_at: Optional[float] = None
        self._alignment_stage_started_at: Dict[int, float] = {}
        self._alignment_last_wait_log_at: float = 0.0
        from src.comm.inproc_bus import get_live_state_bus
        self._live_state = get_live_state_bus()
        self._cycle_timing_lock = threading.Lock()
        self._photo_cycle_timing: Optional[Dict[str, Any]] = None

        self._worker = ProtocolWorker(
            cfg,
            self._points,
            epson_coord_provider=self._resolve_epson_coord,
            on_photo_arrived=self._on_photo_arrived,
            has_pending_picks=self._has_pending_picks,
            batch_received_empty=self._batch_received_empty,
            next_task_ready=self._next_task_ready,
            on_epson_motion_stage=self._on_epson_motion_stage,
            on_nova5_moving=self._on_nova5_moving,
            on_epson_coord_sent=self._on_epson_coord_sent,
            on_epson_pick_request=self._on_epson_pick_request,
            on_epson_next_task_request=self._on_epson_next_task_request,
            on_nova5_coord_sent=self._on_nova5_coord_sent,
        )

    @property
    def name(self) -> str:
        return self._name

    # ---- SideTask protocol ----

    def attach(self, event_bus: EventBus) -> None:
        self._event_bus = event_bus
        self._unsubscribers.append(
            event_bus.subscribe("TASK:WORLD_PICKS", self._on_world_picks)
        )

        # Expose nova5 RT position as a PoseSource so PixelToWorldTask can
        # resolve the live flange pose. Registering here (not at __init__)
        # guarantees the worker is wired before any consumer reads it.
        register_pose_source(
            self._pose_source_name,
            PlcPoseSource(self._worker.get_rt_nova5_xy),
        )

        # Pause the camera loop until nova5 confirms it's parked at a photo
        # position. Sent here (not from the worker thread) so the event bus
        # publish stays on the attach/main thread, and so FrameLoopSideTask
        # — which subscribes to FRAME_LOOP:PAUSE in its own attach() — has
        # already been wired before this point (yaml side_tasks order:
        # pixel_to_world → plc_orchestrator → frame_loop is appended last
        # in main.py).
        self._publish_frame_loop_pause("plc_orchestrator startup")

        self._worker.start()
        self._worker.connect_now()
        if self._cfg.auto_start:
            route_indices = self._build_active_route_indices()
            start_route_pos = self._resolve_start_route_position(
                route_indices,
                self._cfg.start_press_index,
            )
            self._worker.start_auto(
                start_index=route_indices[start_route_pos] if route_indices else 0,
                route_indices=route_indices,
                start_route_pos=start_route_pos,
            )

        logger.info(
            "plc_orchestrator attached: host=%s:%s, %d points, auto_start=%s, "
            "pose_source=%s, nova5_to_epson_mapping=%s",
            self._cfg.host, self._cfg.port, len(self._points), self._cfg.auto_start,
            self._pose_source_name,
            self._cfg.nova5_to_epson_mapping.path if self._nova5_to_epson_mapper else "disabled",
        )

    def close(self) -> None:
        for unsub in self._unsubscribers:
            try:
                unsub()
            except Exception:  # noqa: BLE001
                pass
        self._unsubscribers.clear()
        self._event_bus = None

        try:
            self._worker.stop_auto()
            self._worker.disconnect_now()
        except Exception:  # noqa: BLE001
            pass
        self._worker.stop()

    # ---- Event handlers ----

    def _on_world_picks(self, _event: str, data: dict) -> None:
        payload = data.get("payload") or {}
        picks = payload.get("world_picks") or []
        seg_frame_id = payload.get("seg_frame_id")
        degraded = bool(payload.get("degraded"))
        flange_pose = payload.get("flange_pose_mm")
        frame_metadata = payload.get("metadata") or {}

        if degraded and not picks:
            logger.debug("world_picks: degraded with no picks, ignoring")
            return

        confirm_followup_needed = False
        with self._picks_lock:
            if self._awaiting_confirmation_batch:
                confirm_followup_needed = self._handle_confirmation_batch_locked(
                    picks,
                    seg_frame_id=seg_frame_id,
                )
                buffered = list(self._world_picks)
            elif not self._accepting_batch:
                # Already snapshotted a batch for the current photo position
                # (or nova5 hasn't arrived yet) — ignore until next arrival.
                logger.debug(
                    "world_picks dropped (batch already locked for photo, frame=%s): n=%d",
                    seg_frame_id, len(picks),
                )
                return
            else:
                self._replace_buffer_locked(picks)
                self._last_seg_frame_id = seg_frame_id
                self._accepting_batch = False
                self._batch_received = True
                self._received_was_empty = not self._world_picks
                buffered = list(self._world_picks)

        if confirm_followup_needed:
            self._publish_frame_loop_resume("post-pick confirmation follow-up")

        # One INFO block per photo position. Subsequent frames at this
        # position fall into the debug-drop branch above, so the user gets
        # exactly one log per "camera parks → snapshot → moves on" cycle.
        photo_key = self._worker.active_photo_key or "?"
        if flange_pose and len(flange_pose) >= 2:
            pose_str = f"flange_xy=({flange_pose[0]:.3f}, {flange_pose[1]:.3f}) mm"
        else:
            pose_str = "flange_xy=<unavailable>"
        batch_label = (
            "confirm_world_picks"
            if (
                self._dispatched_pick is not None
                or self._post_pick_stage_decision is not None
                or confirm_followup_needed
            )
            else "world_picks"
        )
        self._record_photo_cycle_event(
            batch_label,
            n=len(buffered),
            seg_frame_id=seg_frame_id,
            region_total_ms=frame_metadata.get("region_total_ms"),
            capture_batch_ms=frame_metadata.get("capture_batch_ms"),
            pipeline_total_ms=frame_metadata.get("cycle_pipeline_total_ms"),
            postprocess_total_ms=frame_metadata.get("cycle_postprocess_total_ms"),
            yolo_total_ms=frame_metadata.get("cycle_yolo_total_ms"),
            pick_total_ms=frame_metadata.get("cycle_pick_total_ms"),
        )
        if buffered:
            lines = [
                f"{batch_label} @ photo[{photo_key}] n={len(buffered)} "
                f"{pose_str} seg_frame={seg_frame_id}"
            ]
            for p in buffered:
                wxy = p.get("world_xy_mm") or [float('nan'), float('nan')]
                pxy = p.get("pick_point_xy_px") or [None, None]
                conf = p.get("confidence")
                conf_str = f"{conf:.2f}" if isinstance(conf, (int, float)) else "?"
                lines.append(
                    f"  pick det={p.get('detection_id')} "
                    f"world_xy_mm=({wxy[0]:.3f}, {wxy[1]:.3f}) "
                    f"px=({pxy[0]}, {pxy[1]}) conf={conf_str}"
                )
            logger.info("\n".join(lines))
        else:
            logger.info(
                "%s @ photo[%s] n=0 %s seg_frame=%s%s",
                batch_label,
                photo_key,
                pose_str,
                seg_frame_id,
                (
                    " (target disappeared or no remaining hairs)"
                    if batch_label == "confirm_world_picks"
                    else " (no hairs at this photo)"
                ),
            )

    def _on_photo_arrived(self, point: PlcPoint) -> None:
        """Fired by ProtocolWorker right before acking nova5 func=11.

        Picks computed while nova5 was in transit may now be in the buffer —
        they're bound to a flange pose that is no longer current. Drop them
        and re-open the batch acceptor so the next TASK:WORLD_PICKS event
        (computed with nova5 parked at the new photo position) gets recorded
        as the canonical pick set for this photo position.

        Also resumes the camera loop: nova5 is now stationary, vision can
        run without pose/frame skew.
        """
        self._close_photo_cycle(point.key)
        self._start_photo_cycle(point.key)
        with self._picks_lock:
            dropped = len(self._world_picks)
            self._world_picks.clear()
            self._accepting_batch = True
            self._batch_received = False
            self._received_was_empty = False
            self._reset_confirmation_locked()
            self._ignored_targets.clear()
        logger.info(
            "photo-arrived %s: flushed %d transit picks, batch acceptor open",
            point.key, dropped,
        )
        self._publish_preview_clear(f"photo arrived {point.key}")
        self._publish_frame_loop_resume(f"nova5 parked at {point.key}")

    def _build_active_press_sequence(self) -> List[int]:
        """Return active press sequence after applying start_press_row filter.

        The 100 press positions are interpreted as a 10x10 matrix in the
        existing route-table order. Under the field definition:
        - X positive direction = row
        - Y positive direction = column
        - traversal order is column-major
        therefore press indices 1..10 are column 1, rows 1..10; 11..20 are
        column 2, rows 1..10; and so on. Each press expands to 7 photo rows
        in the route table.

        Config semantics:
        - ``start_press_row``: 1-based original row to start from; all earlier
          rows are ignored for this run.
        - ``start_press_index``: 1-based sequence within the remaining active
          sub-matrix after those rows are ignored.
        """

        unique_presses: List[int] = []
        seen_presses: set[int] = set()
        for point in self._points:
            press = int(point.press_index)
            if press in seen_presses:
                continue
            seen_presses.add(press)
            unique_presses.append(press)
        if not unique_presses:
            return 0

        rows_per_col = max(1, min(10, len(unique_presses)))
        total_rows = rows_per_col
        try:
            requested_row = int(self._cfg.start_press_row)
        except Exception:  # noqa: BLE001
            requested_row = 1
        requested_row = max(1, min(requested_row, total_rows))

        active_presses = [
            press
            for idx, press in enumerate(unique_presses)
            if ((idx % rows_per_col) + 1) >= requested_row
        ]
        return active_presses

    def _build_active_route_indices(self) -> List[int]:
        active_presses = set(self._build_active_press_sequence())
        route_indices: List[int] = []
        for idx, point in enumerate(self._points):
            if int(point.press_index) in active_presses:
                route_indices.append(idx)
        return route_indices

    def _resolve_start_route_position(
        self,
        route_indices: List[int],
        start_press_index: int,
    ) -> int:
        """Map active-matrix sequence to route position within active route."""
        active_presses = self._build_active_press_sequence()
        if not active_presses or not route_indices:
            return 0
        rows_per_col = max(1, min(10, len({int(p.press_index) for p in self._points})))
        try:
            requested_row = int(self._cfg.start_press_row)
        except Exception:  # noqa: BLE001
            requested_row = 1
        requested_row = max(1, min(requested_row, rows_per_col))

        try:
            requested_sequence = int(start_press_index)
        except Exception:  # noqa: BLE001
            requested_sequence = 1
        requested_sequence = max(1, requested_sequence)
        if requested_sequence > len(active_presses):
            logger.warning(
                "plc_orchestrator start_press_index=%s exceeds active matrix size=%d "
                "(start_press_row=%s); falling back to first active press",
                requested_sequence,
                len(active_presses),
                requested_row,
            )
            requested_sequence = 1

        requested_press = int(active_presses[requested_sequence - 1])
        for route_pos, idx in enumerate(route_indices):
            point = self._points[idx]
            if int(point.press_index) == requested_press:
                logger.info(
                    "plc_orchestrator start_press_row=%s start_press_index=%s "
                    "resolved to press=%s active route point %d/%d (global point %d/%d, %s)",
                    requested_row,
                    requested_sequence,
                    requested_press,
                    route_pos + 1,
                    len(route_indices),
                    idx + 1,
                    len(self._points),
                    point.key,
                )
                return route_pos
        logger.warning(
            "plc_orchestrator resolved press=%s not found in %d route points; falling back to first point",
            requested_press,
            len(route_indices),
        )
        return 0

    def _resolve_start_point_index(self, start_press_index: int) -> int:
        route_indices = self._build_active_route_indices()
        start_route_pos = self._resolve_start_route_position(
            route_indices,
            start_press_index,
        )
        if not route_indices:
            return 0
        return int(route_indices[start_route_pos])

    def _on_nova5_moving(self, point: Optional[PlcPoint], code: int) -> None:
        """Fired by ProtocolWorker before emitting func=21/22/23.

        nova5 is about to start moving (next photo, next press, or retract
        home). Three things to do here, atomically:

        1. Pause the camera loop so we don't waste compute on frames
           captured during deceleration / transit, and so the next photo's
           first vision batch is unambiguously bound to nova5's parked pose.
        2. Drop any picks still in the deque — they were computed for the
           photo nova5 is leaving and must not survive into the next photo.
           This is a defense-in-depth measure: protocol.py also rejects
           epson func=1 while active_photo_key is None (between leaving
           and arriving), but clearing the deque here makes the intent
           explicit and protects against any future strict-order weakening.
        3. Reset batch lifecycle so the next world_picks event for the
           upcoming photo is recorded as the canonical batch.
        """
        from_key = point.key if point else "?"
        self._record_photo_cycle_event(
            "advance_sent",
            advance_code=int(code),
            from_photo=from_key,
        )
        with self._picks_lock:
            dropped = len(self._world_picks)
            self._world_picks.clear()
            self._accepting_batch = False
            self._batch_received = False
            self._received_was_empty = False
            self._reset_confirmation_locked()
            self._ignored_targets.clear()
        if dropped:
            logger.info(
                "nova5-leaving %s: dropped %d picks bound to old photo pose",
                from_key, dropped,
            )
        self._publish_preview_clear(f"nova5 leaving {from_key}")
        self._publish_frame_loop_pause(f"nova5 leaving {from_key} (advance={code})")

    # ---- Frame loop pause/resume helpers ----

    def _publish_frame_loop_pause(self, reason: str) -> None:
        if self._event_bus is None:
            return
        self._event_bus.publish("FRAME_LOOP:PAUSE", {
            "source": self._name,
            "payload": {"reason": reason},
        })

    def _publish_frame_loop_resume(
        self,
        reason: str,
        *,
        pipeline_runs: Optional[int] = None,
    ) -> None:
        if self._event_bus is None:
            return
        payload: Dict[str, Any] = {"reason": reason}
        if pipeline_runs is not None:
            payload["pipeline_runs"] = int(pipeline_runs)
        self._event_bus.publish("FRAME_LOOP:RESUME", {
            "source": self._name,
            "payload": payload,
        })

    def _publish_preview_clear(self, reason: str) -> None:
        if self._event_bus is None:
            return
        self._event_bus.publish("PLC_ORCH:PREVIEW_CLEAR", {
            "source": self._name,
            "payload": {"reason": reason},
        })

    def _has_pending_picks(self) -> bool:
        with self._picks_lock:
            return self._has_pending_picks_for_tool_locked(
                self._worker.get_current_epson_tool()
            )

    def _batch_received_empty(self) -> bool:
        """True iff vision has reported in for the current photo position AND
        the result was 0 picks. Distinguishes a confirmed empty batch from
        the "still waiting for vision" case so the worker can send a skip
        signal instead of stalling forever. Does NOT flip to True after a
        non-empty batch is drained normally — that's the worker's func=21
        path, not a skip."""
        with self._picks_lock:
            if not self._batch_received:
                return False
            if self._dispatched_pick is not None or self._awaiting_confirmation_batch:
                return False
            return not self._has_pending_picks_for_tool_locked(
                self._worker.get_current_epson_tool()
            )

    def _next_task_ready(self) -> bool:
        with self._picks_lock:
            if self._dispatched_pick is None:
                return True
            if self._awaiting_confirmation_batch:
                return False
            self._accepting_batch = True
            self._awaiting_confirmation_batch = True
            self._confirmation_mode = "func21_post_pick"
            self._confirm_started_at = time.perf_counter()
            det_id = self._dispatched_pick.get("detection_id")
            attempt = self._dispatched_pick.get("pick_attempts")
        self._record_photo_cycle_event(
            "post_pick_confirmation_armed",
            detection_id=det_id,
            attempt=attempt,
        )
        logger.info(
            "post-pick confirmation armed: det=%s attempt=%s",
            det_id,
            attempt,
        )
        self._publish_frame_loop_resume("post-pick confirmation")
        return False

    # ---- Provider injected into ProtocolWorker ----

    def _resolve_epson_coord(self, point: PlcPoint, fallback: EpsonCoord) -> Optional[EpsonCoord]:
        """Return one Epson target or None (no pick available).

        Returning None means "do not respond to PLC's func=1 yet" — the
        worker decides whether to wait silently (still expecting a batch)
        or send a skip signal (batch already arrived empty), via the
        ``batch_received_empty`` callback.

        Coordinate strategy (vision-driven, hybrid pixel→world→grid):

        1. PixelToWorldTask has already computed
              flange_target_xy_mm = pick world_xy − (cam_dx, cam_dy)
           which answers "what flange XY would put the optical center
           on this hair". This is the language the grid table speaks
           (anchors store flange XY, not optical-center XY).
        2. Feed flange_target_xy through the nova5↔epson grid for a
           locally-corrected Epson XY. The grid absorbs non-uniform
           mechanical drift across the workspace; the pick pixel has
           already been folded in by step 1, so different hairs at the
           same photo position get different Epson targets.

        Earlier versions used the live nova5 RT XY directly (skipping
        pixel_to_world entirely) on the theory the grid alone could
        compensate the extrinsic bias — it could not, because the
        optical-center vs flange offset (dx, dy) ≈ (-26.68, -21.38) mm
        was being silently absorbed into the per-photo accuracy budget.

        ``fallback`` still provides Z (dip depth) and U (yaw).
        """
        tool_code = self._worker.get_current_epson_tool()
        with self._picks_lock:
            pick = self._pop_next_pick_for_tool_locked(tool_code)
            if pick is None:
                return None
            pick = dict(pick)
            pick["pick_attempts"] = int(pick.get("pick_attempts", 0)) + 1
            attempt = int(pick["pick_attempts"])

        flange_target = pick.get("flange_target_xy_mm")
        if not flange_target or len(flange_target) < 2:
            logger.warning(
                "epson coord refused: pick %s has no flange_target_xy_mm at %s "
                "(pixel_to_world failed or pose unavailable?)",
                pick.get("detection_id"), self._worker.active_photo_key,
            )
            return None
        flange_target_x = float(flange_target[0])
        flange_target_y = float(flange_target[1])

        mapped = self._map_nova5_xy_to_epson(flange_target_x, flange_target_y)
        if mapped is None:
            logger.warning(
                "epson coord refused: nova5_to_epson_mapping disabled or "
                "missing; flange_target_xy_mm=(%.4f, %.4f)",
                flange_target_x, flange_target_y,
            )
            return None

        fallback_u = float(fallback.get("u", 0.0))
        algo_u = pick.get("pick_angle_deg")
        u_src = "fallback"
        selected_u = fallback_u
        if isinstance(algo_u, (int, float)):
            algo_u = float(algo_u)
            selected_u = min(
                max(algo_u, self._cfg.epson_u_min_deg),
                self._cfg.epson_u_max_deg,
            )
            u_src = "algo" if selected_u == algo_u else "algo_clamped"

        selected_z, z_src = self._select_epson_z_for_tool(
            fallback_z=float(fallback.get("z", 0.0)),
            tool_code=tool_code,
            attempt=attempt,
        )

        tool_src = "base"
        x_offset = 0.0
        y_offset = 0.0
        z_offset = 0.0
        if tool_code == 2:
            x_offset = self._cfg.epson_tool2_offset_x_mm
            y_offset = self._cfg.epson_tool2_offset_y_mm
            z_offset = self._cfg.epson_tool2_offset_z_mm
            tool_src = "tool2_offset"

        coord: EpsonCoord = {
            "x": mapped.epson_x + self._cfg.epson_offset_x_mm + x_offset,
            "y": mapped.epson_y + self._cfg.epson_offset_y_mm + y_offset,
            "z": selected_z + z_offset,
            "u": selected_u,
        }

        active_photo = self._worker.active_photo_key or "?"
        nova5_rt = self._worker.get_rt_nova5_xy()
        if nova5_rt is not None:
            nova5_x, nova5_y = nova5_rt
            delta_x = coord["x"] - nova5_x
            delta_y = coord["y"] - nova5_y
            nova5_str = f"nova5_xy_mm=({nova5_x:.4f}, {nova5_y:.4f})"
            delta_str = f"Δ(epson-nova5)=({delta_x:.3f}, {delta_y:.3f})"
        else:
            nova5_str = "nova5_xy_mm=<unavailable>"
            delta_str = "Δ(epson-nova5)=<unavailable>"
        level = (
            logger.warning if mapped.distance_mm > 30.0 else logger.info
        )
        level(
            "epson coord [photo=%s] ← hybrid pixel→world→grid det=%s "
            "world_xy_mm=%s flange_target_xy_mm=(%.4f, %.4f) "
            "%s anchor=(r%s,c%s) "
            "anchor_nova5=(%.4f, %.4f) anchor_epson=(%.4f, %.4f) "
            "dist_to_anchor=%.3fmm offset_mm=(%.4f, %.4f) "
            "epson_xy_mm=(%.4f, %.4f) %s "
            "tool=%s(%s) det_tool=%s tool_src=%s xyz_offset=(%.3f, %.3f, %.3f) "
            "u_src=%s algo_u=%s fallback_u=%.3f u_window=[%.1f, %.1f] "
            "z_src=%s z=%.3f u=%.3f",
            active_photo, pick.get("detection_id"),
            pick.get("world_xy_mm"),
            flange_target_x, flange_target_y,
            nova5_str,
            mapped.anchor.row, mapped.anchor.col,
            mapped.anchor.nova5_x, mapped.anchor.nova5_y,
            mapped.anchor.epson_x, mapped.anchor.epson_y,
            mapped.distance_mm, mapped.offset_x, mapped.offset_y,
            coord["x"], coord["y"], delta_str,
            tool_code, ("tweezer" if tool_code == 1 else "suction" if tool_code == 2 else "unknown"),
            _tool_label_from_code(pick.get("preferred_epson_tool")),
            tool_src, x_offset, y_offset, z_offset,
            u_src,
            (
                f"{algo_u:.3f}"
                if isinstance(algo_u, float)
                else "<unavailable>"
            ),
            fallback_u,
            self._cfg.epson_u_min_deg,
            self._cfg.epson_u_max_deg,
            z_src,
            coord["z"], coord["u"],
        )
        with self._picks_lock:
            self._dispatched_pick = dict(pick)
            self._dispatched_pick["selected_u_deg"] = coord["u"]
            self._dispatched_pick["epson_coord"] = dict(coord)
            self._last_epson_coord = dict(coord)
            self._confirm_frames_remaining = self._confirm_cfg.confirm_window_frames
            self._awaiting_confirmation_batch = False
            self._confirmation_mode = None
            self._post_pick_stage_decision = None
            self._alignment_stage_started_at.clear()
        return coord

    def _map_nova5_xy_to_epson(self, x: float, y: float) -> Optional[ArmGridMatch]:
        if self._nova5_to_epson_mapper is None:
            return None
        return self._nova5_to_epson_mapper.map_nova5_to_epson(x, y)

    def _on_epson_coord_sent(self, point: PlcPoint, coord: EpsonCoord) -> None:
        with self._picks_lock:
            pick = dict(self._dispatched_pick) if self._dispatched_pick is not None else None
            tracking_radius_px = float(self._confirm_cfg.match_distance_threshold)
        self._record_photo_cycle_event(
            "epson_coord_sent",
            photo=point.key,
            x=coord["x"],
            y=coord["y"],
            z=coord["z"],
            u=coord["u"],
            attempt=(pick or {}).get("pick_attempts"),
            detection_id=(pick or {}).get("detection_id"),
        )
        if self._event_bus is None or pick is None:
            return
        self._event_bus.publish("PLC_ORCH:EPSON_TARGET_SENT", {
            "source": self._name,
            "payload": {
                "photo_key": point.key,
                "detection_id": pick.get("detection_id"),
                "attempt": int(pick.get("pick_attempts", 0)),
                "epson_coord": {
                    "x": float(coord["x"]),
                    "y": float(coord["y"]),
                    "z": float(coord["z"]),
                    "u": float(coord["u"]),
                },
                "pick_point_xy_px": pick.get("pick_point_xy_px"),
                "bbox_center_xy_px": pick.get("bbox_center_xy_px"),
                "tracking_radius_px": tracking_radius_px,
            },
        })

    # ---- Epson V2.0 in-motion protocol hooks ----

    def _on_epson_motion_stage(
        self, point: PlcPoint, stage_func: int
    ) -> Optional[EpsonStageDecision]:
        self._record_photo_cycle_event(
            f"epson_func{stage_func}_request", photo=point.key
        )
        if stage_func in (50, 60):
            return self._handle_epson_alignment_request(point, stage_func)
        if stage_func == 70:
            return self._handle_epson_pick_result_request(point)
        logger.warning("unsupported Epson motion stage func=%s at %s", stage_func, point.key)
        return None

    def _handle_epson_alignment_request(
        self, point: PlcPoint, stage_func: int
    ) -> Optional[EpsonStageDecision]:
        """Check tweezer predicted tip vs. current target and optionally correct XY.

        stage_func=50 checks at the pick waiting position and returns PC func=50
        when aligned or func=51 + corrected coord when not aligned.

        stage_func=60 checks after Z has descended and returns PC func=60 when
        aligned or func=61 + corrected coord when not aligned.
        """
        now = time.time()
        publish_resume = False
        with self._picks_lock:
            pick = dict(self._dispatched_pick) if self._dispatched_pick is not None else None
            coord = dict(self._last_epson_coord) if self._last_epson_coord is not None else None
            if coord is None and pick is not None and isinstance(pick.get("epson_coord"), dict):
                coord = dict(pick["epson_coord"])
            stage_started_at = self._alignment_stage_started_at.get(stage_func)
            if stage_started_at is None:
                stage_started_at = now
                self._alignment_stage_started_at[stage_func] = stage_started_at
                publish_resume = True

        if pick is None or coord is None:
            logger.warning(
                "epson func=%s alignment check requested at %s but no active dispatched pick/coord",
                stage_func, point.key,
            )
            return None

        if publish_resume:
            self._publish_frame_loop_resume(
                f"epson func{stage_func} alignment check at {point.key}",
                pipeline_runs=1,
            )

        target_xy = _pick_target_xy(pick)
        if target_xy is None:
            logger.warning(
                "epson func=%s alignment refused: dispatched pick has no pixel target: %s",
                stage_func, pick.get("detection_id"),
            )
            return None

        snapshot = self._live_state.snapshot()
        updated_at = float(snapshot.get("updated_at") or 0.0)
        timeout_s = max(0.1, float(self._cfg.epson_alignment_snapshot_timeout_ms) / 1000.0)
        if updated_at < stage_started_at:
            if now - stage_started_at > timeout_s and now - self._alignment_last_wait_log_at > 1.0:
                self._alignment_last_wait_log_at = now
                logger.warning(
                    "epson func=%s alignment waiting for fresh tweezer frame at %s: age=%.3fs",
                    stage_func, point.key, max(0.0, now - updated_at),
                )
                self._publish_frame_loop_resume(
                    f"epson func{stage_func} alignment retry at {point.key}",
                    pipeline_runs=1,
                )
            return None

        tweezer = snapshot.get("tweezer") if isinstance(snapshot, dict) else None
        if not isinstance(tweezer, dict) or not tweezer.get("found"):
            if now - self._alignment_last_wait_log_at > 1.0:
                self._alignment_last_wait_log_at = now
                logger.warning(
                    "epson func=%s alignment waiting: tweezer tip not found at %s",
                    stage_func, point.key,
                )
            return None
        tip_xy = tweezer.get("tip_xy")
        if not isinstance(tip_xy, (list, tuple)) or len(tip_xy) < 2:
            return None

        tip_x = float(tip_xy[0])
        tip_y = float(tip_xy[1])
        target_x = float(target_xy[0])
        target_y = float(target_xy[1])
        error_px_x = target_x - tip_x
        error_px_y = target_y - tip_y
        dist_px = math.hypot(error_px_x, error_px_y)
        threshold_px = float(self._cfg.epson_alignment_match_distance_px)

        if dist_px <= threshold_px:
            with self._picks_lock:
                self._alignment_stage_started_at.pop(stage_func, None)
            logger.info(
                "epson func=%s aligned at %s: det=%s tip=(%.2f, %.2f) "
                "target=(%.2f, %.2f) dist=%.2fpx <= %.2fpx",
                stage_func, point.key, pick.get("detection_id"),
                tip_x, tip_y, target_x, target_y, dist_px, threshold_px,
            )
            return {
                "code": int(stage_func),
                "coord": None,
                "desc": f"epson func{stage_func} aligned at {point.key}",
            }

        dx_mm = error_px_x * float(self._cfg.epson_alignment_x_mm_per_px)
        dy_mm = error_px_y * float(self._cfg.epson_alignment_y_mm_per_px)
        max_corr = float(self._cfg.epson_alignment_max_correction_mm)
        corr_len = math.hypot(dx_mm, dy_mm)
        if max_corr > 0.0 and corr_len > max_corr:
            scale = max_corr / corr_len
            dx_mm *= scale
            dy_mm *= scale

        corrected: EpsonCoord = {
            "x": float(coord["x"]) + dx_mm,
            "y": float(coord["y"]) + dy_mm,
            "z": float(coord["z"]),
            "u": float(coord["u"]),
        }
        with self._picks_lock:
            self._last_epson_coord = dict(corrected)
            if self._dispatched_pick is not None:
                self._dispatched_pick["epson_coord"] = dict(corrected)
            self._alignment_stage_started_at.pop(stage_func, None)

        response_code = 51 if stage_func == 50 else 61
        logger.warning(
            "epson func=%s not aligned at %s: det=%s tip=(%.2f, %.2f) "
            "target=(%.2f, %.2f) error_px=(%.2f, %.2f) dist=%.2fpx > %.2fpx; "
            "send func=%s correction dxy_mm=(%.4f, %.4f) coord=(%.4f, %.4f, %.4f, %.4f)",
            stage_func, point.key, pick.get("detection_id"),
            tip_x, tip_y, target_x, target_y, error_px_x, error_px_y,
            dist_px, threshold_px, response_code, dx_mm, dy_mm,
            corrected["x"], corrected["y"], corrected["z"], corrected["u"],
        )
        return {
            "code": response_code,
            "coord": corrected,
            "desc": f"epson func{stage_func} XY correction at {point.key}",
        }

    def _handle_epson_pick_result_request(
        self, point: PlcPoint
    ) -> Optional[EpsonStageDecision]:
        """PLC func=70: confirm whether the foreign object was picked.

        The old disappearance-confirmation logic is reused, but the result is
        returned immediately through the new V2.0 func=70/71 handshake:
        - PC func=70: object disappeared → PLC may discard it.
        - PC func=71: object still exists → mark this attempt failed and skip
          this object; remaining objects at the same photo position stay queued.
        """
        publish_resume = False
        with self._picks_lock:
            if self._post_pick_stage_decision is not None:
                decision = dict(self._post_pick_stage_decision)
                self._post_pick_stage_decision = None
                return decision

            if self._dispatched_pick is None:
                logger.warning("epson func=70 at %s but no active dispatched pick", point.key)
                return {
                    "code": 71,
                    "coord": None,
                    "desc": f"epson func70 no active pick at {point.key}",
                }

            if not self._awaiting_confirmation_batch:
                self._accepting_batch = True
                self._awaiting_confirmation_batch = True
                self._confirmation_mode = "func70_pick_result"
                self._confirm_started_at = time.perf_counter()
                self._confirm_frames_remaining = self._confirm_cfg.confirm_window_frames
                publish_resume = True
                det_id = self._dispatched_pick.get("detection_id")
                attempt = self._dispatched_pick.get("pick_attempts")
            else:
                det_id = self._dispatched_pick.get("detection_id")
                attempt = self._dispatched_pick.get("pick_attempts")

        if publish_resume:
            self._record_photo_cycle_event(
                "func70_pick_result_confirmation_armed",
                detection_id=det_id,
                attempt=attempt,
            )
            logger.info(
                "func70 pick-result confirmation armed: photo=%s det=%s attempt=%s frames=%s",
                point.key, det_id, attempt, self._confirm_cfg.confirm_window_frames,
            )
            self._publish_frame_loop_resume(
                f"epson func70 pick-result confirmation at {point.key}",
                pipeline_runs=self._confirm_cfg.confirm_window_frames,
            )
        return None

    def _finish_func70_confirmation_locked(self, decision: EpsonStageDecision) -> None:
        """Store func70/71 decision while clearing the active pick state.

        Do not call _reset_confirmation_locked() here because the ProtocolWorker
        still needs to read _post_pick_stage_decision on its next poll and send
        it to PLC.
        """
        self._post_pick_stage_decision = dict(decision)
        self._dispatched_pick = None
        self._last_epson_coord = None
        self._confirm_frames_remaining = 0
        self._awaiting_confirmation_batch = False
        self._confirmation_mode = None
        self._confirm_started_at = None
        self._alignment_stage_started_at.clear()

    # ---- Diagnostics ----

    def snapshot(self) -> Dict[str, Any]:
        snap = self._worker.snapshot()
        with self._picks_lock:
            snap["pending_world_picks"] = len(self._world_picks)
            snap["last_seg_frame_id"] = self._last_seg_frame_id
            snap["confirm_frames_remaining"] = self._confirm_frames_remaining
            snap["awaiting_confirmation_batch"] = self._awaiting_confirmation_batch
            snap["ignored_targets"] = len(self._ignored_targets)
            snap["dispatched_detection_id"] = (
                self._dispatched_pick.get("detection_id")
                if self._dispatched_pick is not None
                else None
            )
        return snap

    def _replace_buffer_locked(self, picks: List[Dict[str, Any]]) -> None:
        self._world_picks.clear()
        for pick in picks:
            if pick.get("world_xy_mm") and not self._is_ignored_pick_locked(pick):
                self._world_picks.append(dict(pick))

    def _has_pending_picks_for_tool_locked(self, tool_code: int) -> bool:
        for pick in self._world_picks:
            if _pick_matches_tool(pick, tool_code):
                return True
        return False

    def _pop_next_pick_for_tool_locked(self, tool_code: int) -> Optional[Dict[str, Any]]:
        if not self._world_picks:
            return None
        skipped: list[Dict[str, Any]] = []
        matched: Optional[Dict[str, Any]] = None
        while self._world_picks:
            candidate = self._world_picks.popleft()
            if matched is None and _pick_matches_tool(candidate, tool_code):
                matched = candidate
                break
            skipped.append(candidate)
        while skipped:
            self._world_picks.appendleft(skipped.pop())
        return matched

    def _select_epson_z_for_tool(
        self,
        *,
        fallback_z: float,
        tool_code: int,
        attempt: int,
    ) -> tuple[float, str]:
        if tool_code == 2:
            selected_z = (
                self._cfg.epson_suction_z_mm
                if self._cfg.epson_suction_z_mm is not None
                else (
                    self._cfg.epson_z_mm
                    if self._cfg.epson_z_mm is not None
                    else fallback_z
                )
            )
            z_src = (
                "suction_cfg"
                if self._cfg.epson_suction_z_mm is not None
                else "legacy_cfg"
                if self._cfg.epson_z_mm is not None
                else "fallback"
            )
            if attempt == 2:
                selected_z -= self._cfg.epson_suction_z_retry2_offset_mm
            elif attempt >= 3:
                selected_z -= self._cfg.epson_suction_z_retry3_offset_mm
            return float(selected_z), z_src

        selected_z = (
            self._cfg.epson_tweezer_z_mm
            if self._cfg.epson_tweezer_z_mm is not None
            else (
                self._cfg.epson_z_mm
                if self._cfg.epson_z_mm is not None
                else fallback_z
            )
        )
        z_src = (
            "tweezer_cfg"
            if self._cfg.epson_tweezer_z_mm is not None
            else "legacy_cfg"
            if self._cfg.epson_z_mm is not None
            else "fallback"
        )
        if (
            self._cfg.epson_tweezer_z_mm is not None
            or self._cfg.epson_z_mm is not None
        ):
            if attempt == 2:
                selected_z += self._cfg.epson_tweezer_z_retry2_offset_mm
            elif attempt >= 3:
                selected_z += self._cfg.epson_tweezer_z_retry3_offset_mm
        return float(selected_z), z_src

    def _reset_confirmation_locked(self) -> None:
        self._dispatched_pick = None
        self._last_epson_coord = None
        self._confirm_frames_remaining = 0
        self._awaiting_confirmation_batch = False
        self._confirmation_mode = None
        self._post_pick_stage_decision = None
        self._confirm_started_at = None
        self._alignment_stage_started_at.clear()

    def _confirmation_elapsed_ms_locked(self) -> Optional[float]:
        if self._confirm_started_at is None:
            return None
        return round((time.perf_counter() - self._confirm_started_at) * 1000.0, 2)

    def _handle_confirmation_batch_locked(
        self,
        picks: List[Dict[str, Any]],
        *,
        seg_frame_id: Any,
    ) -> bool:
        """Process one stabilized confirmation batch and decide immediately.

        For seg_pick_stabilized, one confirmation round already consists of
        N frames (e.g. 5-in-3 or 7-in-4). If the target is still present
        after that stabilized batch, we should immediately queue a retry
        instead of silently requesting many more confirmation rounds.
        """
        if self._confirmation_mode == "func70_pick_result":
            return self._handle_func70_pick_result_batch_locked(
                picks,
                seg_frame_id=seg_frame_id,
            )

        self._awaiting_confirmation_batch = False
        self._accepting_batch = False
        self._batch_received = True
        self._received_was_empty = False
        self._last_seg_frame_id = seg_frame_id

        current_picks = [
            dict(p)
            for p in picks
            if p.get("world_xy_mm") and not self._is_ignored_pick_locked(p)
        ]
        target = self._dispatched_pick
        if target is None:
            self._replace_buffer_locked(current_picks)
            return False

        match_index = self._find_matching_pick_index(target, current_picks)
        if match_index is None:
            self._replace_buffer_locked(current_picks)
            confirm_total_ms = self._confirmation_elapsed_ms_locked()
            self._record_photo_cycle_event(
                "post_pick_confirmed",
                detection_id=target.get("detection_id"),
                attempt=target.get("pick_attempts"),
                confirm_total_ms=confirm_total_ms,
            )
            logger.info(
                "post-pick confirmed: det=%s attempt=%s disappeared confirm_total_ms=%s",
                target.get("detection_id"),
                target.get("pick_attempts"),
                f"{confirm_total_ms:.2f}" if confirm_total_ms is not None else "<unknown>",
            )
            self._reset_confirmation_locked()
            return False

        matched = dict(current_picks[match_index])
        matched["pick_attempts"] = int(target.get("pick_attempts", 0))
        remaining_picks = [
            dict(p)
            for idx, p in enumerate(current_picks)
            if idx != match_index
        ]

        if int(matched.get("pick_attempts", 0)) >= self._confirm_cfg.max_pick_attempts:
            self._ignored_targets.append(dict(matched))
            self._world_picks.clear()
            for pick in remaining_picks:
                self._world_picks.append(pick)
            confirm_total_ms = self._confirmation_elapsed_ms_locked()
            self._record_photo_cycle_event(
                "post_pick_abandoned",
                detection_id=matched.get("detection_id"),
                attempt=matched.get("pick_attempts"),
                confirm_total_ms=confirm_total_ms,
                remaining=len(self._world_picks),
            )
            logger.warning(
                "post-pick abandoned: det=%s attempt=%s max_attempts=%s confirm_total_ms=%s",
                matched.get("detection_id"),
                matched.get("pick_attempts"),
                self._confirm_cfg.max_pick_attempts,
                f"{confirm_total_ms:.2f}" if confirm_total_ms is not None else "<unknown>",
            )
        else:
            self._world_picks.clear()
            self._world_picks.append(matched)
            for pick in remaining_picks:
                self._world_picks.append(pick)
            confirm_total_ms = self._confirmation_elapsed_ms_locked()
            self._record_photo_cycle_event(
                "post_pick_retry_queued",
                detection_id=matched.get("detection_id"),
                attempt=matched.get("pick_attempts"),
                confirm_total_ms=confirm_total_ms,
                remaining=len(self._world_picks),
            )
            logger.warning(
                "post-pick retry queued: det=%s attempt=%s remaining=%d confirm_total_ms=%s",
                matched.get("detection_id"),
                matched.get("pick_attempts"),
                len(self._world_picks),
                f"{confirm_total_ms:.2f}" if confirm_total_ms is not None else "<unknown>",
            )
        self._reset_confirmation_locked()
        return False

    def _handle_func70_pick_result_batch_locked(
        self,
        picks: List[Dict[str, Any]],
        *,
        seg_frame_id: Any,
    ) -> bool:
        """Handle the new PLC func=70 pick-result confirmation batch.

        Unlike the old func=21 confirmation, func=70 does not retry the same
        object. If the target still exists, reply PC func=71 and ignore that
        target for the rest of the current photo position; then the remaining
        objects can continue normally.
        """
        self._awaiting_confirmation_batch = False
        self._accepting_batch = False
        self._batch_received = True
        self._received_was_empty = False
        self._last_seg_frame_id = seg_frame_id

        current_picks = [
            dict(p)
            for p in picks
            if p.get("world_xy_mm") and not self._is_ignored_pick_locked(p)
        ]
        target = self._dispatched_pick
        confirm_total_ms = self._confirmation_elapsed_ms_locked()
        if target is None:
            self._replace_buffer_locked(current_picks)
            decision: EpsonStageDecision = {
                "code": 71,
                "coord": None,
                "desc": "func70 confirmation had no active dispatched pick",
            }
            self._finish_func70_confirmation_locked(decision)
            logger.warning("func70 confirmation finished with no active dispatched pick")
            return False

        match_index = self._find_matching_pick_index(target, current_picks)
        if match_index is None:
            # Target disappeared: successful clamp/pick. Keep any newly visible
            # remaining objects, and tell PLC to discard the clamped foreign body.
            self._replace_buffer_locked(current_picks)
            decision = {
                "code": 70,
                "coord": None,
                "desc": f"pick success at {self._worker.active_photo_key or '?'}",
            }
            self._record_photo_cycle_event(
                "func70_pick_success",
                detection_id=target.get("detection_id"),
                attempt=target.get("pick_attempts"),
                confirm_total_ms=confirm_total_ms,
                remaining=len(self._world_picks),
            )
            logger.info(
                "func70 pick success: det=%s attempt=%s disappeared confirm_total_ms=%s remaining=%d",
                target.get("detection_id"),
                target.get("pick_attempts"),
                f"{confirm_total_ms:.2f}" if confirm_total_ms is not None else "<unknown>",
                len(self._world_picks),
            )
            self._finish_func70_confirmation_locked(decision)
            return False

        matched = dict(current_picks[match_index])
        matched["pick_attempts"] = int(target.get("pick_attempts", 0))
        remaining_picks = [
            dict(p)
            for idx, p in enumerate(current_picks)
            if idx != match_index
        ]
        self._ignored_targets.append(dict(matched))
        self._world_picks.clear()
        for pick in remaining_picks:
            if not self._is_ignored_pick_locked(pick):
                self._world_picks.append(pick)

        decision = {
            "code": 71,
            "coord": None,
            "desc": f"pick failed at {self._worker.active_photo_key or '?'}",
        }
        self._record_photo_cycle_event(
            "func70_pick_failed_skip",
            detection_id=matched.get("detection_id"),
            attempt=matched.get("pick_attempts"),
            confirm_total_ms=confirm_total_ms,
            remaining=len(self._world_picks),
        )
        logger.warning(
            "func70 pick failed: det=%s attempt=%s still present; reply func=71 and skip target; "
            "confirm_total_ms=%s remaining=%d",
            matched.get("detection_id"),
            matched.get("pick_attempts"),
            f"{confirm_total_ms:.2f}" if confirm_total_ms is not None else "<unknown>",
            len(self._world_picks),
        )
        self._finish_func70_confirmation_locked(decision)
        return False

    def _find_matching_pick_index(
        self,
        target: Dict[str, Any],
        picks: List[Dict[str, Any]],
    ) -> Optional[int]:
        best_index: Optional[int] = None
        best_distance = float("inf")
        for idx, pick in enumerate(picks):
            matched, distance = self._match_pick_to_target(target, pick)
            if not matched:
                continue
            if distance < best_distance:
                best_distance = distance
                best_index = idx

        return best_index

    def _is_ignored_pick_locked(self, pick: Dict[str, Any]) -> bool:
        for target in self._ignored_targets:
            matched, _distance = self._match_pick_to_target(target, pick)
            if matched:
                return True
        return False

    def _match_pick_to_target(
        self,
        target: Dict[str, Any],
        pick: Dict[str, Any],
    ) -> tuple[bool, float]:
        target_center = target.get("bbox_center_xy_px")
        center = pick.get("bbox_center_xy_px")
        if not isinstance(target_center, list) or len(target_center) < 2:
            return False, float("inf")
        if not isinstance(center, list) or len(center) < 2:
            return False, float("inf")

        distance = math.hypot(
            float(center[0]) - float(target_center[0]),
            float(center[1]) - float(target_center[1]),
        )
        if distance > self._confirm_cfg.match_distance_threshold:
            return False, distance

        target_width = float(target.get("bbox_width_px") or 0.0)
        target_height = float(target.get("bbox_height_px") or 0.0)
        width = float(pick.get("bbox_width_px") or 0.0)
        height = float(pick.get("bbox_height_px") or 0.0)
        width_ratio = abs(width - target_width) / max(target_width, 1.0)
        height_ratio = abs(height - target_height) / max(target_height, 1.0)
        if width_ratio > self._confirm_cfg.match_size_ratio_threshold:
            return False, distance
        if height_ratio > self._confirm_cfg.match_size_ratio_threshold:
            return False, distance
        return True, distance

    def _on_epson_pick_request(self, point: PlcPoint) -> None:
        self._record_photo_cycle_event("epson_func1_request", photo=point.key)

    def _on_epson_next_task_request(self, point: PlcPoint) -> None:
        self._record_photo_cycle_event("epson_func21_request", photo=point.key)

    def _on_nova5_coord_sent(self, point: PlcPoint) -> None:
        self._record_photo_cycle_event("nova5_coord_sent", next_photo=point.key)

    def _start_photo_cycle(self, photo_key: str) -> None:
        now = time.perf_counter()
        with self._cycle_timing_lock:
            self._photo_cycle_timing = {
                "photo_key": photo_key,
                "started_at": now,
                "events": [
                    {
                        "event": "photo_arrived",
                        "at": now,
                        "details": {"photo": photo_key},
                    }
                ],
            }

    def _close_photo_cycle(self, next_photo_key: str) -> None:
        now = time.perf_counter()
        with self._cycle_timing_lock:
            cycle = self._photo_cycle_timing
            if cycle is None:
                return
            current_photo_key = str(cycle.get("photo_key") or "?")
            if current_photo_key == str(next_photo_key):
                return
            cycle["events"].append(
                {
                    "event": "next_photo_arrived",
                    "at": now,
                    "details": {"next_photo": next_photo_key},
                }
            )
            started_at = float(cycle.get("started_at") or now)
            prev_at = started_at
            rows: List[Dict[str, Any]] = []
            for event in cycle.get("events", []):
                at = float(event.get("at") or prev_at)
                rows.append(
                    {
                        "event": event.get("event"),
                        "dt_ms": round((at - prev_at) * 1000.0, 2),
                        "elapsed_ms": round((at - started_at) * 1000.0, 2),
                        **dict(event.get("details") or {}),
                    }
                )
                prev_at = at
            total_ms = round((now - started_at) * 1000.0, 2)
            logger.info(
                "photo-cycle timing photo=%s next_photo=%s total_ms=%.2f timeline=%s",
                current_photo_key,
                next_photo_key,
                total_ms,
                rows,
            )
            self._photo_cycle_timing = None

    def _record_photo_cycle_event(self, event_name: str, **details: Any) -> None:
        now = time.perf_counter()
        with self._cycle_timing_lock:
            if self._photo_cycle_timing is None:
                return
            self._photo_cycle_timing.setdefault("events", []).append(
                {
                    "event": event_name,
                    "at": now,
                    "details": details,
                }
            )



def _pick_target_xy(pick: Dict[str, Any]) -> Optional[tuple[float, float]]:
    """Return the pixel point used for tweezer-target alignment.

    Prefer the actual algorithm pick point; fall back to bbox center so older
    pick payloads still work.
    """
    for key in ("pick_point_xy_px", "bbox_center_xy_px"):
        value = pick.get(key)
        if isinstance(value, (list, tuple)) and len(value) >= 2:
            try:
                return float(value[0]), float(value[1])
            except Exception:  # noqa: BLE001
                continue
    return None

def _pick_matches_tool(pick: Dict[str, Any], tool_code: int) -> bool:
    preferred = pick.get("preferred_epson_tool")
    try:
        preferred_code = int(preferred)
    except Exception:  # noqa: BLE001
        return True
    return preferred_code == int(tool_code)


def _tool_label_from_code(tool_code: Any) -> str:
    try:
        code = int(tool_code)
    except Exception:  # noqa: BLE001
        return "unknown"
    if code == 1:
        return "tweezers"
    if code == 2:
        return "suck"
    return "unknown"
