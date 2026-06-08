"""Headless ModbusTCP protocol worker for the 3-arm PLC orchestrator.

Ported from the validated standalone test GUI
``plc_modbus_auto_test_gui_v2_strict.py`` — business logic untouched, but:

- All ``tk``/``queue`` GUI plumbing removed; status updates go through
  ``logger`` instead.
- Names ``press`` / ``vision`` / ``pick`` renamed to ``nova2`` / ``nova5``
  / ``epson_ls6`` to match the actual robot models on the rig.
- The Epson LS6 target coord is supplied by an injected callable so
  vision picks (``TASK:WORLD_PICKS``) can override the points-table
  fallback at request time.

The strict-order protection is preserved verbatim — it's the property
that took the longest to get right on hardware:

1. nova5 can only receive a coord after nova2 has acked the current press
   position.
2. Epson can only receive a coord when nova5 has acked the matching
   photo position.
3. Repeats stay on the same point; advancing to next photo only happens
   after the configured repeat count is exhausted; advancing to next
   press position only happens after all 7 photo positions of the current
   press cycle finish.
"""

from __future__ import annotations

import logging
import math
import queue
import threading
import time
import traceback
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

from src.comm.plc_modbus import ModbusHelper
from src.config import PlcOrchestratorConfig

from .points import PlcPoint

logger = logging.getLogger(__name__)

# 4-axis epson coord supplied to the worker (x/y/z/u).
EpsonCoord = Dict[str, float]
EpsonCoordProvider = Callable[[PlcPoint, EpsonCoord], Optional[EpsonCoord]]
"""Resolve epson_ls6 target. Args: (current_point, fallback_coord). Return
a 4-axis dict {x,y,z,u} or None to mean "no override, use fallback"."""

PhotoArrivedHook = Callable[[PlcPoint], None]
"""Fired right before the worker acks nova5 func=11 (local photo arrived).
Use this to flush any vision picks computed while nova5 was still moving
between photo positions — they are stale by the time Epson asks for a coord."""

PendingPicksQuery = Callable[[], bool]
"""Return True if the orchestrator still has un-served vision picks for the
current photo position. The worker uses this to decide between sending
func=20 (continue picking at same position) vs. func=21/22/23 (advance)."""

BatchEmptyQuery = Callable[[], bool]
"""Return True iff a vision batch has been received for the current photo
position AND it contained 0 picks. Distinct from "no batch yet" (False)
so the worker can skip a hairless photo position instead of stalling."""

NextTaskReadyQuery = Callable[[], bool]
"""Return True when Epson's func=21 can be answered immediately.

Allows higher-level orchestrators to stall the "what next?" response while
they run post-pick confirmation logic on one or more fresh camera frames at
the same photo position.
"""

EpsonStageDecision = Dict[str, object]
"""Decision returned by the high-level orchestrator for Epson in-motion stages.

Expected keys:
    code: int
        Function code to return to PLC (50/51/60/61/70/71).
    coord: Optional[EpsonCoord]
        Present only for correction responses 51/61. X/Y are corrected; Z/U
        remain the current target's Z/U.
    desc: str
        Human-readable reason for logging.
"""

EpsonMotionStageHook = Callable[[PlcPoint, int], Optional[EpsonStageDecision]]
"""Handle Epson in-motion protocol requests.

PLC → PC func=50: Epson arrived at pick waiting position. The hook checks
tweezer-tip alignment and returns PC func=50 (aligned) or func=51 + corrected
coord (XY correction).

PLC → PC func=60: Epson arrived at pick-down position. The hook checks
alignment again and returns PC func=60 (pick foreign object) or func=61 +
corrected coord (XY correction).

PLC → PC func=70: Epson has clamped and moved aside. The hook confirms whether
the foreign object disappeared and returns PC func=70 (discard) or func=71
(pick failed, skip current object).

Return None to keep silent while the vision/tweezer check is still waiting for
a fresh frame.
"""

NovaMovingHook = Callable[[Optional[PlcPoint], int], None]
"""Fired right before the worker emits an advance code (21/22/23). Args:
(point_just_finished, advance_code). Use this to signal downstream
consumers — e.g. the camera loop should pause while nova5 is in transit."""

EpsonCoordSentHook = Callable[[PlcPoint, EpsonCoord], None]
"""Fired after Epson coord write succeeds and echo matches."""

EpsonPickRequestHook = Callable[[PlcPoint], None]
"""Fired when PLC issues Epson func=1 for the current photo position."""

EpsonNextTaskRequestHook = Callable[[PlcPoint], None]
"""Fired when PLC issues Epson func=21 ("what next?") for the current photo."""

Nova5CoordSentHook = Callable[[PlcPoint], None]
"""Fired after a nova5 photo-position coord write succeeds and echo matches."""


_ROBOT_LABEL = {
    "nova2": "Nova2 press arm",
    "nova5": "Nova5 vision arm",
    "epson_ls6": "Epson LS6 pick arm",
}


def _describe_epson_tool(tool_code: int) -> str:
    if tool_code == 1:
        return "tweezer"
    if tool_code == 2:
        return "suction"
    return "unknown"


@dataclass
class _RequestState:
    flag: float = 0.0
    func: float = 0.0
    pc_flag: float = 0.0
    last_key: str = ""
    handled: bool = False


def _default_epson_provider(point: PlcPoint, fallback: EpsonCoord) -> Optional[EpsonCoord]:
    return fallback


def _default_photo_arrived(_point: PlcPoint) -> None:
    pass


def _default_has_pending_picks() -> bool:
    # No vision integration → behave like the original repeat=1 test rig:
    # one pick per photo position then advance.
    return False


def _default_batch_received_empty() -> bool:
    # No vision integration → never report "empty batch" so the worker
    # falls into the silent-wait path; combined with the default epson
    # provider returning the YAML fallback, that means a coord is always
    # available and the wait never triggers.
    return False


def _default_next_task_ready() -> bool:
    return True


def _default_epson_motion_stage(
    _point: PlcPoint, _stage_func: int
) -> Optional[EpsonStageDecision]:
    return None


def _default_nova5_moving(_point: Optional[PlcPoint], _code: int) -> None:
    pass


def _default_epson_coord_sent(_point: PlcPoint, _coord: EpsonCoord) -> None:
    pass


def _default_epson_pick_request(_point: PlcPoint) -> None:
    pass


def _default_epson_next_task_request(_point: PlcPoint) -> None:
    pass


def _default_nova5_coord_sent(_point: PlcPoint) -> None:
    pass


class ProtocolWorker(threading.Thread):
    """Long-running thread that owns one PLC TCP connection.

    Lifecycle:
        worker = ProtocolWorker(cfg, points)
        worker.start()
        worker.connect_now()
        worker.start_auto()
        ...
        worker.stop()         # joins
    """

    def __init__(
        self,
        cfg: PlcOrchestratorConfig,
        points: List[PlcPoint],
        *,
        epson_coord_provider: EpsonCoordProvider = _default_epson_provider,
        on_photo_arrived: PhotoArrivedHook = _default_photo_arrived,
        has_pending_picks: PendingPicksQuery = _default_has_pending_picks,
        batch_received_empty: BatchEmptyQuery = _default_batch_received_empty,
        next_task_ready: NextTaskReadyQuery = _default_next_task_ready,
        on_epson_motion_stage: EpsonMotionStageHook = _default_epson_motion_stage,
        on_nova5_moving: NovaMovingHook = _default_nova5_moving,
        on_epson_coord_sent: EpsonCoordSentHook = _default_epson_coord_sent,
        on_epson_pick_request: EpsonPickRequestHook = _default_epson_pick_request,
        on_epson_next_task_request: EpsonNextTaskRequestHook = _default_epson_next_task_request,
        on_nova5_coord_sent: Nova5CoordSentHook = _default_nova5_coord_sent,
    ) -> None:
        if not points:
            raise ValueError("ProtocolWorker requires at least one point")
        super().__init__(name="plc-protocol-worker", daemon=True)
        self.cfg = cfg
        self.points = points
        self.mb = ModbusHelper(cfg)
        self.epson_coord_provider = epson_coord_provider
        self.on_photo_arrived = on_photo_arrived
        self.has_pending_picks = has_pending_picks
        self.batch_received_empty = batch_received_empty
        self.next_task_ready = next_task_ready
        self.on_epson_motion_stage = on_epson_motion_stage
        self.on_nova5_moving = on_nova5_moving
        self.on_epson_coord_sent = on_epson_coord_sent
        self.on_epson_pick_request = on_epson_pick_request
        self.on_epson_next_task_request = on_epson_next_task_request
        self.on_nova5_coord_sent = on_nova5_coord_sent

        self._stop_event = threading.Event()
        self._command_queue: "queue.Queue[Tuple[str, dict]]" = queue.Queue()
        self._status_lock = threading.Lock()
        self._status: Dict[str, object] = {
            "connected": False,
            "auto_running": False,
            "current_index": 0,
        }

        self.connected = False
        self.last_heartbeat = 0.0
        self.last_poll = 0.0
        self.heartbeat_value = 0
        self.requests: Dict[str, _RequestState] = {
            "nova2": _RequestState(),
            "nova5": _RequestState(),
            "epson_ls6": _RequestState(),
        }
        self.auto_running = False
        self.current_index = 0
        self.start_index = 0
        self.route_indices: List[int] = list(range(len(points)))
        self.start_route_pos = 0
        self.current_route_pos = 0
        self.pick_sent_count = 0
        self.current_epson_tool = 1
        self.completed_cycle = False
        self.repeat_map: Dict[str, int] = {p.key: p.repeat for p in points}

        # Strict-order protection state:
        # active_press_index — press position nova2 has currently acked
        # active_photo_key   — photo position key nova5 has currently acked
        # Epson can only return a coord when both match the current point.
        self.active_press_index: Optional[int] = None
        self.active_photo_key: Optional[str] = None

    # ---- Public API ----

    def stop(self) -> None:
        self._stop_event.set()
        try:
            self.join(timeout=3.0)
        except RuntimeError:
            pass

    def connect_now(self) -> None:
        self._command_queue.put(("connect", {}))

    def disconnect_now(self) -> None:
        self._command_queue.put(("disconnect", {}))

    def start_auto(
        self,
        repeats: Optional[Dict[str, int]] = None,
        start_index: int = 0,
        route_indices: Optional[List[int]] = None,
        start_route_pos: int = 0,
    ) -> None:
        self._command_queue.put(
            (
                "start_auto",
                {
                    "repeats": repeats or {},
                    "start_index": start_index,
                    "route_indices": route_indices or [],
                    "start_route_pos": start_route_pos,
                },
            )
        )

    def stop_auto(self) -> None:
        self._command_queue.put(("stop_auto", {}))

    def reset_auto(self) -> None:
        self._command_queue.put(("reset_auto", {}))

    def snapshot(self) -> Dict[str, object]:
        """Thread-safe shallow copy of the latest status dict."""
        with self._status_lock:
            return dict(self._status)

    def get_rt_nova5_xy(self) -> Optional[Tuple[float, float]]:
        """Last-polled nova5 (x, y) in mm — None if not connected yet."""
        snap = self.snapshot()
        rt = snap.get("rt") or {}
        nova5 = rt.get("nova5") if isinstance(rt, dict) else None
        if not nova5 or len(nova5) < 2:
            return None
        x, y = nova5[0], nova5[1]
        if math.isnan(x) or math.isnan(y):
            return None
        return float(x), float(y)

    def get_current_epson_tool(self) -> int:
        """Last-polled Epson tool code. 1=tweezer, 2=suction head."""
        snap = self.snapshot()
        raw = snap.get("epson_tool")
        try:
            return int(raw)
        except Exception:  # noqa: BLE001
            return int(self.current_epson_tool)

    # ---- Thread loop ----

    def run(self) -> None:
        logger.info("PLC protocol worker thread starting")
        try:
            while not self._stop_event.is_set():
                try:
                    self._handle_commands()
                    if self.connected:
                        now = time.time()
                        if now - self.last_heartbeat >= max(self.cfg.heartbeat_ms / 1000.0, 0.1):
                            self._write_heartbeat()
                            self.last_heartbeat = now
                        if now - self.last_poll >= max(self.cfg.poll_ms / 1000.0, 0.05):
                            self._poll_once()
                            self.last_poll = now
                    time.sleep(0.02)
                except Exception as exc:  # noqa: BLE001
                    logger.error("PLC worker exception: %s", exc)
                    logger.debug("%s", traceback.format_exc())
                    self.connected = False
                    self.mb.close()
                    self._update_status(connected=False)
                    time.sleep(0.5)
        finally:
            self.mb.close()
            logger.info("PLC protocol worker thread exiting")

    # ---- Command processing ----

    def _handle_commands(self) -> None:
        while True:
            try:
                cmd, data = self._command_queue.get_nowait()
            except queue.Empty:
                return
            try:
                if cmd == "connect":
                    ok = self.mb.connect()
                    self.connected = ok
                    self._update_status(connected=ok)
                    logger.info(
                        "PLC connect %s://%s:%s -> %s",
                        "modbus", self.cfg.host, self.cfg.port,
                        "OK" if ok else "FAIL",
                    )
                elif cmd == "disconnect":
                    self.mb.close()
                    self.connected = False
                    self.auto_running = False
                    self._update_status(connected=False, auto_running=False)
                    logger.info("PLC disconnected")
                elif cmd == "start_auto":
                    self.repeat_map = {
                        **{p.key: p.repeat for p in self.points},
                        **{str(k): max(1, int(v)) for k, v in data.get("repeats", {}).items()},
                    }
                    requested_route_indices = [
                        int(idx)
                        for idx in (data.get("route_indices") or [])
                        if 0 <= int(idx) < len(self.points)
                    ]
                    self.route_indices = (
                        requested_route_indices
                        if requested_route_indices
                        else list(range(len(self.points)))
                    )
                    self.start_route_pos = max(
                        0,
                        min(
                            int(data.get("start_route_pos", 0)),
                            len(self.route_indices) - 1,
                        ),
                    )
                    self.current_route_pos = self.start_route_pos
                    self.start_index = self.route_indices[self.start_route_pos]
                    self.current_index = self.start_index
                    self.pick_sent_count = 0
                    self.active_press_index = None
                    self.active_photo_key = None
                    self.completed_cycle = False
                    self.auto_running = True
                    for st in self.requests.values():
                        st.handled = False
                        st.last_key = ""
                    logger.info(
                        "auto-flow started from route point %d/%d (global point %d/%d)",
                        self.current_route_pos + 1,
                        len(self.route_indices),
                        self.current_index + 1,
                        len(self.points),
                    )
                    self._push_auto_status()
                elif cmd == "stop_auto":
                    self.auto_running = False
                    self.completed_cycle = False
                    logger.info("auto-flow paused")
                    self._push_auto_status()
                elif cmd == "reset_auto":
                    self.auto_running = False
                    self.current_route_pos = self.start_route_pos
                    self.current_index = self.start_index
                    self.pick_sent_count = 0
                    self.active_press_index = None
                    self.active_photo_key = None
                    self.completed_cycle = False
                    for st in self.requests.values():
                        st.handled = False
                        st.last_key = ""
                    logger.info("auto-flow reset to first point")
                    self._push_auto_status()
            except Exception as exc:  # noqa: BLE001
                logger.error("command %s failed: %s", cmd, exc)

    # ---- Polling ----

    def _write_heartbeat(self) -> None:
        self.heartbeat_value = 0 if self.heartbeat_value else 1
        self.mb.write_code(self.cfg.registers.pc_heartbeat, self.heartbeat_value)
        self._update_status(heartbeat=self.heartbeat_value)

    def _poll_once(self) -> None:
        r = self.cfg.registers

        # Read everything this cycle exposes: request flags, function codes,
        # and the three arms' real-time positions. RT must land in the
        # snapshot BEFORE _auto_handle_requests fires — on_photo_arrived
        # downstream consumers (e.g. PlcPoseSource) need to see nova5's
        # arrived position, not the last cycle's transit reading.
        nova2_flag = self.mb.read_code(r.plc_nova2_send_flag)
        nova2_func = self.mb.read_code(r.plc_nova2_func)
        nova5_flag = self.mb.read_code(r.plc_nova5_send_flag)
        nova5_func = self.mb.read_code(r.plc_nova5_func)
        epson_flag = self.mb.read_code(r.plc_epson_send_flag)
        epson_func = self.mb.read_code(r.plc_epson_func)
        pc_nova2_flag = self.mb.read_code(r.pc_nova2_send_flag)
        pc_nova5_flag = self.mb.read_code(r.pc_nova5_send_flag)
        pc_epson_flag = self.mb.read_code(r.pc_epson_send_flag)
        rt = {
            "nova2": self._read_axes([
                r.plc_rt_nova2_x, r.plc_rt_nova2_y, r.plc_rt_nova2_z,
                r.plc_rt_nova2_u, r.plc_rt_nova2_v, r.plc_rt_nova2_w,
            ]),
            "nova5": self._read_axes([
                r.plc_rt_nova5_x, r.plc_rt_nova5_y, r.plc_rt_nova5_z,
                r.plc_rt_nova5_u, r.plc_rt_nova5_v, r.plc_rt_nova5_w,
            ]),
            "epson_ls6": self._read_axes([
                r.plc_rt_epson_x, r.plc_rt_epson_y, r.plc_rt_epson_z, r.plc_rt_epson_u,
            ]),
        }
        epson_tool = int(round(self.mb.read_code(r.plc_rt_epson_tool)))
        if epson_tool != self.current_epson_tool:
            logger.info(
                "Epson current tool changed: %s -> %s (%s)",
                self.current_epson_tool,
                epson_tool,
                _describe_epson_tool(epson_tool),
            )
            self.current_epson_tool = epson_tool

        self._update_request("nova2", nova2_flag, nova2_func, pc_nova2_flag)
        self._update_request("nova5", nova5_flag, nova5_func, pc_nova5_flag)
        self._update_request("epson_ls6", epson_flag, epson_func, pc_epson_flag)

        self._clear_pc_flag_after_plc_clear("nova2", nova2_flag, pc_nova2_flag, r.pc_nova2_send_flag)
        self._clear_pc_flag_after_plc_clear("nova5", nova5_flag, pc_nova5_flag, r.pc_nova5_send_flag)
        self._clear_pc_flag_after_plc_clear("epson_ls6", epson_flag, pc_epson_flag, r.pc_epson_send_flag)

        # Publish RT first so consumers reading via snapshot see this cycle's
        # fresh values when hooks like on_photo_arrived fire below.
        self._update_status(
            connected=True,
            requests={
                "nova2": {"flag": nova2_flag, "func": nova2_func, "pc_flag": pc_nova2_flag,
                          "handled": self.requests["nova2"].handled},
                "nova5": {"flag": nova5_flag, "func": nova5_func, "pc_flag": pc_nova5_flag,
                          "handled": self.requests["nova5"].handled},
                "epson_ls6": {"flag": epson_flag, "func": epson_func, "pc_flag": pc_epson_flag,
                              "handled": self.requests["epson_ls6"].handled},
            },
            rt=rt,
            epson_tool=epson_tool,
        )

        if (not self.auto_running) and self.completed_cycle and self.cfg.auto_start:
            self._maybe_restart_completed_cycle()

        if self.auto_running:
            self._auto_handle_requests()

    def _maybe_restart_completed_cycle(self) -> None:
        """Re-arm auto flow after a completed cycle when PLC starts a new tray.

        Restart trigger depends on the tool currently mounted on Epson:

        - tool=2 (suction): a new round may start immediately from nova2's
          next coord request, without waiting for nova5 global-arrival func=2.
        - tool=1 (tweezer): require nova5 func=2 to mark the new tray's
          global photo-arrival before re-arming auto flow.

        In all cases the route restarts from ``self.start_index`` so a
        configured start_press_index applies to every round, not just the
        first round after process startup.
        """
        trigger_nova5_global = self._is_unhandled("nova5", 2)
        trigger_nova2_press = self._is_unhandled("nova2", 1)

        if self.current_epson_tool == 2:
            if not (trigger_nova2_press or trigger_nova5_global):
                return
            trigger = "nova2 func=1" if trigger_nova2_press else "nova5 func=2"
        else:
            if not trigger_nova5_global:
                return
            trigger = "nova5 func=2"

        if self.route_indices and self.start_index in self.route_indices:
            self.start_route_pos = self.route_indices.index(self.start_index)
        self.current_route_pos = self.start_route_pos
        if not (0 <= self.current_route_pos < len(self.route_indices)):
            return
        self.current_index = self.route_indices[self.current_route_pos]
        if not (0 <= self.current_index < len(self.points)):
            return
        self.pick_sent_count = 0
        self.active_press_index = None
        self.active_photo_key = None
        self.completed_cycle = False
        self.auto_running = True
        for st in self.requests.values():
            st.handled = False
        logger.info(
            "auto-flow restart triggered by %s with tool=%s (%s); reset to route point %d/%d (global point %d/%d)",
            trigger,
            self.current_epson_tool,
            _describe_epson_tool(self.current_epson_tool),
            self.current_route_pos + 1,
            len(self.route_indices),
            self.current_index + 1,
            len(self.points),
        )
        self._push_auto_status()

    def _read_axes(self, addrs: List[int]) -> List[float]:
        vals: List[float] = []
        for a in addrs:
            try:
                vals.append(self.mb.read_real32(a))
            except Exception:  # noqa: BLE001
                vals.append(float("nan"))
        return vals

    def _update_request(self, name: str, flag: float, func: float, pc_flag: float) -> None:
        st = self.requests[name]
        key = f"{int(round(flag))}:{int(round(func))}"
        st.flag = flag
        st.func = func
        st.pc_flag = pc_flag
        if int(round(flag)) == 1:
            if key != st.last_key:
                st.last_key = key
                st.handled = False
                logger.info(
                    "RX %s request func=%g", _ROBOT_LABEL.get(name, name), func
                )
        else:
            if st.last_key:
                st.last_key = ""
                st.handled = False

    def _clear_pc_flag_after_plc_clear(
        self, name: str, plc_flag: float, pc_flag: float, pc_flag_addr: int
    ) -> None:
        if int(round(plc_flag)) == 0 and int(round(pc_flag)) == 1:
            self.mb.write_code(pc_flag_addr, 0)
            self.requests[name].handled = False
            logger.info(
                "TX clear %s pc_send_flag=0 (PLC cleared first)",
                _ROBOT_LABEL.get(name, name),
            )

    # ---- Auto flow state machine (strict order) ----

    def _auto_handle_requests(self) -> None:
        if not (0 <= self.current_route_pos < len(self.route_indices)):
            self.auto_running = False
            self._push_auto_status()
            return
        self.current_index = self.route_indices[self.current_route_pos]
        p = self.points[self.current_index]
        cur_key = p.key
        cur_press = p.press_index
        r = self.cfg.registers

        # nova2 (press arm) requests current press coord.
        # On success, lock active_press_index and clear photo (nova5 must re-ack).
        if self._is_unhandled("nova2", 1):
            ok = self._send_coords(
                robot="nova2",
                axes=["x", "y", "z", "u", "v", "w"],
                values=p.nova2,
                pc_addrs=[r.pc_nova2_x, r.pc_nova2_y, r.pc_nova2_z,
                          r.pc_nova2_u, r.pc_nova2_v, r.pc_nova2_w],
                plc_echo_addrs=[r.plc_recv_nova2_x, r.plc_recv_nova2_y, r.plc_recv_nova2_z,
                                r.plc_recv_nova2_u, r.plc_recv_nova2_v, r.plc_recv_nova2_w],
                pc_func_addr=r.pc_nova2_func,
                pc_flag_addr=r.pc_nova2_send_flag,
                return_code=1,
                tag=f"nova2[{cur_press}]",
            )
            if ok:
                self.active_press_index = cur_press
                self.active_photo_key = None
                logger.info(
                    "order-guard: active press=nova2[%s], awaiting nova5 request", cur_press
                )
            self._push_auto_status()

        # nova5 (vision arm) requests current photo coord.
        # Refuse if press hasn't been confirmed for this press_index.
        if self._is_unhandled("nova5", 1):
            if self.active_press_index != cur_press:
                logger.warning(
                    "order-guard: press nova2[%s] not yet acked, refusing nova5[%s,%s]",
                    cur_press, p.press_index, p.photo_index,
                )
            else:
                ok = self._send_coords(
                    robot="nova5",
                    axes=["x", "y", "z", "u", "v", "w"],
                    values=p.nova5,
                    pc_addrs=[r.pc_nova5_x, r.pc_nova5_y, r.pc_nova5_z,
                              r.pc_nova5_u, r.pc_nova5_v, r.pc_nova5_w],
                    plc_echo_addrs=[r.plc_recv_nova5_x, r.plc_recv_nova5_y, r.plc_recv_nova5_z,
                                    r.plc_recv_nova5_u, r.plc_recv_nova5_v, r.plc_recv_nova5_w],
                    pc_func_addr=r.pc_nova5_func,
                    pc_flag_addr=r.pc_nova5_send_flag,
                    return_code=1,
                    tag=f"nova5[{p.press_index},{p.photo_index}]",
                )
                if ok:
                    # NOTE: do NOT mark this photo active here. nova5 has
                    # only just *received* the coord — it still needs to
                    # physically move to the new pose. Setting the active
                    # key now would let stale picks from the previous photo
                    # leak through to the next epson func=1 (strict-order
                    # checks active_photo_key == cur_key). Defer to func=11
                    # ack ("nova5 arrived"), where on_photo_arrived also
                    # clears the pick deque atomically.
                    logger.info(
                        "order-guard: nova5 coord sent for [%s,%s], "
                        "awaiting func=11 (arrived) before activating photo",
                        p.press_index, p.photo_index,
                    )
                    try:
                        self.on_nova5_coord_sent(p)
                    except Exception as exc:  # noqa: BLE001
                        logger.warning("on_nova5_coord_sent hook raised: %s", exc)
                self._push_auto_status()

        # nova5 reports "arrived at photo position".
        # func=2 (global) is unconditional; func=11 (local) is gated by photo key.
        if self._is_unhandled("nova5", 2):
            self._send_non_coord("nova5", r.pc_nova5_func, r.pc_nova5_send_flag, 2,
                                 "global photo-arrived")
        if self._is_unhandled("nova5", 11):
            # Accept when press matches AND (no photo currently active, i.e.
            # this is the first ack at the new photo, OR active_photo_key
            # already equals cur_key, i.e. PLC is re-requesting the ack
            # because it didn't see our flag clear yet — must be idempotent).
            ok_press = self.active_press_index == cur_press
            ok_photo = (
                self.active_photo_key is None
                or self.active_photo_key == cur_key
            )
            if not (ok_press and ok_photo):
                logger.warning(
                    "order-guard: local photo-arrived rejected, "
                    "active_press=%s expected_press=%s active_photo=%s expected=%s",
                    self.active_press_index, cur_press,
                    self.active_photo_key, cur_key,
                )
            else:
                # Atomic on the first ack only: flush stale picks via
                # on_photo_arrived, then mark photo active. on_photo_arrived
                # itself is idempotent (clear() on an empty deque is fine),
                # but we only need to fire it when photo first becomes active.
                first_arrival = self.active_photo_key != cur_key
                if first_arrival:
                    try:
                        self.on_photo_arrived(p)
                    except Exception as exc:  # noqa: BLE001
                        logger.warning("on_photo_arrived hook raised: %s", exc)
                    self.active_photo_key = cur_key
                    logger.info(
                        "order-guard: nova5[%s,%s] arrived, photo activated",
                        p.press_index, p.photo_index,
                    )
                self._send_non_coord("nova5", r.pc_nova5_func, r.pc_nova5_send_flag, 11,
                                     "local photo-arrived")

        # epson_ls6 (pick arm) requests pick coord.
        # Gated by both press AND photo being currently active.
        if self._is_unhandled("epson_ls6", 1):
            if self.active_press_index != cur_press or self.active_photo_key != cur_key:
                logger.warning(
                    "order-guard: photo nova5[%s,%s] not acked, refusing epson",
                    p.press_index, p.photo_index,
                )
            else:
                try:
                    self.on_epson_pick_request(p)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("on_epson_pick_request hook raised: %s", exc)
                epson_target = self._resolve_epson_target(p)
                if epson_target is not None:
                    ok = self._send_coords(
                        robot="epson_ls6",
                        axes=["x", "y", "z", "u"],
                        values=epson_target,
                        pc_addrs=[r.pc_epson_x, r.pc_epson_y, r.pc_epson_z, r.pc_epson_u],
                        plc_echo_addrs=[r.plc_recv_epson_x, r.plc_recv_epson_y,
                                        r.plc_recv_epson_z, r.plc_recv_epson_u],
                        pc_func_addr=r.pc_epson_func,
                        pc_flag_addr=r.pc_epson_send_flag,
                        return_code=1,
                        tag=f"epson_ls6[{p.press_index},{p.photo_index}]",
                    )
                    if ok:
                        self.pick_sent_count += 1
                        try:
                            self.on_epson_coord_sent(p, dict(epson_target))
                        except Exception as exc:  # noqa: BLE001
                            logger.warning("on_epson_coord_sent hook raised: %s", exc)
                    self._push_auto_status()
                elif self._is_batch_empty():
                    # Vision ran and confirmed this photo position has no hair.
                    # For an Epson coord request (func=1), return func=2 to tell
                    # PLC "no hair / no pick coord". Do not advance here; wait
                    # for PLC's next-task request (func=21), then _handle_next_task()
                    # will emit 21/22/23 according to strict-order flow.
                    logger.info(
                        "no picks at %s; sending epson func=2 (no hair)",
                        cur_key,
                    )
                    self._send_non_coord(
                        "epson_ls6", r.pc_epson_func, r.pc_epson_send_flag, 2,
                        f"no-hair at {cur_key}",
                    )
                    self._push_auto_status()
                else:
                    # Vision hasn't reported yet — stay silent. PLC's flag stays
                    # at 1, we'll see this request again next poll and try again.
                    logger.debug(
                        "epson func=1 at %s but no vision batch yet; waiting",
                        cur_key,
                    )

        # epson_ls6 in-motion protocol (V2.0):
        # - func=50: arrived at pick waiting position → PC returns 50 or 51+coord
        # - func=60: arrived at pick-down position     → PC returns 60 or 61+coord
        # - func=70: moved aside after clamping        → PC returns 70 or 71
        for stage_func in (50, 60, 70):
            if self._is_unhandled("epson_ls6", stage_func):
                if self.active_press_index != cur_press or self.active_photo_key != cur_key:
                    logger.warning(
                        "order-guard: photo nova5[%s,%s] not acked, refusing epson func=%s",
                        p.press_index, p.photo_index, stage_func,
                    )
                    continue
                try:
                    decision = self.on_epson_motion_stage(p, stage_func)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("on_epson_motion_stage hook raised for func=%s: %s", stage_func, exc)
                    decision = None
                if decision is None:
                    logger.debug(
                        "epson func=%s at %s pending high-level vision/tweezer decision",
                        stage_func, cur_key,
                    )
                    continue
                ok = self._send_epson_stage_decision(p, stage_func, decision)
                if ok:
                    self._push_auto_status()

        # epson_ls6 asks "what next?" → 20 (repeat), 21 (next photo), 22 (next press),
        # 23 (all done).
        if self._is_unhandled("epson_ls6", 21):
            try:
                self.on_epson_next_task_request(p)
            except Exception as exc:  # noqa: BLE001
                logger.warning("on_epson_next_task_request hook raised: %s", exc)
            self._handle_next_task()

        # epson_ls6 reports "out of safety range" (func=3 per protocol Excel
        # row 7 / addr 41069). The Epson controller refused our last pick
        # coord because it sits outside the configured safe envelope. The
        # right reaction is the same as "Epson finished" — advance to the
        # next photo position (or next press / done) — except we skip the
        # _has_more_picks continue-pick branch, since chasing leftover
        # picks in the same batch would only produce more out-of-range
        # rejects (they all came from the same frame). Mirroring the
        # func=21 dispatch keeps the strict-order flow consistent.
        if self._is_unhandled("epson_ls6", 3):
            logger.warning(
                "epson_ls6 reported out-of-safety-range at %s; advancing",
                cur_key,
            )
            self._handle_safety_reject()

    def _send_epson_stage_decision(
        self, point: PlcPoint, stage_func: int, decision: EpsonStageDecision
    ) -> bool:
        """Send a V2.0 Epson in-motion decision back to PLC.

        50/60/70/71 are function-only responses; 51/61 carry corrected Epson
        X/Y while keeping the current target's Z/U unchanged.
        """
        r = self.cfg.registers
        try:
            code = int(decision.get("code"))
        except Exception:  # noqa: BLE001
            logger.warning(
                "invalid Epson stage decision for func=%s at %s: %s",
                stage_func, point.key, decision,
            )
            return False
        desc = str(decision.get("desc") or f"epson stage func={stage_func}")
        coord_obj = decision.get("coord")

        if code in (51, 61):
            if not isinstance(coord_obj, dict):
                logger.warning(
                    "Epson correction code=%s requires coord, got %s", code, coord_obj
                )
                return False
            coord: EpsonCoord = {
                "x": float(coord_obj["x"]),
                "y": float(coord_obj["y"]),
                "z": float(coord_obj["z"]),
                "u": float(coord_obj["u"]),
            }
            return self._send_coords(
                robot="epson_ls6",
                axes=["x", "y", "z", "u"],
                values=coord,
                pc_addrs=[r.pc_epson_x, r.pc_epson_y, r.pc_epson_z, r.pc_epson_u],
                plc_echo_addrs=[r.plc_recv_epson_x, r.plc_recv_epson_y,
                                r.plc_recv_epson_z, r.plc_recv_epson_u],
                pc_func_addr=r.pc_epson_func,
                pc_flag_addr=r.pc_epson_send_flag,
                return_code=code,
                tag=f"epson_ls6_stage{stage_func}[{point.press_index},{point.photo_index}]",
            )

        if code in (50, 60, 70, 71):
            return self._send_non_coord(
                "epson_ls6", r.pc_epson_func, r.pc_epson_send_flag, code, desc
            )

        logger.warning(
            "unsupported Epson stage response code=%s for PLC func=%s at %s",
            code, stage_func, point.key,
        )
        return False

    def _handle_safety_reject(self) -> None:
        """Epson rejected the last coord as out-of-safety-range (func=3).

        Drop any remaining picks in the current batch — they share the
        same pose snapshot and would just re-trip the same envelope —
        and route through ``_advance_or_finish`` so the rest of the
        cycle's state machine (active_press_index, active_photo_key,
        ``on_nova5_moving`` hook, frame_loop PAUSE) stays consistent
        with the normal "Epson finished" transition.
        """
        p = self.points[self.current_index]
        if self.active_photo_key != p.key:
            logger.warning(
                "order-guard: safety-reject at %s but active=%s; ignoring",
                p.key, self.active_photo_key,
            )
            return
        self._advance_or_finish()

    def _resolve_epson_target(self, point: PlcPoint) -> Optional[EpsonCoord]:
        """Resolve Epson LS6 target from vision-derived picks.

        nova2 and nova5 still come from the fixed point table.
        Epson LS6 uses algorithm output for X/Y via ``epson_coord_provider``.
        The point-table fallback is only used to fill fixed Z/U.

        If the provider returns None, it means either vision has not reported
        yet or the current batch has no coordinate ready. In that case we must
        not fall back to the fixed epson_ls6 X/Y point, otherwise epson_ls6 would pick
        the test point instead of the detected hair.
        """
        fallback = point.epson_ls6_fallback or {}
        try:
            resolved = self.epson_coord_provider(point, dict(fallback))
        except Exception as exc:  # noqa: BLE001
            logger.warning("epson_coord_provider raised: %s", exc)
            return None

        if resolved is None:
            return None

        # Provider owns algorithm X/Y. Fill fixed Z/U from fallback when omitted.
        merged: EpsonCoord = dict(fallback)
        merged.update(resolved)
        for ax in ("x", "y", "z", "u"):
            if ax not in merged:
                logger.warning(
                    "epson vision target missing axis %s at %s; refusing coord",
                    ax, point.key,
                )
                return None
        return merged

    def _is_unhandled(self, robot: str, func_code: int) -> bool:
        st = self.requests[robot]
        return (
            int(round(st.flag)) == 1
            and int(round(st.func)) == int(func_code)
            and not st.handled
        )

    def _handle_next_task(self) -> None:
        """Triggered by PLC epson func=21 ("Epson finished a pick, what next?").

        Drain semantics: keep picking at the same photo position as long as
        the orchestrator's vision-pick buffer still has entries. Advance
        only when the buffer drains. With no vision integration (default
        has_pending_picks), this collapses back to one pick per photo
        position — matching the original test rig's repeat=1 behavior.
        """
        p = self.points[self.current_index]
        key = p.key
        r = self.cfg.registers

        if self.active_photo_key != key:
            logger.warning(
                "order-guard: photo %s not acked, refusing next-task. active=%s",
                key, self.active_photo_key,
            )
            return

        try:
            if not bool(self.next_task_ready()):
                logger.debug(
                    "next-task at %s deferred by orchestrator (post-pick confirmation pending)",
                    key,
                )
                return
        except Exception as exc:  # noqa: BLE001
            logger.warning("next_task_ready raised: %s", exc)

        if self._has_more_picks():
            self._send_non_coord(
                "epson_ls6", r.pc_epson_func, r.pc_epson_send_flag, 20,
                f"continue-pick at {key} (served={self.pick_sent_count})",
            )
            self._push_auto_status()
            return

        self._advance_or_finish()

    def _advance_or_finish(self) -> None:
        """Emit func=21 (next photo) / func=22 (next press) / func=23 (done)
        based on current_index and the relationship between the current and
        next point's press_index. Updates worker state to match.

        Called from two places:
        - PLC epson func=21 with no pending picks → drained, advance
        - PLC epson func=1  with empty vision batch → skip this photo

        Fires on_nova5_moving BEFORE emitting the advance code so consumers
        (e.g. the camera loop) can pause before nova5 starts physically
        moving. After 23 (all done) we also signal moving since nova5 will
        retract to home — gives the camera loop a clean shutdown signal.
        """
        p = self.points[self.current_index]
        r = self.cfg.registers

        if self.current_route_pos < len(self.route_indices) - 1:
            old_press = p.press_index
            next_route_pos = self.current_route_pos + 1
            next_p = self.points[self.route_indices[next_route_pos]]
            if next_p.press_index == old_press:
                self._notify_nova5_moving(p, 21)
                self._send_non_coord(
                    "epson_ls6", r.pc_epson_func, r.pc_epson_send_flag, 21,
                    f"photo[{old_press},{p.photo_index}] done, moving to "
                    f"photo[{next_p.press_index},{next_p.photo_index}]",
                )
                self.current_route_pos = next_route_pos
                self.current_index = self.route_indices[self.current_route_pos]
                self.pick_sent_count = 0
                self.active_photo_key = None
                logger.info(
                    "order-guard: same press nova2[%s], awaiting nova5[%s,%s]",
                    old_press, next_p.press_index, next_p.photo_index,
                )
            else:
                self._notify_nova5_moving(p, 22)
                self._send_non_coord(
                    "epson_ls6", r.pc_epson_func, r.pc_epson_send_flag, 22,
                    f"press nova2[{old_press}] done, moving to nova2[{next_p.press_index}]",
                )
                self.current_route_pos = next_route_pos
                self.current_index = self.route_indices[self.current_route_pos]
                self.pick_sent_count = 0
                self.active_press_index = None
                self.active_photo_key = None
                logger.info(
                    "order-guard: new press nova2[%s], awaiting nova2 coord request",
                    next_p.press_index,
                )
            self._push_auto_status()
            return

        # Last point finished.
        self._notify_nova5_moving(p, 23)
        self._send_non_coord(
            "epson_ls6", r.pc_epson_func, r.pc_epson_send_flag, 23,
            f"all {len(self.route_indices)} active route points done",
        )
        self.auto_running = False
        self.completed_cycle = True
        self.active_press_index = None
        self.active_photo_key = None
        self._push_auto_status()
        logger.info("auto-flow complete: returned func=23")

    def _notify_nova5_moving(self, point: Optional[PlcPoint], code: int) -> None:
        try:
            self.on_nova5_moving(point, code)
        except Exception as exc:  # noqa: BLE001
            logger.warning("on_nova5_moving hook raised: %s", exc)

    def _has_more_picks(self) -> bool:
        try:
            return bool(self.has_pending_picks())
        except Exception as exc:  # noqa: BLE001
            logger.warning("has_pending_picks raised: %s", exc)
            return False

    def _is_batch_empty(self) -> bool:
        try:
            return bool(self.batch_received_empty())
        except Exception as exc:  # noqa: BLE001
            logger.warning("batch_received_empty raised: %s", exc)
            return False

    def _send_non_coord(
        self, robot: str, pc_func_addr: int, pc_flag_addr: int, code: int, desc: str
    ) -> bool:
        if self.requests[robot].handled:
            return False
        # Coord registers are intentionally NOT touched here. The PLC
        # program reads X/Y/Z/U continuously (not gated on send_flag
        # rising edge), so writing zeros on every function-only handshake
        # was being flagged as "PC keeps sending zero". The right scope
        # for any clearing is the moment immediately after a successful
        # coord consumption — handled separately if needed.
        self.mb.write_code(pc_func_addr, code)
        self.mb.write_code(pc_flag_addr, 1)
        self.requests[robot].handled = True
        logger.info(
            "TX %s code=%d pc_send_flag=1 (%s)",
            _ROBOT_LABEL.get(robot, robot), code, desc,
        )
        return True

    def _send_coords(
        self,
        robot: str,
        axes: List[str],
        values: Dict[str, float],
        pc_addrs: List[int],
        plc_echo_addrs: List[int],
        pc_func_addr: int,
        pc_flag_addr: int,
        return_code: int,
        tag: str,
    ) -> bool:
        if self.requests[robot].handled:
            return False
        target_vals = [float(values[a]) for a in axes]
        for axis, addr, val in zip(axes, pc_addrs, target_vals):
            self.mb.write_real32(addr, val)
            logger.info(
                "TX %s %s.%s=%.4f -> %d",
                _ROBOT_LABEL.get(robot, robot), tag, axis.upper(), val, addr,
            )
        self.mb.write_code(pc_func_addr, return_code)
        logger.info(
            "TX %s func=%d -> %d", _ROBOT_LABEL.get(robot, robot), return_code, pc_func_addr
        )
        ok, echo_vals = self._wait_echo_match(plc_echo_addrs, target_vals)
        if ok:
            self.mb.write_code(pc_flag_addr, 1)
            self.requests[robot].handled = True
            # Per-axis target/echo dump so a Δ between what we asked the
            # PLC to receive and what it actually has in the echo regs
            # is auditable for any pick — relevant when an Epson lands
            # somewhere we did NOT command and we need to rule out the
            # PLC bridge as the source.
            echo_str = ", ".join(
                f"{a.upper()}: tgt={t:.4f} echo={e:.4f} Δ={(e-t):+.4f}"
                for a, t, e in zip(axes, target_vals, echo_vals)
            )
            logger.info(
                "%s echo matched, pc_send_flag=1 -> %d  [%s]",
                tag, pc_flag_addr, echo_str,
            )
            return True
        logger.warning(
            "%s echo mismatch (kept pc_send_flag=0). target=%s echo=%s",
            tag, target_vals, echo_vals,
        )
        return False

    def _wait_echo_match(
        self, echo_addrs: List[int], target_vals: List[float]
    ) -> Tuple[bool, List[float]]:
        last_vals: List[float] = []
        deadline = time.time() + max(0.1, self.cfg.echo_wait_ms / 1000.0)
        while time.time() < deadline:
            last_vals = [self.mb.read_real32(a) for a in echo_addrs]
            if self._values_match(last_vals, target_vals):
                return True, last_vals
            time.sleep(0.08)
        return False, last_vals

    def _values_match(self, a: List[float], b: List[float]) -> bool:
        if len(a) != len(b):
            return False
        eps = float(self.cfg.compare_epsilon)
        for x, y in zip(a, b):
            if math.isnan(x) or abs(float(x) - float(y)) > eps:
                return False
        return True

    # ---- Status snapshot ----

    def _push_auto_status(self) -> None:
        if 0 <= self.current_route_pos < len(self.route_indices):
            self.current_index = self.route_indices[self.current_route_pos]
            p = self.points[self.current_index]
            self._update_status(
                auto_running=self.auto_running,
                current_index=self.current_index,
                current_route_pos=self.current_route_pos,
                route_point_count=len(self.route_indices),
                press_index=p.press_index,
                photo_index=p.photo_index,
                pick_sent_count=self.pick_sent_count,
                repeat_target=self.repeat_map.get(p.key, p.repeat),
            )
        else:
            self._update_status(
                auto_running=self.auto_running,
                current_index=self.current_index,
                current_route_pos=self.current_route_pos,
                route_point_count=len(self.route_indices),
            )

    def _update_status(self, **kwargs) -> None:
        with self._status_lock:
            self._status.update(kwargs)
