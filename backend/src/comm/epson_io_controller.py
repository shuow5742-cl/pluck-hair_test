"""Epson LS6 manual-control + device-IO controller (pluck-hair_test, NEW).

Drives the right-hand console: jog / move the Epson SCARA and toggle the
cylinder & utility IO. The *communication map* is intentionally external
(``config/epson_io.yaml``) and currently a placeholder — see that file's header.

Two interchangeable backends, chosen by ``epson_io.backend``:

* ``mock``   — pure software simulation. Jog nudges an in-memory pose, moves
               snap to the target, IO writes flip an in-memory state, and
               feedback follows the commanded state. Lets the entire UI work
               with no hardware attached (the default, so the script "just
               runs").
* ``modbus`` — talks to the real controller using the addresses in the YAML via
               pymodbus. Untested against live hardware until the map is filled
               in, so every transaction is defensive and surfaces errors instead
               of crashing the console.

The controller exposes JSON-serialisable dicts so the FastAPI routes can return
them directly.
"""

from __future__ import annotations

import logging
import struct
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_AXES_DEFAULT = ["X", "Y", "Z", "U"]


# ============================================================================
# Parsed config model
# ============================================================================
@dataclass
class TaughtPoint:
    name: str
    coords: Dict[str, float]


@dataclass
class IoDevice:
    id: str
    name: str
    plc_tag: str
    open_label: str
    close_label: str
    kind: str            # "cylinder" | "other"
    write: Dict[str, Any]
    feedback: Dict[str, Any]


class EpsonIoController:
    def __init__(self, config: Dict[str, Any]) -> None:
        root = (config or {}).get("epson_io", config or {})
        self._lock = threading.RLock()
        self.backend_kind: str = str(root.get("backend", "mock")).lower()

        epson = root.get("epson", {}) or {}
        self.axes: List[str] = list(epson.get("axes") or _AXES_DEFAULT)
        self.units: Dict[str, str] = dict(epson.get("units") or {})
        self.limits: Dict[str, List[float]] = {
            k: list(v) for k, v in (epson.get("limits") or {}).items()
        }
        self._pose_read = epson.get("pose_read", {}) or {}
        self._target_write = epson.get("target_write", {}) or {}
        self._command = epson.get("command", {}) or {}
        self._jog = epson.get("jog", {}) or {}
        self.points: List[TaughtPoint] = [
            TaughtPoint(
                name=str(p.get("name", f"P{i}")),
                coords={a: float(p.get(a, 0.0)) for a in self.axes},
            )
            for i, p in enumerate(epson.get("points") or [])
        ]

        self.devices: List[IoDevice] = []
        io = root.get("io", {}) or {}
        for d in io.get("cylinders", []) or []:
            self.devices.append(self._parse_device(d, "cylinder"))
        for d in io.get("others", []) or []:
            self.devices.append(self._parse_device(d, "other"))
        self._device_index = {d.id: d for d in self.devices}

        self.system_buttons: Dict[str, Any] = (root.get("system", {}) or {}).get("buttons", {}) or {}
        self._modbus_cfg = root.get("modbus", {}) or {}

        # Backend selection.
        self._backend: _Backend
        if self.backend_kind == "modbus":
            try:
                self._backend = _ModbusBackend(self, self._modbus_cfg)
                logger.info("EpsonIoController using MODBUS backend (%s:%s)",
                            self._modbus_cfg.get("host"), self._modbus_cfg.get("port"))
            except Exception as exc:  # noqa: BLE001
                logger.warning("Modbus backend init failed (%s); falling back to mock", exc)
                self.backend_kind = "mock"
                self._backend = _MockBackend(self)
        else:
            self._backend = _MockBackend(self)
            logger.info("EpsonIoController using MOCK backend (no hardware)")

    # -- parsing helpers ------------------------------------------------------
    def _parse_device(self, d: Dict[str, Any], kind: str) -> IoDevice:
        return IoDevice(
            id=str(d["id"]),
            name=str(d.get("name", d["id"])),
            plc_tag=str(d.get("plc_tag", "")),
            open_label=str(d.get("open_label", "打开")),
            close_label=str(d.get("close_label", "关闭")),
            kind=kind,
            write=d.get("write", {}) or {},
            feedback=d.get("feedback", {}) or {},
        )

    def clamp(self, axis: str, value: float) -> float:
        lim = self.limits.get(axis)
        if not lim or len(lim) != 2:
            return value
        return max(lim[0], min(lim[1], value))

    # ========================================================================
    # Public API (thread-safe)
    # ========================================================================
    def get_pose(self) -> Dict[str, float]:
        with self._lock:
            return self._backend.get_pose()

    def jog(self, axis: str, direction: str, step_mode: str = "short") -> Dict[str, Any]:
        axis = axis.upper()
        if axis not in self.axes:
            return {"ok": False, "error": f"unknown axis {axis}"}
        if direction not in ("+", "-"):
            return {"ok": False, "error": "direction must be + or -"}
        with self._lock:
            return self._backend.jog(axis, direction, step_mode)

    def move(self, target: Dict[str, float], command: str = "Move") -> Dict[str, Any]:
        command = "Go" if str(command).lower() == "go" else "Move"
        clean = {}
        for a in self.axes:
            if a in target and target[a] is not None:
                clean[a] = self.clamp(a, float(target[a]))
        if not clean:
            return {"ok": False, "error": "no target coordinates provided"}
        with self._lock:
            return self._backend.move(clean, command)

    def goto_point(self, name: str, command: str = "Move") -> Dict[str, Any]:
        pt = next((p for p in self.points if p.name == name), None)
        if pt is None:
            return {"ok": False, "error": f"unknown point {name}"}
        return self.move(dict(pt.coords), command)

    def list_points(self) -> List[Dict[str, Any]]:
        return [{"name": p.name, **p.coords} for p in self.points]

    def get_io_states(self) -> List[Dict[str, Any]]:
        with self._lock:
            return self._backend.get_io_states()

    def set_io(self, device_id: str, action: str) -> Dict[str, Any]:
        dev = self._device_index.get(device_id)
        if dev is None:
            return {"ok": False, "error": f"unknown device {device_id}"}
        with self._lock:
            return self._backend.set_io(dev, action.lower())

    def system_command(self, command: str) -> Dict[str, Any]:
        command = command.lower()
        if command not in self.system_buttons:
            return {"ok": False, "error": f"unknown system command {command}"}
        with self._lock:
            return self._backend.system_command(command)

    def get_system_state(self) -> Dict[str, Any]:
        with self._lock:
            return self._backend.get_system_state()

    def describe(self) -> Dict[str, Any]:
        """Static metadata for the frontend to render labels/limits."""
        return {
            "backend": self.backend_kind,
            "axes": self.axes,
            "units": self.units,
            "limits": self.limits,
            "points": self.list_points(),
            "commands": ["Move", "Go"],
            "step_modes": list((self._jog.get("step_modes") or {"short": 0}).keys()),
            "devices": [
                {
                    "id": d.id, "name": d.name, "plc_tag": d.plc_tag,
                    "kind": d.kind, "open_label": d.open_label,
                    "close_label": d.close_label,
                }
                for d in self.devices
            ],
            "system_buttons": [
                {"id": k, "name": (v or {}).get("name", k)}
                for k, v in self.system_buttons.items()
            ],
        }

    def close(self) -> None:
        with self._lock:
            self._backend.close()


# ============================================================================
# Backends
# ============================================================================
class _Backend:
    def get_pose(self) -> Dict[str, float]: raise NotImplementedError
    def jog(self, axis: str, direction: str, step_mode: str) -> Dict[str, Any]: raise NotImplementedError
    def move(self, target: Dict[str, float], command: str) -> Dict[str, Any]: raise NotImplementedError
    def get_io_states(self) -> List[Dict[str, Any]]: raise NotImplementedError
    def set_io(self, dev: IoDevice, action: str) -> Dict[str, Any]: raise NotImplementedError
    def system_command(self, command: str) -> Dict[str, Any]: raise NotImplementedError
    def get_system_state(self) -> Dict[str, Any]: raise NotImplementedError
    def close(self) -> None: ...


def _feedback_text(dev: IoDevice, state: str) -> str:
    """Render the 反馈状态 string a proximity switch would report."""
    if dev.kind == "cylinder":
        if state == "extended":
            return f"{dev.open_label}到位"
        if state == "retracted":
            return f"{dev.close_label}到位"
        return "未到位"
    # other (start/stop) devices
    if state == "running":
        return "运行中"
    if state == "stopped":
        return "已停止"
    return "未知"


def _io_state_dict(dev: IoDevice, state: str) -> Dict[str, Any]:
    in_position = state in ("extended", "retracted", "running", "stopped")
    return {
        "id": dev.id,
        "name": dev.name,
        "plc_tag": dev.plc_tag,
        "kind": dev.kind,
        "open_label": dev.open_label,
        "close_label": dev.close_label,
        "state": state,
        "feedback": _feedback_text(dev, state),
        "in_position": in_position,
        "is_open": state in ("extended", "running"),
    }


class _MockBackend(_Backend):
    """Software simulation so the console is fully usable with no hardware."""

    def __init__(self, ctl: EpsonIoController) -> None:
        self._ctl = ctl
        # Seed pose from the first taught point ("Home") if available.
        if ctl.points:
            self._pose = dict(ctl.points[0].coords)
        else:
            self._pose = {a: 0.0 for a in ctl.axes}
        self._io_state = {d.id: "unknown" for d in ctl.devices}
        self._system_mode = "idle"
        self._last_system = ""

    def get_pose(self) -> Dict[str, float]:
        return {a: round(float(self._pose.get(a, 0.0)), 3) for a in self._ctl.axes}

    def jog(self, axis: str, direction: str, step_mode: str) -> Dict[str, Any]:
        jog = self._ctl._jog
        is_angle = self._ctl.units.get(axis, "mm") == "deg"
        steps = (jog.get("step_deg") if is_angle else jog.get("step_mm")) or {}
        step = float(steps.get(step_mode, steps.get("short", 1.0)))
        sign = 1.0 if direction == "+" else -1.0
        self._pose[axis] = self._ctl.clamp(axis, float(self._pose.get(axis, 0.0)) + sign * step)
        return {"ok": True, "pose": self.get_pose(), "step": step}

    def move(self, target: Dict[str, float], command: str) -> Dict[str, Any]:
        # Mock snaps instantly to the (clamped) target.
        for a, v in target.items():
            self._pose[a] = v
        return {"ok": True, "command": command, "pose": self.get_pose()}

    def get_io_states(self) -> List[Dict[str, Any]]:
        return [_io_state_dict(d, self._io_state.get(d.id, "unknown")) for d in self._ctl.devices]

    def set_io(self, dev: IoDevice, action: str) -> Dict[str, Any]:
        mapping = {
            "extend": "extended", "open": "extended", "unlock": "extended",
            "retract": "retracted", "close": "retracted", "lock": "retracted",
            "start": "running", "stop": "stopped",
        }
        state = mapping.get(action)
        if state is None:
            return {"ok": False, "error": f"unknown action {action}"}
        self._io_state[dev.id] = state
        return {"ok": True, "device": _io_state_dict(dev, state)}

    def system_command(self, command: str) -> Dict[str, Any]:
        self._last_system = command
        # Reflect a coarse mode for the lamp display.
        mode_map = {
            "auto": "auto", "init": "init", "start": "running",
            "stop": "stopped", "pause": "paused", "reset": "idle",
            "manual_pick_ok": self._system_mode,
        }
        self._system_mode = mode_map.get(command, self._system_mode)
        return {"ok": True, "command": command, "mode": self._system_mode}

    def get_system_state(self) -> Dict[str, Any]:
        return {"mode": self._system_mode, "last_command": self._last_system}

    def close(self) -> None:  # nothing to release
        ...


class _ModbusBackend(_Backend):
    """Real Modbus-TCP backend. Defensive: a transport error never crashes the
    console — it returns ``{"ok": False, "error": ...}`` and the UI shows it."""

    def __init__(self, ctl: EpsonIoController, cfg: Dict[str, Any]) -> None:
        from pymodbus.client import ModbusTcpClient  # lazy import

        self._ctl = ctl
        self._host = cfg.get("host", "127.0.0.1")
        self._port = int(cfg.get("port", 502))
        self._unit = int(cfg.get("unit_id", 1))
        self._word_order = str(cfg.get("word_order", "big")).lower()
        self._client = ModbusTcpClient(
            self._host, port=self._port, timeout=float(cfg.get("timeout_s", 1.0))
        )
        self._client.connect()

    # ---- register helpers ---------------------------------------------------
    def _read_float(self, addr: int) -> Optional[float]:
        try:
            rr = self._client.read_holding_registers(addr, count=2, slave=self._unit)
            if rr.isError():
                return None
            regs = rr.registers
            if self._word_order == "little":
                regs = list(reversed(regs))
            return struct.unpack(">f", struct.pack(">HH", regs[0], regs[1]))[0]
        except Exception as exc:  # noqa: BLE001
            logger.debug("modbus read_float %s failed: %s", addr, exc)
            return None

    def _write_float(self, addr: int, value: float) -> bool:
        try:
            hi, lo = struct.unpack(">HH", struct.pack(">f", float(value)))
            regs = [hi, lo] if self._word_order != "little" else [lo, hi]
            wr = self._client.write_registers(addr, regs, slave=self._unit)
            return not wr.isError()
        except Exception as exc:  # noqa: BLE001
            logger.debug("modbus write_float %s failed: %s", addr, exc)
            return False

    def _write_reg(self, addr: int, value: int) -> bool:
        try:
            wr = self._client.write_register(addr, int(value), slave=self._unit)
            return not wr.isError()
        except Exception as exc:  # noqa: BLE001
            logger.debug("modbus write_register %s failed: %s", addr, exc)
            return False

    def _write_coil(self, addr: int, value: bool) -> bool:
        try:
            wr = self._client.write_coil(addr, bool(value), slave=self._unit)
            return not wr.isError()
        except Exception as exc:  # noqa: BLE001
            logger.debug("modbus write_coil %s failed: %s", addr, exc)
            return False

    def _read_coil(self, addr: int) -> Optional[bool]:
        try:
            rr = self._client.read_discrete_inputs(addr, count=1, slave=self._unit)
            if rr.isError():
                rr = self._client.read_coils(addr, count=1, slave=self._unit)
            if rr.isError():
                return None
            return bool(rr.bits[0])
        except Exception as exc:  # noqa: BLE001
            logger.debug("modbus read_coil %s failed: %s", addr, exc)
            return None

    # ---- backend API --------------------------------------------------------
    def get_pose(self) -> Dict[str, float]:
        pose = {}
        for a in self._ctl.axes:
            spec = self._ctl._pose_read.get(a, {})
            val = self._read_float(int(spec["addr"])) if "addr" in spec else None
            pose[a] = round(val, 3) if val is not None else 0.0
        return pose

    def jog(self, axis: str, direction: str, step_mode: str) -> Dict[str, Any]:
        jog = self._ctl._jog
        key = f"{axis}{direction}"
        cmd_val = (jog.get("values") or {}).get(key)
        if cmd_val is None or "command_addr" not in jog:
            return {"ok": False, "error": f"no jog mapping for {key}"}
        modes = jog.get("step_modes") or {}
        if "step_mode_addr" in jog and step_mode in modes:
            self._write_reg(int(jog["step_mode_addr"]), int(modes[step_mode]))
        ok = self._write_reg(int(jog["command_addr"]), int(cmd_val))
        return {"ok": ok, "pose": self.get_pose()}

    def move(self, target: Dict[str, float], command: str) -> Dict[str, Any]:
        for a, v in target.items():
            spec = self._ctl._target_write.get(a, {})
            if "addr" in spec:
                self._write_float(int(spec["addr"]), v)
        cmd = self._ctl._command
        if "mode_addr" in cmd:
            mode_val = (cmd.get("mode_values") or {}).get(command, 0)
            self._write_reg(int(cmd["mode_addr"]), int(mode_val))
        ok = True
        if "trigger_addr" in cmd:
            ok = self._write_reg(int(cmd["trigger_addr"]), int(cmd.get("trigger_value", 1)))
        return {"ok": ok, "command": command, "pose": self.get_pose()}

    def get_io_states(self) -> List[Dict[str, Any]]:
        out = []
        for dev in self._ctl.devices:
            state = "unknown"
            fb = dev.feedback or {}
            if dev.kind == "cylinder":
                ext = self._read_coil(int(fb["extended"]["addr"])) if "extended" in fb else None
                ret = self._read_coil(int(fb["retracted"]["addr"])) if "retracted" in fb else None
                if ext:
                    state = "extended"
                elif ret:
                    state = "retracted"
            else:
                run = self._read_coil(int(fb["running"]["addr"])) if "running" in fb else None
                if run is True:
                    state = "running"
                elif run is False:
                    state = "stopped"
            out.append(_io_state_dict(dev, state))
        return out

    def set_io(self, dev: IoDevice, action: str) -> Dict[str, Any]:
        # Normalise UI action verbs to the YAML write keys present on this device.
        alias = {
            "open": "extend", "close": "retract", "unlock": "extend", "lock": "retract",
        }
        key = action if action in dev.write else alias.get(action, action)
        spec = dev.write.get(key)
        if spec is None:
            return {"ok": False, "error": f"no write mapping for {action} on {dev.id}"}
        ok = self._write_coil(int(spec["addr"]), bool(spec.get("value", 1)))
        return {"ok": ok}

    def system_command(self, command: str) -> Dict[str, Any]:
        spec = self._ctl.system_buttons.get(command, {})
        if "addr" not in spec:
            return {"ok": False, "error": f"no addr for {command}"}
        ok = self._write_coil(int(spec["addr"]), bool(spec.get("value", 1)))
        return {"ok": ok, "command": command}

    def get_system_state(self) -> Dict[str, Any]:
        lamps = {}
        for name, spec in self._ctl.system_buttons.items():
            if "state_addr" in spec:
                lamps[name] = self._read_coil(int(spec["state_addr"]))
        return {"lamps": lamps}

    def close(self) -> None:
        try:
            self._client.close()
        except Exception:  # noqa: BLE001
            ...


# ============================================================================
# Loader
# ============================================================================
def load_epson_io_controller(path: str) -> EpsonIoController:
    import yaml

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return EpsonIoController(data)
