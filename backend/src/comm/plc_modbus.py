"""ModbusTCP helper for the 3-arm PLC orchestrator.

Pure protocol layer — no GUI, no threading. Wire format matches the
validated standalone test GUI ``plc_modbus_auto_test_gui_v2_strict.py``
(kept outside the repo) so the PLC program needs no changes:

- Modbus addresses in the config are PLC-style 4xxxx, converted to
  0-based offsets by subtracting ``base_addr``.
- Coords travel as 32-bit floats split across two 16-bit registers,
  word order configurable (PLC ships LOW_WORD_FIRST in our setup).
- "Codes" (flags + function codes) live in a single 16-bit register by
  default; one PLC variant stores them as floats, hence ``code_mode``.
"""

from __future__ import annotations

import struct
import threading
from typing import List, Optional

try:
    from pymodbus.client import ModbusTcpClient  # pymodbus >= 3.x
except Exception:  # pragma: no cover
    from pymodbus.client.sync import ModbusTcpClient  # pymodbus 2.x

from src.config import PlcOrchestratorConfig


class ModbusHelper:
    """Thread-safe pymodbus wrapper with float<->register encoding baked in."""

    def __init__(self, cfg: PlcOrchestratorConfig):
        self.cfg = cfg
        self.client: Optional[ModbusTcpClient] = None
        self.lock = threading.RLock()

    def set_config(self, cfg: PlcOrchestratorConfig) -> None:
        self.cfg = cfg

    def connect(self) -> bool:
        with self.lock:
            self.close()
            self.client = ModbusTcpClient(
                host=self.cfg.host,
                port=int(self.cfg.port),
                timeout=float(self.cfg.connect_timeout_s),
            )
            return bool(self.client.connect())

    def close(self) -> None:
        with self.lock:
            if self.client is not None:
                try:
                    self.client.close()
                except Exception:
                    pass
            self.client = None

    # ---- Internal IO ----

    def _offset(self, protocol_addr: int) -> int:
        return int(protocol_addr) - int(self.cfg.registers.base_addr)

    def _unit_kwargs_candidates(self) -> List[dict]:
        """Return Modbus unit/slave keyword candidates for different pymodbus versions.

        Different pymodbus releases changed this parameter name:
        - older versions: unit=...
        - common 3.x versions: slave=...
        - newer versions: device_id=...

        The device IPC currently uses whatever version uv resolved on that machine,
        so we try the compatible forms in order instead of hard-coding one name.
        """
        unit_id = int(self.cfg.unit_id)
        return [
            {"device_id": unit_id},
            {"slave": unit_id},
            {"unit": unit_id},
            {},
        ]

    def _call_modbus(self, func_name: str, **base_kwargs):
        """Call a pymodbus method with a version-compatible unit-id keyword."""
        if self.client is None:
            raise RuntimeError("Modbus not connected")
        func = getattr(self.client, func_name)
        last_type_error: TypeError | None = None
        for unit_kwargs in self._unit_kwargs_candidates():
            try:
                return func(**base_kwargs, **unit_kwargs)
            except TypeError as exc:
                # Signature mismatch for this pymodbus version. Try the next
                # spelling instead of killing the PLC heartbeat thread.
                last_type_error = exc
                continue
        if last_type_error is not None:
            raise last_type_error
        raise RuntimeError(f"modbus call failed: {func_name}")

    def _read_holding_registers(self, protocol_addr: int, count: int) -> List[int]:
        offset = self._offset(protocol_addr)
        rr = self._call_modbus(
            "read_holding_registers", address=offset, count=count
        )
        if rr is None or getattr(rr, "isError", lambda: True)():
            raise RuntimeError(
                f"modbus read failed addr={protocol_addr} count={count} resp={rr}"
            )
        return list(rr.registers)

    def _write_registers(self, protocol_addr: int, values: List[int]) -> None:
        offset = self._offset(protocol_addr)
        wr = self._call_modbus(
            "write_registers", address=offset, values=values
        )
        if wr is None or getattr(wr, "isError", lambda: True)():
            raise RuntimeError(
                f"modbus write failed addr={protocol_addr} values={values} resp={wr}"
            )

    def _write_register(self, protocol_addr: int, value: int) -> None:
        offset = self._offset(protocol_addr)
        wr = self._call_modbus(
            "write_register", address=offset, value=value
        )
        if wr is None or getattr(wr, "isError", lambda: True)():
            raise RuntimeError(
                f"modbus write failed addr={protocol_addr} value={value} resp={wr}"
            )

    # ---- Float<->word encoding ----

    @staticmethod
    def _float_to_words(value: float, word_order: str) -> List[int]:
        packed = struct.pack(">f", float(value))
        high, low = struct.unpack(">HH", packed)
        return [low, high] if word_order == "LOW_WORD_FIRST" else [high, low]

    @staticmethod
    def _words_to_float(words: List[int], word_order: str) -> float:
        if len(words) != 2:
            return float("nan")
        if word_order == "LOW_WORD_FIRST":
            low, high = words[0], words[1]
        else:
            high, low = words[0], words[1]
        packed = struct.pack(">HH", high & 0xFFFF, low & 0xFFFF)
        return float(struct.unpack(">f", packed)[0])

    # ---- Public typed accessors ----

    def read_real32(self, protocol_addr: int) -> float:
        with self.lock:
            return self._words_to_float(
                self._read_holding_registers(protocol_addr, 2), self.cfg.word_order
            )

    def write_real32(self, protocol_addr: int, value: float) -> None:
        with self.lock:
            self._write_registers(
                protocol_addr, self._float_to_words(value, self.cfg.word_order)
            )

    def read_code(self, protocol_addr: int) -> float:
        with self.lock:
            if self.cfg.code_mode == "REAL32":
                return self.read_real32(protocol_addr)
            regs = self._read_holding_registers(protocol_addr, 1)
            return float(regs[0])

    def write_code(self, protocol_addr: int, value: float) -> None:
        with self.lock:
            if self.cfg.code_mode == "REAL32":
                self.write_real32(protocol_addr, value)
            else:
                self._write_register(
                    protocol_addr, int(round(float(value))) & 0xFFFF
                )
