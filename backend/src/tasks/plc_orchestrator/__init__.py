"""PLC 3-arm orchestrator (Modbus TCP).

Drives nova2 (press) + nova5 (vision) + Epson LS6 (pick) by responding to
PLC poll requests. Epson coords are vision-driven (TASK:WORLD_PICKS) with
a points-table fallback.
"""

from .task import PlcOrchestratorTask

__all__ = ["PlcOrchestratorTask"]
