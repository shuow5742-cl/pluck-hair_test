"""Robot flange pose sources for downstream coordinate transforms.

The pixel→world coordinate transform task needs the live robot flange pose
at the moment a frame was captured. The mechanism for getting that pose
differs across deployment modes (real PLC push, simulated arm, recorded
trace replay), so the consumers reference pose sources **by name** and look
them up via the registry below.

Main.py registers the appropriate source at startup; tasks that need a pose
pull it from the registry without knowing the concrete class.

Eye-in-hand 3-DOF assumption: flange has XYZ translation only, no rotation.
This is a deliberate simplification matching the current mechanical setup —
the source is expected to deliver only (x, y, z) and tasks must not depend
on orientation channels.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Callable, Optional, Protocol

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class FlangePose:
    """Snapshot of robot flange pose at a single instant (mm)."""

    x: float
    y: float
    z: float


class PoseSource(Protocol):
    """A source that can return the most recent known flange pose, or None.

    Implementations may block briefly to fetch a fresh sample, but should
    NOT do long-running I/O — this is called once per processed frame.
    Returning None indicates "no pose available" and the caller should
    degrade gracefully (e.g. leave world_xy unset).
    """

    def read(self) -> Optional[FlangePose]:
        ...


class MockPoseSource:
    """Constant-pose source for offline / static-image testing.

    Equivalent to the legacy ``dx=dy=0`` extrinsic config — pretends the
    flange sits at the origin so the pixel→world transform collapses to
    image-space mm with no robot offset.
    """

    def __init__(self, x: float = 0.0, y: float = 0.0, z: float = 0.0) -> None:
        self._pose = FlangePose(x=float(x), y=float(y), z=float(z))

    def read(self) -> Optional[FlangePose]:
        return self._pose


class NullPoseSource:
    """Always-None source for exercising the degraded path."""

    def read(self) -> Optional[FlangePose]:
        return None


class PlcPoseSource:
    """Live nova5 flange pose, sampled from a PLC telemetry snapshot.

    The pick arm (Epson LS6) and the press arm (Nova2) don't carry the
    camera. The camera is eye-in-hand on **Nova5** — that's the arm
    whose flange position appears in the pixel→world transform.

    The actual modbus IO lives in the orchestrator's ProtocolWorker; this
    class is just a 1-call adapter: it takes a callable that returns the
    last-polled (x, y) (or None if not connected / NaN) and wraps it as a
    PoseSource. We default Z to 0 because the transform is 2D (telecentric
    lens) — Z exists in FlangePose for symmetry with future 3D variants.
    """

    def __init__(self, reader: Callable[[], Optional[tuple[float, float]]]) -> None:
        self._reader = reader

    def read(self) -> Optional[FlangePose]:
        try:
            xy = self._reader()
        except Exception as exc:  # noqa: BLE001
            logger.debug("PlcPoseSource reader raised: %s", exc)
            return None
        if xy is None:
            logger.debug("PlcPoseSource reader returned None (PLC not connected?)")
            return None
        x, y = float(xy[0]), float(xy[1])
        if math.isnan(x) or math.isnan(y):
            logger.debug("PlcPoseSource reader returned NaN: (%s, %s)", xy[0], xy[1])
            return None
        # DEBUG only — turn on with --log-level DEBUG when reproducing the
        # field anomaly. Stays out of routine logs because it fires on every
        # processed frame.
        logger.debug("PlcPoseSource raw nova5 RT XY=(%.4f, %.4f)", x, y)
        return FlangePose(x=x, y=y, z=0.0)


_REGISTRY: dict[str, PoseSource] = {}


def register_pose_source(name: str, source: PoseSource) -> None:
    """Register a pose source under ``name`` for later lookup.

    Re-registration replaces the previous entry — useful in tests that need
    to swap a mock in and out.
    """
    _REGISTRY[name] = source


def get_pose_source(name: str) -> PoseSource:
    """Look up a previously registered pose source by name."""
    try:
        return _REGISTRY[name]
    except KeyError as e:
        available = ", ".join(sorted(_REGISTRY)) or "(none)"
        raise KeyError(
            f"PoseSource '{name}' not registered. Available: {available}"
        ) from e


def clear_pose_sources() -> None:
    """Forget all registered sources (test helper)."""
    _REGISTRY.clear()


__all__ = [
    "FlangePose",
    "PoseSource",
    "MockPoseSource",
    "NullPoseSource",
    "PlcPoseSource",
    "register_pose_source",
    "get_pose_source",
    "clear_pose_sources",
]
