"""Load and validate the PLC press/photo route table."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

_NOVA_AXES = ("x", "y", "z", "u", "v", "w")
_EPSON_AXES = ("x", "y", "z", "u")


@dataclass(frozen=True)
class PlcPoint:
    """One row in the press/photo route table."""

    press_index: int
    photo_index: int
    nova2: Dict[str, float]                    # 6-axis pose
    nova5: Dict[str, float]                    # 6-axis pose
    epson_ls6_fallback: Optional[Dict[str, float]]  # 4-axis, optional
    repeat: int

    @property
    def key(self) -> str:
        return f"{self.press_index}-{self.photo_index}"


def load_points(path: str | Path) -> List[PlcPoint]:
    """Read ``plc_points.yaml`` into a list of validated PlcPoint."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"plc_points file not found: {p}")
    with p.open(encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    raw_points = data.get("points") or []
    if not isinstance(raw_points, list) or not raw_points:
        raise ValueError(f"{p}: missing or empty 'points' list")

    out: List[PlcPoint] = []
    for i, item in enumerate(raw_points):
        out.append(_parse_point(item, index=i, path=str(p)))
    return out


def _parse_point(item: Any, *, index: int, path: str) -> PlcPoint:
    if not isinstance(item, dict):
        raise ValueError(f"{path} points[{index}] is not a mapping")
    nova2 = _require_pose(item.get("nova2"), _NOVA_AXES, f"{path} points[{index}].nova2")
    nova5 = _require_pose(item.get("nova5"), _NOVA_AXES, f"{path} points[{index}].nova5")
    fallback_raw = item.get("epson_ls6_fallback")
    fallback = (
        _require_pose(fallback_raw, _EPSON_AXES, f"{path} points[{index}].epson_ls6_fallback")
        if fallback_raw is not None
        else None
    )
    return PlcPoint(
        press_index=int(item.get("press_index", index + 1)),
        photo_index=int(item.get("photo_index", index + 1)),
        nova2=nova2,
        nova5=nova5,
        epson_ls6_fallback=fallback,
        repeat=max(1, int(item.get("repeat", 1))),
    )


def _require_pose(raw: Any, axes: tuple[str, ...], where: str) -> Dict[str, float]:
    if not isinstance(raw, dict):
        raise ValueError(f"{where}: expected mapping of {axes}, got {type(raw).__name__}")
    pose: Dict[str, float] = {}
    for ax in axes:
        if ax not in raw:
            raise ValueError(f"{where}: missing axis '{ax}'")
        pose[ax] = float(raw[ax])
    return pose
