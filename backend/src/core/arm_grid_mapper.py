"""Nearest-point XY mapping from Nova5-space into Epson-space.

This is used when pixel_to_world produces a real XY in Nova5's calibrated
coordinate space, but the Epson LS6 needs a locally compensated XY built
from a separate 10x10 calibration grid.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import yaml


@dataclass(frozen=True)
class ArmGridPoint:
    """One matched Nova5/Epson calibration anchor."""

    row: int
    col: int
    nova5_x: float
    nova5_y: float
    epson_x: float
    epson_y: float


@dataclass(frozen=True)
class ArmGridMatch:
    """Result of mapping one Nova5 XY into Epson XY."""

    anchor: ArmGridPoint
    offset_x: float
    offset_y: float
    distance_mm: float
    epson_x: float
    epson_y: float


class ArmGridMapper:
    """Map Nova5 XY into Epson XY using the nearest calibration anchor."""

    def __init__(
        self,
        points: Iterable[ArmGridPoint],
        *,
        rows: int | None = None,
        cols: int | None = None,
    ) -> None:
        self._points = list(points)
        if not self._points:
            raise ValueError("ArmGridMapper requires at least one calibration point")
        self.rows = rows
        self.cols = cols
        seen: set[tuple[int, int]] = set()
        for point in self._points:
            key = (point.row, point.col)
            if key in seen:
                raise ValueError(f"duplicate calibration grid index row={point.row} col={point.col}")
            seen.add(key)

    @classmethod
    def load(cls, path: str | Path) -> "ArmGridMapper":
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"arm grid mapping file not found: {p}")
        with p.open(encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        grid = data.get("grid") or {}
        raw_pairs = grid.get("pairs") or []
        if not isinstance(raw_pairs, list) or not raw_pairs:
            raise ValueError(f"{p}: missing or empty grid.pairs list")

        points = [_parse_grid_point(item, index=i, path=str(p)) for i, item in enumerate(raw_pairs)]
        rows = grid.get("rows")
        cols = grid.get("cols")
        return cls(
            points,
            rows=int(rows) if rows is not None else None,
            cols=int(cols) if cols is not None else None,
        )

    def map_nova5_to_epson(self, x: float, y: float) -> ArmGridMatch:
        anchor = min(
            self._points,
            key=lambda point: (point.nova5_x - x) ** 2 + (point.nova5_y - y) ** 2,
        )
        offset_x = float(x) - anchor.nova5_x
        offset_y = float(y) - anchor.nova5_y
        return ArmGridMatch(
            anchor=anchor,
            offset_x=offset_x,
            offset_y=offset_y,
            distance_mm=math.hypot(offset_x, offset_y),
            epson_x=anchor.epson_x + offset_x,
            epson_y=anchor.epson_y + offset_y,
        )


def _parse_grid_point(item: Any, *, index: int, path: str) -> ArmGridPoint:
    if not isinstance(item, dict):
        raise ValueError(f"{path} grid.pairs[{index}] is not a mapping")
    nova5 = _require_xy(item.get("nova5"), f"{path} grid.pairs[{index}].nova5")
    epson = _require_xy(item.get("epson"), f"{path} grid.pairs[{index}].epson")
    return ArmGridPoint(
        row=int(item.get("row")),
        col=int(item.get("col")),
        nova5_x=nova5["x"],
        nova5_y=nova5["y"],
        epson_x=epson["x"],
        epson_y=epson["y"],
    )


def _require_xy(raw: Any, where: str) -> dict[str, float]:
    if not isinstance(raw, dict):
        raise ValueError(f"{where}: expected mapping with x/y")
    missing = [axis for axis in ("x", "y") if axis not in raw]
    if missing:
        raise ValueError(f"{where}: missing axis {missing[0]!r}")
    return {"x": float(raw["x"]), "y": float(raw["y"])}
