"""Tests for Nova5→Epson nearest-grid XY compensation."""

from __future__ import annotations

import math

from src.config import AppConfig
from src.core.arm_grid_mapper import ArmGridMapper


def test_arm_grid_mapper_applies_nearest_anchor_offset(tmp_path):
    grid_path = tmp_path / "grid.yaml"
    grid_path.write_text(
        """
grid:
  rows: 2
  cols: 2
  pairs:
    - row: 0
      col: 1
      nova5: {x: 0.0, y: 0.0}
      epson: {x: 100.0, y: 200.0}
    - row: 0
      col: 2
      nova5: {x: 10.0, y: 0.0}
      epson: {x: 110.0, y: 200.0}
    - row: 1
      col: 1
      nova5: {x: 0.0, y: 10.0}
      epson: {x: 100.0, y: 210.0}
    - row: 1
      col: 2
      nova5: {x: 10.0, y: 10.0}
      epson: {x: 110.0, y: 210.0}
""".strip(),
        encoding="utf-8",
    )

    mapper = ArmGridMapper.load(grid_path)
    match = mapper.map_nova5_to_epson(10.3, 10.4)

    assert (match.anchor.row, match.anchor.col) == (1, 2)
    assert math.isclose(match.offset_x, 0.3, abs_tol=1e-9)
    assert math.isclose(match.offset_y, 0.4, abs_tol=1e-9)
    assert math.isclose(match.epson_x, 110.3, abs_tol=1e-9)
    assert math.isclose(match.epson_y, 210.4, abs_tol=1e-9)


def test_plc_orchestrator_config_parses_nova5_to_epson_mapping():
    config = AppConfig.from_dict({
        "plc_orchestrator": {
            "enabled": True,
            "epson_u_min_deg": 30.0,
            "epson_u_max_deg": 110.0,
            "pick_confirm_match_distance_px": 42.0,
            "pick_confirm_match_size_ratio": 0.45,
            "nova5_to_epson_mapping": {
                "enabled": True,
                "path": "config/custom_grid.yaml",
            },
        },
    })

    assert config.plc_orchestrator.nova5_to_epson_mapping.enabled is True
    assert config.plc_orchestrator.nova5_to_epson_mapping.path == "config/custom_grid.yaml"
    assert math.isclose(config.plc_orchestrator.epson_u_min_deg, 30.0, abs_tol=1e-9)
    assert math.isclose(config.plc_orchestrator.epson_u_max_deg, 110.0, abs_tol=1e-9)
    assert math.isclose(
        config.plc_orchestrator.pick_confirm_match_distance_px,
        42.0,
        abs_tol=1e-9,
    )
    assert math.isclose(
        config.plc_orchestrator.pick_confirm_match_size_ratio,
        0.45,
        abs_tol=1e-9,
    )
