from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import yaml


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _merge_dict(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged: Dict[str, Any] = deepcopy(base)
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = _merge_dict(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _resolve_base(
    config: Dict[str, Any],
    cwd: Path,
    visited: Optional[Set[Path]] = None,
) -> Dict[str, Any]:
    base_files = config.pop("_base_", None)
    if not base_files:
        return config

    if isinstance(base_files, str):
        base_list: List[str] = [base_files]
    else:
        base_list = list(base_files)

    merged: Dict[str, Any] = {}
    visited = set(visited or set())

    for rel_path in base_list:
        base_path = (cwd / rel_path).resolve()
        if base_path in visited:
            raise ValueError(f"Circular _base_ reference detected: {base_path}")
        if not base_path.exists():
            raise FileNotFoundError(f"Base config not found: {base_path}")
        base_config = _load_yaml(base_path)
        base_config = _resolve_base(
            base_config,
            base_path.parent,
            visited | {base_path},
        )
        merged = _merge_dict(merged, base_config)

    return _merge_dict(merged, config)


def _validate_config(config: Dict[str, Any]) -> None:
    required_top = ["experiment", "detector", "training"]
    missing = [key for key in required_top if key not in config]
    if missing:
        raise ValueError(f"Config missing required sections: {missing}")

    detector_section = config["detector"]
    if "type" not in detector_section:
        raise ValueError("detector.type is required")

    if "runner" not in detector_section and "runner" not in config["training"]:
        raise ValueError("runner must be defined in detector or training section")


def load_config(path: str) -> Dict[str, Any]:
    cfg_path = Path(path).resolve()
    if not cfg_path.exists():
        raise FileNotFoundError(cfg_path)

    config = _load_yaml(cfg_path)
    config = _resolve_base(config, cfg_path.parent, {cfg_path})
    _validate_config(config)
    return config
