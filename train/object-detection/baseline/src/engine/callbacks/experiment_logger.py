from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import yaml

from src.core.interfaces import TrainingCallback


class ExperimentLogger(TrainingCallback):
    def __init__(self, experiment_cfg: Dict, base_dir: str | None = None):
        self.experiment_cfg = experiment_cfg
        exp_root = Path(base_dir or experiment_cfg.get("output_dir", "experiments"))
        self.exp_dir = exp_root / experiment_cfg.get("name", "run")
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        self.config_written = False

    def on_train_begin(self, config: Dict):
        if not self.config_written:
            config_path = self.exp_dir / "config.yaml"
            with config_path.open("w", encoding="utf-8") as f:
                yaml.safe_dump(config, f, sort_keys=False, allow_unicode=True)
            self.config_written = True

    def on_framework_output(self, path: str) -> None:
        output_file = self.exp_dir / "framework_output.txt"
        output_file.write_text(path, encoding="utf-8")

    def on_train_end(self, metrics: Dict):
        metrics_path = self.exp_dir / "metrics.json"
        with metrics_path.open("w", encoding="utf-8") as f:
            json.dump(
                metrics,
                f,
                indent=2,
                ensure_ascii=False,
                default=_json_default,
            )


def _json_default(obj: Any):
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (set, frozenset)):
        return list(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")
