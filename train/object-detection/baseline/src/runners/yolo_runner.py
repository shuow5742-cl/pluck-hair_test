from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from src.core.interfaces import TrainingCallback
from src.core.registry import RUNNERS
from src.runners.base import Runner


@RUNNERS.register("yolo")
class YoloRunner(Runner):
    name = "yolo"

    def run(self, detector, exp_config: Dict, callbacks: List[TrainingCallback]) -> Dict:
        training_cfg = dict(exp_config.get("training", {}))
        experiment_cfg = exp_config.get("experiment", {})

        exp_root = Path(experiment_cfg.get("output_dir", "experiments")) / experiment_cfg.get("name", "run")
        framework_dir = exp_root / "framework"
        training_cfg.setdefault("project", str(exp_root))
        training_cfg.setdefault("name", framework_dir.name)

        for cb in callbacks:
            cb.on_train_begin(exp_config)

        result = detector.train(**training_cfg)

        save_dir = getattr(result, "save_dir", None)
        if save_dir:
            for cb in callbacks:
                cb.on_framework_output(str(save_dir))

        metrics = _extract_metrics(result)

        for cb in callbacks:
            cb.on_train_end(metrics)

        return metrics


def _extract_metrics(result: object) -> Dict:
    if result is None:
        return {}
    if isinstance(result, dict):
        return result
    for attr in ("results_dict", "metrics", "metrics_dict"):
        if hasattr(result, attr):
            metrics = getattr(result, attr)
            if isinstance(metrics, dict):
                return metrics
    return {"status": "completed"}
