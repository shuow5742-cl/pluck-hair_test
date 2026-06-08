from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict

import yaml
from ultralytics import YOLO


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _log_event(log_path: Path, payload: Dict[str, Any]) -> None:
    payload.setdefault("timestamp", time.time())
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _resolve_path(value: str | None, root_hint: str | None) -> str | None:
    if not value:
        return None
    path = Path(value)
    if not path.is_absolute() and root_hint:
        path = Path(root_hint) / path
    return str(path.resolve())


def _prepare_train_args(framework_args: Dict[str, Any], shared_resources: Dict[str, Any], output_dir: Path) -> tuple[str, str, Dict[str, Any]]:
    args = dict(framework_args)
    model_path = args.pop("model", "yolov8n.pt")
    data_yaml = args.pop("data_yaml", None)
    task_type = args.pop("task_type", "hbb")
    datasets_root = shared_resources.get("datasets_root")
    weights_root = shared_resources.get("weights_cache")

    resolved_model = _resolve_path(model_path, weights_root) or model_path
    resolved_data = _resolve_path(data_yaml, datasets_root)
    if not resolved_data:
        raise ValueError("framework_args.data_yaml is required for YOLO training")

    train_kwargs = dict(args)
    train_kwargs.setdefault("project", str(output_dir))
    train_kwargs.setdefault("name", "yolo_run")
    train_kwargs["data"] = resolved_data
    return resolved_model, task_type, train_kwargs


def _build_artifacts(save_dir: Path) -> Dict[str, str]:
    weights_dir = save_dir / "weights"
    return {
        "best_checkpoint": str((weights_dir / "best.pt").resolve()),
        "last_checkpoint": str((weights_dir / "last.pt").resolve()),
        "export": {}
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Isolated YOLO trainer")
    parser.add_argument("--exp-config", required=True, help="Path to orchestrator-generated config")
    parser.add_argument("--output-dir", required=True, help="Experiment output directory")
    parser.add_argument("--run-id", required=True, help="Run identifier")
    args = parser.parse_args()

    exp_cfg = _load_yaml(Path(args.exp_config))
    shared_resources = exp_cfg.get("shared_resources", {})
    framework_args = exp_cfg.get("framework_args", {})

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "framework.log"

    _log_event(log_path, {"event": "train_begin", "run_id": args.run_id})

    model_path, task_type, train_kwargs = _prepare_train_args(framework_args, shared_resources, output_dir)
    train_kwargs.setdefault("name", args.run_id)

    model = YOLO(model_path)

    def on_epoch_end(trainer):  # type: ignore[override]
        payload = {"event": "epoch_end", "epoch": getattr(trainer, "epoch", None)}
        metrics = getattr(trainer, "metrics", None)
        if isinstance(metrics, dict):
            payload["metrics"] = metrics
        losses = getattr(trainer, "loss_items", None)
        if losses is not None:
            try:
                payload["loss_items"] = [float(x) for x in list(losses)]
            except Exception:  # pragma: no cover - defensive
                payload["loss_items"] = str(losses)
        _log_event(log_path, payload)

    try:
        model.add_callback("on_fit_epoch_end", on_epoch_end)
    except Exception:  # pragma: no cover - fallback if callback name changes
        model.add_callback("on_train_epoch_end", on_epoch_end)

    training_result = model.train(**train_kwargs)

    metrics: Dict[str, Any] = {}
    for attr in ("results_dict", "metrics", "metrics_dict"):
        value = getattr(training_result, attr, None)
        if isinstance(value, dict):
            metrics = value
            break
    save_dir = Path(getattr(training_result, "save_dir", output_dir))

    artifacts = _build_artifacts(save_dir)
    metrics_payload = {
        "task_type": framework_args.get("task_type", task_type),
        "metrics": metrics,
        "training_curves": {},
        "artifacts": artifacts,
    }

    metrics_path = output_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, ensure_ascii=False, indent=2)

    _log_event(log_path, {"event": "train_end", "run_id": args.run_id, "metrics_file": str(metrics_path)})


if __name__ == "__main__":
    main()
