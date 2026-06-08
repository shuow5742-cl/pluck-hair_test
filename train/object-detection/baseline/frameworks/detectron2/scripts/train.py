from __future__ import annotations

import argparse
import json
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml
from PIL import Image

# Pillow 11+ renamed resampling constants; Detectron2 still expects Image.LINEAR.
if not hasattr(Image, "LINEAR"):
    if hasattr(Image, "Resampling"):
        Image.LINEAR = Image.Resampling.BILINEAR  # type: ignore[attr-defined]
    else:  # fallback to closest existing constant
        Image.LINEAR = Image.BILINEAR  # type: ignore[assignment]

from detectron2.config import CfgNode, get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer, hooks
from detectron2.utils.logger import setup_logger


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_DIR = SCRIPT_PATH.parents[1]
FRAMEWORKS_DIR = PROJECT_DIR.parent
REPO_ROOT = FRAMEWORKS_DIR.parent if FRAMEWORKS_DIR.name == "frameworks" else PROJECT_DIR.parent

def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _log_event(log_path: Path, payload: Dict[str, Any]) -> None:
    payload.setdefault("timestamp", time.time())
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _resolve_root_hint(value: str | Path | None) -> Path | None:
    if not value:
        return None
    path = Path(value)
    if not path.is_absolute():
        path = (REPO_ROOT / path).resolve()
    return path


def _resolve_path(
    value: str | Path,
    *,
    root_hint: str | Path | None = None,
    default_base: Path | None = None,
    must_exist: bool = False,
) -> Path:
    path = Path(value)
    if path.is_absolute():
        resolved = path
    else:
        candidates: List[Path] = []
        if root_hint:
            candidates.append(Path(root_hint))
        if default_base:
            candidates.append(default_base)
        candidates.append(REPO_ROOT)
        for base in candidates:
            candidate = (base / path).resolve()
            if not must_exist or candidate.exists():
                resolved = candidate
                break
        else:
            resolved = (candidates[0] if candidates else Path.cwd()) / path
    resolved = resolved.resolve()
    if must_exist and not resolved.exists():
        raise FileNotFoundError(f"Path not found: {resolved}")
    return resolved


def _resolve_base_config(name: str) -> Path:
    errors: List[str] = []
    try:
        from detectron2.model_zoo import get_config_file

        cfg_path = Path(get_config_file(name)).resolve()
        if cfg_path.exists():
            return cfg_path
    except Exception as exc:  # pragma: no cover - best effort
        errors.append(f"model_zoo lookup failed: {exc}")

    direct = Path(name)
    if direct.exists():
        return direct.resolve()

    mirror = PROJECT_DIR / "detectron2" / "configs" / name
    if mirror.exists():
        return mirror.resolve()

    raise FileNotFoundError(f"Unable to resolve base_config '{name}'. Attempts: {errors or ['model_zoo']} -> {direct} -> {mirror}")


def _flatten_overrides(overrides: Dict[str, Any]) -> List[str]:
    items: List[str] = []

    def _encode(value: Any) -> str:
        if isinstance(value, bool):
            return "true" if value else "false"
        if value is None:
            return "null"
        if isinstance(value, (int, float)):
            return str(value)
        if isinstance(value, (list, dict)):
            return json.dumps(value)
        return str(value)

    for key, value in overrides.items():
        items.append(key)
        items.append(_encode(value))
    return items


def _register_dataset(dataset_cfg: Dict[str, Any], datasets_root: str | None) -> Tuple[str, str]:
    name = dataset_cfg["name"]
    train_name = f"{name}_train"
    val_name = f"{name}_val"

    train_json = _resolve_path(dataset_cfg["train_json"], root_hint=datasets_root, default_base=REPO_ROOT, must_exist=True)
    train_images = _resolve_path(dataset_cfg["train_images"], root_hint=datasets_root, default_base=REPO_ROOT, must_exist=True)
    val_json = _resolve_path(dataset_cfg["val_json"], root_hint=datasets_root, default_base=REPO_ROOT, must_exist=True)
    val_images = _resolve_path(dataset_cfg["val_images"], root_hint=datasets_root, default_base=REPO_ROOT, must_exist=True)

    thing_classes = dataset_cfg.get("thing_classes")

    for ds_name in (train_name, val_name):
        if ds_name in DatasetCatalog.list():
            DatasetCatalog.remove(ds_name)

    register_coco_instances(train_name, {}, str(train_json), str(train_images))
    register_coco_instances(val_name, {}, str(val_json), str(val_images))

    metadata_kwargs = {}
    if thing_classes:
        metadata_kwargs["thing_classes"] = list(thing_classes)

    if metadata_kwargs:
        for ds_name in (train_name, val_name):
            MetadataCatalog.get(ds_name).set(**metadata_kwargs)

    return train_name, val_name


class JsonLineHook(hooks.HookBase):
    def __init__(self, log_path: Path):
        self.log_path = log_path

    def after_step(self) -> None:  # type: ignore[override]
        storage = self.trainer.storage
        latest = storage.latest()
        payload = {"event": "iter_end", "iteration": int(storage.iter)}
        metrics: Dict[str, Any] = {}
        for key, value in latest.items():
            try:
                metrics[key] = float(value)
            except Exception:  # pragma: no cover - fallback for non-float scalars
                metrics[key] = value if isinstance(value, (int, float, str)) else str(value)
        if metrics:
            payload["metrics"] = metrics
        _log_event(self.log_path, payload)


def _build_cfg(
    framework_args: Dict[str, Any],
    shared_resources: Dict[str, Any],
    output_dir: Path,
) -> tuple[CfgNode, Dict[str, Any]]:
    cfg = get_cfg()
    config_section = framework_args.get("config", {})
    overrides = framework_args.get("overrides", {})
    dataset_cfg = framework_args.get("dataset")
    trainer_cfg = framework_args.get("trainer", {})

    if base_config := config_section.get("base_config"):
        cfg.merge_from_file(str(_resolve_base_config(base_config)))
    if config_file := config_section.get("config_file"):
        cfg.merge_from_file(str(_resolve_path(config_file, default_base=PROJECT_DIR, must_exist=True)))

    if overrides:
        cfg.merge_from_list(_flatten_overrides(overrides))

    cfg.OUTPUT_DIR = str(output_dir)

    weights_root = _resolve_root_hint(shared_resources.get("weights_cache"))
    weights_path = config_section.get("weights")
    if weights_path:
        resolved_weights = _resolve_path(
            weights_path,
            root_hint=weights_root,
            default_base=PROJECT_DIR,
            must_exist=False,
        )
        cfg.MODEL.WEIGHTS = str(resolved_weights)

    datasets_root = _resolve_root_hint(shared_resources.get("datasets_root"))
    if dataset_cfg:
        train_name, val_name = _register_dataset(dataset_cfg, datasets_root)
        cfg.DATASETS.TRAIN = (train_name,)
        cfg.DATASETS.TEST = (val_name,)

    if eval_period := trainer_cfg.get("eval_period"):
        cfg.TEST.EVAL_PERIOD = int(eval_period)
    if amp := trainer_cfg.get("amp"):
        cfg.SOLVER.AMP.ENABLED = bool(amp)
    if seed := trainer_cfg.get("seed"):
        cfg.SEED = int(seed)

    return cfg, trainer_cfg


def _build_artifacts(output_dir: Path) -> Dict[str, str]:
    return {
        "best_checkpoint": str((output_dir / "model_final.pth").resolve()),
        "last_checkpoint": str((output_dir / "last_checkpoint").resolve()),
        "export": {},
    }


def _ordered_to_dict(metrics: OrderedDict | Dict[str, Any]) -> Dict[str, Any]:
    if isinstance(metrics, OrderedDict):
        return {k: _ordered_to_dict(v) if isinstance(v, OrderedDict) else v for k, v in metrics.items()}
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Isolated Detectron2 trainer")
    parser.add_argument("--exp-config", required=True, help="Path to orchestrator-generated config")
    parser.add_argument("--output-dir", required=True, help="Experiment output directory")
    parser.add_argument("--run-id", required=True, help="Run identifier")
    parser.add_argument("--resume", action="store_true", help="Force resume even if trainer config disables it")
    args = parser.parse_args()

    exp_cfg = _load_yaml(Path(args.exp_config))
    shared_resources = exp_cfg.get("shared_resources", {})
    framework_args = exp_cfg.get("framework_args", {})

    output_dir = Path(args.output_dir).resolve()
    _ensure_dir(output_dir)
    log_path = output_dir / "framework.log"

    setup_logger(output=str(output_dir))

    _log_event(log_path, {"event": "train_begin", "run_id": args.run_id})

    cfg, trainer_cfg = _build_cfg(framework_args, shared_resources, output_dir)

    trainer = DefaultTrainer(cfg)
    trainer.register_hooks([JsonLineHook(log_path)])

    resume_flag = bool(trainer_cfg.get("resume", False))
    if args.resume:
        resume_flag = True

    trainer.resume_or_load(resume=resume_flag)
    trainer.train()

    metrics_result = trainer.test(cfg, trainer.model)
    if isinstance(metrics_result, (OrderedDict, dict)):
        metrics = _ordered_to_dict(metrics_result)
    else:
        metrics = {"result": metrics_result}

    artifacts = _build_artifacts(Path(cfg.OUTPUT_DIR))
    metrics_payload = {
        "status": "completed",
        "task_type": framework_args.get("task_type", "hbb"),
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
