from __future__ import annotations

import argparse
import datetime as dt
import subprocess
from pathlib import Path
from typing import Any, Dict

import yaml


def _read_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _write_yaml(path: Path, data: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)


def _log(log_path: Path, message: str) -> None:
    timestamp = dt.datetime.now().isoformat()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] {message}\n")


def _resolve_project_and_script(framework_cfg: Dict[str, Any]) -> tuple[Path, Path]:
    script_path = Path(framework_cfg["script"]).resolve()
    project_dir = Path(framework_cfg.get("project_dir", script_path.parent.parent)).resolve()
    if not project_dir.exists():
        project_dir = script_path.parent.resolve()
    return project_dir, script_path


def dispatch(config_path: Path, dry_run: bool = False) -> None:
    cfg = _read_yaml(config_path)
    experiment = cfg["experiment"]
    framework_cfg = cfg["framework_cfg"]
    shared_resources = cfg.get("shared_resources", {})

    output_dir = Path(experiment.get("output_dir", f"experiments/{experiment['name']}"))
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    project_dir, script_path = _resolve_project_and_script(framework_cfg)
    if not script_path.exists():
        raise FileNotFoundError(f"Framework script not found: {script_path}")

    run_id = framework_cfg.get("run_id") or experiment.get("run_id") or experiment["name"]
    exp_config_payload: Dict[str, Any] = {
        "experiment": experiment,
        "shared_resources": shared_resources,
        "framework_args": framework_cfg.get("args", {}),
    }
    exp_config_file = output_dir / "exp_config.generated.yaml"
    _write_yaml(exp_config_file, exp_config_payload)

    log_path = output_dir / "orchestrator.log"
    _log(log_path, f"Dispatching framework '{experiment['framework']}' via {project_dir} -> {script_path}")

    cmd = [
        "uv",
        "run",
        "--project",
        str(project_dir),
        "python",
        str(script_path),
        "--exp-config",
        str(exp_config_file),
        "--output-dir",
        str(output_dir),
        "--run-id",
        run_id,
    ]

    if dry_run:
        _log(log_path, f"Dry run enabled. Command: {' '.join(cmd)}")
        print("[orchestrator] dry-run", " ".join(cmd))
        return

    process = subprocess.run(cmd, check=False)
    if process.returncode != 0:
        _log(log_path, f"Framework process failed with code {process.returncode}")
        raise SystemExit(process.returncode)
    _log(log_path, "Framework process completed successfully")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dispatch multi-framework training jobs")
    parser.add_argument("--config", required=True, help="Path to orchestrator experiment YAML")
    parser.add_argument("--dry-run", action="store_true", help="Print command without executing")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    dispatch(config_path, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
