"""Console entry points for development workflows."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def _backend_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _run(args: list[str]) -> None:
    root = _backend_root()
    main_py = root / "main.py"
    cmd = [sys.executable, str(main_py), *args]
    raise SystemExit(subprocess.call(cmd, cwd=str(root)))


def dev_run() -> None:
    """Run detection loop with dev config."""
    _run(["--mode", "run", "--config", "./config/settings.dev.yaml"])


def dev_api() -> None:
    """Run API server with dev config."""
    _run(["--mode", "api", "--config", "./config/settings.dev.yaml"])


__all__ = ["dev_run", "dev_api"]
