#!/usr/bin/env python3
"""
Sequential training queue runner.

This script reads pipelines/queue.yaml, verifies GPU availability,
executes each task via subprocess, captures logs, and updates task status.
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import yaml
except ModuleNotFoundError as exc:  # pragma: no cover - dependency should exist
        raise SystemExit("PyYAML is required to run the queue. Install with `pip install pyyaml`.") from exc


REPO_ROOT = Path(__file__).resolve().parents[1]
PIPELINES_DIR = REPO_ROOT / "pipelines"
LOG_DIR = PIPELINES_DIR / "logs"
TASK_LOG_DIR = PIPELINES_DIR / "task_logs"


def timestamp() -> str:
    return dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def slugify(value: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in value)
    return safe.strip("_") or "task"


@dataclass
class Task:
    index: int
    raw: Dict[str, Any]
    name: str
    command: List[str]
    description: str = ""
    device: List[int] = field(default_factory=lambda: [0])
    min_free_mem_gb: Optional[float] = None
    max_retries: int = 0
    retry_interval_sec: int = 60
    output_dir: Optional[Path] = None
    status: str = "pending"
    workdir: Path = REPO_ROOT
    env: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if isinstance(self.command, str):
            # Split string using shlex for backwards compatibility.
            self.command = shlex.split(self.command)
        if not self.command:
            raise ValueError(f"Task '{self.name}' has an empty command.")
        self.device = [int(d) for d in (self.device or [0])]
        if self.output_dir:
            self.output_dir = (REPO_ROOT / self.output_dir).resolve()
        if "status" in self.raw:
            self.status = self.raw["status"]
        if "workdir" in self.raw:
            self.workdir = (REPO_ROOT / self.raw["workdir"]).resolve()
        if "env" in self.raw and isinstance(self.raw["env"], dict):
            self.env = {k: str(v) for k, v in self.raw["env"].items()}

    def mark_status(self, new_status: str) -> None:
        self.status = new_status
        self.raw["status"] = new_status


def load_queue(path: Path) -> List[Task]:
    if not path.exists():
        raise FileNotFoundError(f"Queue file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or []
    if not isinstance(data, list):
        raise ValueError("Queue file must contain a list of tasks.")
    tasks: List[Task] = []
    for idx, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"Task at index {idx} must be a mapping.")
        task = Task(
            index=idx,
            raw=item,
            name=str(item.get("name") or f"task_{idx}"),
            command=item.get("command") or [],
            description=str(item.get("description") or ""),
            device=item.get("device") or [0],
            min_free_mem_gb=item.get("min_free_mem_gb"),
            max_retries=int(item.get("max_retries") or 0),
            retry_interval_sec=int(item.get("retry_interval_sec") or 60),
            output_dir=Path(item["output_dir"]) if item.get("output_dir") else None,
        )
        tasks.append(task)
    return tasks


def save_queue(path: Path, tasks: List[Task]) -> None:
    serialisable = [task.raw for task in tasks]
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(serialisable, f, sort_keys=False)


def read_gpu_stats() -> Optional[Dict[int, Dict[str, float]]]:
    """Return {gpu_id: {total, used, free}} in GB."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.total,memory.used",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
    except FileNotFoundError:
        return None
    except subprocess.CalledProcessError:
        return None

    stats: Dict[int, Dict[str, float]] = {}
    for line in result.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 3:
            continue
        gpu_id, total, used = parts
        try:
            gpu_idx = int(gpu_id)
            total_val = float(total)
            used_val = float(used)
        except ValueError:
            continue

        # nvidia-smi returns MiB without units; keep total/used in the same scale before diff.
        if total_val > 1024:
            total_gb = total_val / 1024
            used_gb = used_val / 1024
        else:
            total_gb = total_val
            used_gb = used_val

        stats[gpu_idx] = {
            "total": total_gb,
            "used": used_gb,
            "free": max(total_gb - used_gb, 0.0),
        }
    return stats


def wait_for_gpu(task: Task, log) -> None:
    if task.min_free_mem_gb is None:
        return
    while True:
        stats = read_gpu_stats()
        if stats is None:
            log.write(f"{timestamp()} [WARN] nvidia-smi unavailable; skipping GPU free-memory check.\n")
            log.flush()
            return
        unmet = []
        for dev in task.device:
            gpu = stats.get(dev)
            if not gpu:
                unmet.append(f"GPU{dev} unavailable")
                continue
            if gpu["free"] < task.min_free_mem_gb:
                unmet.append(f"GPU{dev} free {gpu['free']:.1f}GB < {task.min_free_mem_gb}GB")
        if not unmet:
            return
        log.write(f"{timestamp()} [INFO] Waiting for GPU resources ({'; '.join(unmet)})...\n")
        log.flush()
        time.sleep(60)


def run_subprocess(task: Task, log_file: Path, aggregate_log) -> int:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    attempt_env = os.environ.copy()
    attempt_env.update(task.env)
    with log_file.open("w", encoding="utf-8") as task_log:
        header = (
            f"{timestamp()} Task '{task.name}'\n"
            f"Command: {' '.join(shlex.quote(c) for c in task.command)}\n"
            f"Workdir: {task.workdir}\n"
        )
        task_log.write(header)
        task_log.flush()
        aggregate_log.write(f"{timestamp()} [INFO] Spawning task '{task.name}'. Logs -> {log_file}\n")
        aggregate_log.flush()
        process = subprocess.Popen(
            task.command,
            cwd=task.workdir,
            env=attempt_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        try:
            assert process.stdout is not None
            for line in process.stdout:
                task_log.write(line)
                task_log.flush()
                sys.stdout.write(f"[{task.name}] {line}")
            return process.wait()
        except KeyboardInterrupt:
            process.terminate()
            process.wait()
            raise


def summarize_queue(tasks: List[Task]) -> str:
    parts = []
    for t in tasks:
        parts.append(f"{t.name}: {t.status}")
    return ", ".join(parts)


def run_queue(tasks: List[Task], queue_path: Path, args) -> int:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    TASK_LOG_DIR.mkdir(parents=True, exist_ok=True)
    queue_log_path = LOG_DIR / f"queue_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    with queue_log_path.open("w", encoding="utf-8") as qlog:
        qlog.write(f"{timestamp()} Queue start. Tasks: {len(tasks)}\n")
        if args.resume:
            qlog.write(f"{timestamp()} Resume mode: completed tasks will be skipped.\n")
        if args.start_from:
            qlog.write(f"{timestamp()} Start-from filter: {args.start_from}\n")
        qlog.flush()
        start_index_reached = not bool(args.start_from)
        exit_code = 0
        for idx, task in enumerate(tasks, start=1):
            if args.start_from and task.name == args.start_from:
                start_index_reached = True
            if not start_index_reached:
                qlog.write(f"{timestamp()} [SKIP] {task.name} (before start_from)\n")
                qlog.flush()
                continue
            if args.resume and task.status == "success":
                qlog.write(f"{timestamp()} [SKIP] {task.name} already successful.\n")
                qlog.flush()
                continue
            qlog.write(f"{timestamp()} [RUN] ({idx}/{len(tasks)}) {task.name}\n")
            qlog.flush()
            attempt = 0
            while True:
                wait_for_gpu(task, qlog)
                log_path = TASK_LOG_DIR / f"{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}_{slugify(task.name)}.log"
                try:
                    code = run_subprocess(task, log_path, qlog)
                except KeyboardInterrupt:
                    qlog.write(f"{timestamp()} [ABORT] User interruption.\n")
                    qlog.flush()
                    save_queue(queue_path, tasks)
                    raise
                if code == 0:
                    task.mark_status("success")
                    save_queue(queue_path, tasks)
                    qlog.write(f"{timestamp()} [OK] {task.name} completed.\n")
                    qlog.flush()
                    break
                attempt += 1
                task.mark_status("failed")
                save_queue(queue_path, tasks)
                qlog.write(
                    f"{timestamp()} [ERR] {task.name} exited with code {code} "
                    f"(attempt {attempt}/{task.max_retries}). See {log_path}\n"
                )
                qlog.flush()
                if attempt > task.max_retries:
                    qlog.write(f"{timestamp()} [FAIL] Giving up on {task.name}.\n")
                    exit_code = code
                    break
                qlog.write(
                    f"{timestamp()} [INFO] Retrying {task.name} after {task.retry_interval_sec}s...\n"
                )
                qlog.flush()
                time.sleep(task.retry_interval_sec)
            if exit_code != 0 and not args.keep_going:
                qlog.write(
                    f"{timestamp()} [STOP] Aborting queue due to failure and keep-going disabled.\n"
                )
                qlog.flush()
                break
        qlog.write(f"{timestamp()} Queue finished. Summary: {summarize_queue(tasks)}\n")
        qlog.flush()
    return exit_code


def print_status(tasks: List[Task]) -> None:
    print("Current queue status:")
    for task in tasks:
        print(f"- {task.name}: {task.status}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run sequential training queue.")
    parser.add_argument(
        "--queue",
        type=Path,
        default=PIPELINES_DIR / "queue.yaml",
        help="Path to queue YAML file.",
    )
    parser.add_argument("--resume", action="store_true", help="Skip tasks already marked success.")
    parser.add_argument("--start-from", help="Task name to start from.")
    parser.add_argument("--check", action="store_true", help="Only validate queue file and exit.")
    parser.add_argument("--status", action="store_true", help="Print queue status and exit.")
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="Continue with next tasks even if one fails.",
    )
    args = parser.parse_args()

    queue_path = args.queue if args.queue.is_absolute() else (REPO_ROOT / args.queue)
    tasks = load_queue(queue_path)

    if args.status:
        print_status(tasks)
        return 0
    if args.check:
        print("Queue validation OK. Tasks:")
        for task in tasks:
            cmd = " ".join(shlex.quote(c) for c in task.command)
            print(f"- {task.name}: {cmd} (status={task.status})")
        return 0

    return run_queue(tasks, queue_path, args)


if __name__ == "__main__":
    raise SystemExit(main())
