#!/usr/bin/env python3
"""
ROS2 one-shot camera intrinsic calibration runner.

Runs:
  1) v4l2_camera_node (publishes /image_raw)
  2) camera_calibration cameracalibrator (GUI)

Then extracts `/tmp/calibrationdata.tar.gz` into an output directory and (optionally)
converts the produced `camera.yaml`/`ost.yaml` into this repo's backend format:
  config/calibration/camera_intrinsic.yaml
"""

from __future__ import annotations

import argparse
import os
import shlex
import signal
import subprocess
import sys
import tarfile
import time
from dataclasses import dataclass
from glob import glob
from pathlib import Path
from typing import Any

import yaml

from .ros_convert_intrinsic import save_backend_yaml, to_backend_yaml_dict, load_ros_camera_yaml


def _expand_path(value: str) -> str:
    return os.path.expandvars(os.path.expanduser(value))


def _require_mapping(value: Any, *, name: str) -> dict:
    if not isinstance(value, dict):
        raise ValueError(f"Config field `{name}` must be a mapping, got {type(value).__name__}")
    return value


def _safe_extract_tar(tar: tarfile.TarFile, path: Path) -> None:
    base = path.resolve()
    for member in tar.getmembers():
        member_path = (path / member.name).resolve()
        if not str(member_path).startswith(str(base) + os.sep) and member_path != base:
            raise ValueError(f"Unsafe tar member path: {member.name}")
    tar.extractall(path)  # noqa: S202


@dataclass(frozen=True)
class RunnerConfig:
    ros_setup_bash: str | None
    backend: str
    video_device: str
    video_device_match: str | None
    daheng_device_index: int
    daheng_serial: str | None
    daheng_device_id: str | None
    image_width: int
    image_height: int
    frame_id: str
    output_dir: Path
    extracted_yaml_name: str
    backend_intrinsic_yaml: Path | None
    tarball_path: Path
    pattern: str | None
    target_size: str
    square: float
    no_service_check: bool
    remaps: dict[str, str]
    probe: bool


def load_config(path: str | Path) -> RunnerConfig:
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    raw = _require_mapping(raw, name=str(path))
    ros_cfg = _require_mapping(raw.get("ros", {}), name="ros")
    camera_cfg = _require_mapping(raw.get("camera", {}), name="camera")
    output_cfg = _require_mapping(raw.get("output", {}), name="output")
    calib_cfg = _require_mapping(raw.get("calibrator", {}), name="calibrator")
    tar_cfg = _require_mapping(raw.get("tarball", {}), name="tarball")

    image_size = camera_cfg.get("image_size")
    if (
        not isinstance(image_size, list)
        or len(image_size) != 2
        or not all(isinstance(x, (int, float)) for x in image_size)
    ):
        raise ValueError("`camera.image_size` must be a 2-element list, e.g. [640, 480]")

    backend_yaml = output_cfg.get("backend_intrinsic_yaml")
    backend_yaml_path = Path(_expand_path(backend_yaml)) if isinstance(backend_yaml, str) and backend_yaml else None

    backend = str(camera_cfg.get("backend", "v4l2")).lower()
    if backend not in {"v4l2", "daheng"}:
        raise ValueError("`camera.backend` must be one of: v4l2, daheng")

    daheng_cfg = _require_mapping(camera_cfg.get("daheng", {}), name="camera.daheng")

    return RunnerConfig(
        ros_setup_bash=_expand_path(ros_cfg["setup_bash"]) if ros_cfg.get("setup_bash") else None,
        backend=backend,
        video_device=str(camera_cfg.get("video_device", "/dev/video0")),
        video_device_match=str(camera_cfg.get("match")) if camera_cfg.get("match") else None,
        daheng_device_index=int(daheng_cfg.get("device_index", 1)),
        daheng_serial=str(daheng_cfg.get("serial")) if daheng_cfg.get("serial") else None,
        daheng_device_id=str(daheng_cfg.get("device_id")) if daheng_cfg.get("device_id") else None,
        image_width=int(image_size[0]),
        image_height=int(image_size[1]),
        frame_id=str(camera_cfg.get("frame_id", "camera_link")),
        output_dir=Path(_expand_path(str(output_cfg.get("dir", "data/calibration/ros2_intrinsic")))),
        extracted_yaml_name=str(output_cfg.get("extracted_yaml", "camera.yaml")),
        backend_intrinsic_yaml=backend_yaml_path,
        tarball_path=Path(_expand_path(str(tar_cfg.get("path", "/tmp/calibrationdata.tar.gz")))),
        pattern=str(calib_cfg.get("pattern")) if calib_cfg.get("pattern") else None,
        target_size=str(calib_cfg.get("size", "10x7")),
        square=float(calib_cfg.get("square", 0.015)),
        no_service_check=bool(calib_cfg.get("no_service_check", True)),
        remaps={str(k): str(v) for k, v in (calib_cfg.get("remaps") or {}).items()},
        probe=bool(raw.get("probe", True)),
    )


def _run(cmd: list[str], *, env: dict[str, str] | None = None, dry_run: bool = False) -> int:
    printable = " ".join(cmd)
    print(f"\n$ {printable}")
    if dry_run:
        return 0
    return subprocess.call(cmd, env=env)  # noqa: S603


def _start_background(cmd: list[str], *, env: dict[str, str] | None = None, dry_run: bool = False) -> subprocess.Popen | None:
    printable = " ".join(cmd)
    print(f"\n$ {printable}  (background)")
    if dry_run:
        return None
    return subprocess.Popen(cmd, env=env, start_new_session=True)  # noqa: S603


def _stop_process(proc: subprocess.Popen | None) -> None:
    if proc is None:
        return
    try:
        os.killpg(proc.pid, signal.SIGINT)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            return


def _maybe_source_ros(setup_bash: str | None) -> dict[str, str]:
    if not setup_bash:
        return dict(os.environ)
    setup_path = Path(setup_bash)
    if not setup_path.exists():
        raise FileNotFoundError(f"ROS setup not found: {setup_bash}")

    quoted_setup = shlex.quote(setup_bash)
    cmd = [
        "bash",
        "-lc",
        f"source {quoted_setup} >/dev/null 2>&1 && /usr/bin/python3 -c 'import os; print(\"\\n\".join([k+\"=\"+v for k,v in os.environ.items()]))'",
    ]
    out = subprocess.check_output(cmd, text=True)  # noqa: S603
    env = {}
    for line in out.splitlines():
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        env[k] = v
    return _sanitize_ros_env(env)


def _sanitize_ros_env(env: dict[str, str]) -> dict[str, str]:
    """
    Make the subprocess env resilient against an active virtualenv.

    If the caller's PYTHONPATH contains venv paths, `cameracalibrator` (shebang: /usr/bin/python3)
    may import opencv from the venv and crash with Qt plugin errors. Keep ROS paths only.
    """
    clean = dict(env)
    for k in ("VIRTUAL_ENV", "CONDA_PREFIX", "CONDA_DEFAULT_ENV", "PYTHONHOME"):
        clean.pop(k, None)

    pythonpath = clean.get("PYTHONPATH", "")
    parts = [p for p in pythonpath.split(os.pathsep) if p]
    ros_parts = [p for p in parts if p.startswith("/opt/ros/")]

    if ros_parts:
        seen: set[str] = set()
        uniq: list[str] = []
        for p in ros_parts:
            if p in seen:
                continue
            seen.add(p)
            uniq.append(p)
        clean["PYTHONPATH"] = os.pathsep.join(uniq)
    else:
        clean["PYTHONPATH"] = os.pathsep.join([p for p in parts if ".venv" not in p])

    return clean


def _list_v4l2_candidates() -> list[str]:
    candidates: list[str] = []

    by_id = sorted(glob("/dev/v4l/by-id/*"))
    # prefer stable symlinks first
    for p in by_id:
        if Path(p).is_symlink():
            candidates.append(p)

    for p in sorted(glob("/dev/video*")):
        if Path(p).exists():
            candidates.append(p)

    # de-dup while preserving order
    seen: set[str] = set()
    uniq: list[str] = []
    for c in candidates:
        if c in seen:
            continue
        seen.add(c)
        uniq.append(c)
    return uniq


def _resolve_video_device(video_device: str, match: str | None) -> str:
    if video_device and video_device.lower() != "auto" and Path(video_device).exists():
        return video_device

    candidates = _list_v4l2_candidates()
    if not candidates:
        raise RuntimeError(
            "No V4L2 camera device found under /dev/video* or /dev/v4l/by-id. "
            "If you're using a Daheng/GenICam camera (gxipy), it won't appear as /dev/videoX, "
            "so `v4l2_camera` can't open it; use a Daheng ROS2 driver/node or capture images via gxipy then run "
            "`python -m tools.calibrate.intrinsic`."
        )

    if match:
        lowered = match.lower()
        filtered = [c for c in candidates if lowered in c.lower()]
        if len(filtered) == 1:
            return filtered[0]
        if len(filtered) > 1:
            raise RuntimeError(f"Multiple V4L2 devices match `{match}`: {filtered}. Please set `camera.video_device` explicitly.")
        raise RuntimeError(f"No V4L2 devices match `{match}`. Candidates: {candidates}")

    if len(candidates) == 1:
        return candidates[0]

    raise RuntimeError(
        "Multiple V4L2 devices found; please set `camera.video_device` to a stable path (recommended: /dev/v4l/by-id/...). "
        f"Candidates: {candidates}"
    )


def extract_calibration_tarball(tarball_path: Path, output_dir: Path, extracted_yaml_name: str) -> Path:
    if not tarball_path.exists():
        raise FileNotFoundError(f"Calibration tarball not found: {tarball_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tarball_path, "r:gz") as tar:
        _safe_extract_tar(tar, output_dir)

    # camera_calibration often writes `ost.yaml`
    preferred = output_dir / extracted_yaml_name
    if preferred.exists():
        return preferred

    ost = output_dir / "ost.yaml"
    if ost.exists():
        ost.rename(preferred)
        return preferred

    yaml_files = sorted(output_dir.glob("*.yaml"))
    if len(yaml_files) == 1:
        yaml_files[0].rename(preferred)
        return preferred
    if yaml_files:
        raise RuntimeError(f"Multiple yaml files extracted, please pick one: {[p.name for p in yaml_files]}")
    raise RuntimeError("No YAML extracted from calibration tarball")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Runner YAML config")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing")
    args = parser.parse_args()

    cfg = load_config(args.config)
    env = _maybe_source_ros(cfg.ros_setup_bash)
    calib_env = dict(env)
    calib_env["PYTHONNOUSERSITE"] = "1"
    for k in ("QT_PLUGIN_PATH", "QT_QPA_PLATFORM_PLUGIN_PATH"):
        calib_env.pop(k, None)

    # Optional quick probe
    if cfg.probe:
        if cfg.backend == "v4l2":
            _run(["bash", "-lc", "ls /dev/video* || true"], env=env, dry_run=args.dry_run)
            try:
                video_device = _resolve_video_device(cfg.video_device, cfg.video_device_match)
            except Exception as e:  # noqa: BLE001
                print(f"V4L2 probe error: {e}")
                video_device = cfg.video_device
            _run(
                ["bash", "-lc", f"command -v v4l2-ctl >/dev/null 2>&1 && v4l2-ctl -d {video_device} --list-formats-ext || true"],
                env=env,
                dry_run=args.dry_run,
            )
        else:
            _run(
                [
                    sys.executable,
                    "-m",
                    "tools.calibrate.ros2_daheng_camera_node",
                    "--list",
                ],
                env=env,
                dry_run=args.dry_run,
            )

    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    if cfg.backend == "v4l2":
        video_device = _resolve_video_device(cfg.video_device, cfg.video_device_match)
        camera_cmd = [
            "ros2",
            "run",
            "v4l2_camera",
            "v4l2_camera_node",
            "--ros-args",
            "-p",
            f"video_device:={video_device}",
            "-p",
            f"image_size:=[{cfg.image_width},{cfg.image_height}]",
            "-p",
            f"camera_frame_id:={cfg.frame_id}",
            "-p",
            'camera_info_url:=""',
        ]
        next_camera_hint = (
            "  ros2 run v4l2_camera v4l2_camera_node --ros-args "
            f"-p video_device:={video_device} -p image_size:=[{cfg.image_width},{cfg.image_height}] "
            f"-p camera_frame_id:={cfg.frame_id} -p camera_info_url:={{file_url}}"
        )
    else:
        camera_cmd = [
            sys.executable,
            "-m",
            "tools.calibrate.ros2_daheng_camera_node",
            "--device-index",
            str(cfg.daheng_device_index),
            "--width",
            str(cfg.image_width),
            "--height",
            str(cfg.image_height),
            "--frame-id",
            cfg.frame_id,
            "--image-topic",
            "/image_raw",
            "--camera-info-topic",
            "/camera_info",
        ]
        if cfg.daheng_serial:
            camera_cmd += ["--serial", cfg.daheng_serial]
        if cfg.daheng_device_id:
            camera_cmd += ["--device-id", cfg.daheng_device_id]
        next_camera_hint = (
            "  python3 -m tools.calibrate.ros2_daheng_camera_node "
            f"--device-index {cfg.daheng_device_index} "
            f"{('--serial ' + cfg.daheng_serial + ' ') if cfg.daheng_serial else ''}"
            f"--width {cfg.image_width} --height {cfg.image_height} --frame-id {cfg.frame_id} "
            "--camera-info-yaml {file_path}"
        )

    remap_args: list[str] = []
    for src, dst in cfg.remaps.items():
        remap_args += ["--remap", f"{src}:={dst}"]

    calib_cmd = [
        "ros2",
        "run",
        "camera_calibration",
        "cameracalibrator",
    ]
    if cfg.pattern:
        calib_cmd += ["--pattern", cfg.pattern]
    calib_cmd += [
        "--size",
        cfg.target_size,
        "--square",
        str(cfg.square),
    ]
    if cfg.no_service_check:
        calib_cmd.append("--no-service-check")
    if remap_args:
        calib_cmd += ["--ros-args", *remap_args]

    camera_proc = None
    try:
        camera_proc = _start_background(camera_cmd, env=env, dry_run=args.dry_run)
        if not args.dry_run:
            time.sleep(1.0)

        exit_code = _run(calib_cmd, env=calib_env, dry_run=args.dry_run)
        if exit_code != 0:
            print(f"\ncalibrator exited with code {exit_code}")
            return int(exit_code)
    finally:
        _stop_process(camera_proc)

    if args.dry_run:
        return 0

    extracted = extract_calibration_tarball(cfg.tarball_path, cfg.output_dir, cfg.extracted_yaml_name)
    print(f"\nExtracted ROS calibration yaml: {extracted}")

    if cfg.backend_intrinsic_yaml:
        ros_model = load_ros_camera_yaml(extracted)
        backend_dict = to_backend_yaml_dict(ros_model)
        save_backend_yaml(backend_dict, cfg.backend_intrinsic_yaml)
        print(f"Wrote backend intrinsics: {cfg.backend_intrinsic_yaml}")

    file_url = f"file://{extracted.resolve()}"
    print("\nNext (to publish CameraInfo from this yaml):")
    if cfg.backend == "v4l2":
        print(next_camera_hint.format(file_url=file_url))
    else:
        print(next_camera_hint.format(file_path=str(extracted.resolve())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
