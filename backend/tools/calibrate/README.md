# Calibration tools

## Existing (offline, OpenCV)

- Intrinsics: ROS2 one-shot flow below
- Eye-in-hand: `python -m tools.calibrate.eye_in_hand --help`
- Verify: `python -m tools.calibrate.verify --help`

These tools operate on locally captured images and write backend-format YAML under `config/calibration/`.

## ROS2 one-shot intrinsics (v4l2_camera + camera_calibration)

This repo is not a ROS2 package, but you can still use ROS2 tools for capture + calibration and keep results under this repo.

- One-time machine setup (recommended for blank machines):
  - This script will auto-configure ROS2 apt source/key on Ubuntu.

```bash
bash scripts/setup_calibration_env.sh --ros-distro ${ROS_DISTRO}
```

- Config: `tools/calibrate/ros2_intrinsic.example.yaml`
- Runner: `bash tools/calibrate/ros2_intrinsic_calibrate.sh tools/calibrate/ros2_intrinsic.example.yaml`

Outputs:

- ROS camera calibration YAML extracted to `output.dir` (usually `camera.yaml`)
- Optional conversion to backend format at `output.backend_intrinsic_yaml`

Manual prerequisites (Ubuntu / ROS2):

```bash
sudo apt install ros-$ROS_DISTRO-v4l2-camera ros-$ROS_DISTRO-camera-calibration
```

Industrial camera note (Daheng):

- Daheng (gxipy/GenICam) cameras usually do **not** appear as `/dev/videoX`, so `v4l2_camera` cannot open them.
- Use `camera.backend: daheng` in the runner config; it starts `tools/calibrate/ros2_daheng_camera_node.py` to publish `/image_raw` for `cameracalibrator`.

Target types:

- Chessboard: set `calibrator.pattern: chessboard`
- Dots grid (symmetric): set `calibrator.pattern: circles`
- Dots grid (asymmetric): set `calibrator.pattern: acircles`
- For a 9x9 symmetric dots board where first-to-ninth center span is 6mm, use: `pattern: circles`, `size: 9x9`, `square: 0.00075`

Device selection (V4L2 cameras only):

- Prefer using a stable symlink: `/dev/v4l/by-id/...` instead of `/dev/video0`
- If you set `camera.video_device: auto`, the runner will:
  - pick the only available V4L2 device, or
  - error out with candidates if multiple exist (then set `camera.video_device` explicitly)

Convert an existing ROS `camera.yaml`/`ost.yaml` to backend format:

```bash
python -m tools.calibrate.ros_convert_intrinsic \
  --ros-yaml /path/to/camera.yaml \
  --output config/calibration/camera_intrinsic.yaml
```

## Verify current pixel -> world(mm) transform

For the current fixed camera + fixed arm setup, use the verification tool below to
click a known point on a calibration board and compare the measured world
coordinate against the expected physical coordinate.

```bash
python -m tools.calibrate.verify_transform \
  --config config/settings.dev.yaml \
  --intrinsic config/calibration/camera_intrinsic.yaml \
  --extrinsic config/calibration/extrinsic.yaml \
  --arm-pose 0,0 \
  --expected 0,0
```

Controls:

- Left click: select a known point on the board
- `SPACE`: save the current sample
- `e`: update expected world coordinate
- `a`: update arm pose
- `c`: clear current selection
- `q` / `ESC`: quit and write a YAML report

If you only measure the board origin, you are mostly checking translation error.
If you also want to verify `mm_per_pixel`, click several known board points with
different expected coordinates and compare the residuals.
