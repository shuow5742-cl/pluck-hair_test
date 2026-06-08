# Detectron2 Framework Subproject

This directory hosts the Detectron2 runner invoked by the top-level orchestrator.

## Prerequisites

- NVIDIA driver + CUDA **12.1** toolkit (the supplied PyTorch wheels are `cu121`; building Detectron2 with a newer toolkit such as CUDA 13.0 triggers `RuntimeError: The detected CUDA version (13.0) mismatches the version that was used to compile PyTorch (12.1)`).
- [`uv`](https://github.com/astral-sh/uv) 0.4+ on your PATH.

## Bootstrap the virtualenv

1. `cd frameworks/detectron2`
2. Run the helper script (creates `.venv` if missing and installs the pinned CUDA wheels + Detectron2):
   ```bash
   scripts/bootstrap.sh
   ```
   - 该脚本会检测/设置 `CUDA_HOME`（默认 `/usr/local/cuda-12.1`）以及 `CC=/usr/bin/gcc-12` / `CXX=/usr/bin/g++-12`，然后执行 `uv sync --preview-features extra-build-dependencies`。
   - 如果你使用不同版本的 CUDA 或编译器，请在运行脚本前显式导出 `CUDA_HOME`、`CC`、`CXX`。
   - 若仍出现“detected CUDA version mismatches ...”或 “unsupported GNU version” 等错误，请确认系统已安装 CUDA 12.1 toolkit 与 gcc-12/g++-12，然后再次运行脚本。
3. Activate when needed: `source .venv/bin/activate`

## Prepare sample dataset (COCO8)

检测示例使用 `dataset/coco8`（COCO json + symlink 图像）。如果你只下载了 YOLO 版本的 COCO8，可运行：
```bash
python scripts/prepare_coco8_dataset.py \
  --source-root dataset/ultra/coco8 \
  --target-root dataset/coco8
```
脚本会将 YOLO 标签转换成 COCO JSON，并在 `dataset/coco8/images/{train,val}` 下创建指向原图的符号链接。随后即可在 Detectron2 实验 YAML 中引用 `dataset/coco8/annotations/instances_*.json`。

## Usage

Orchestrator example:
```bash
uv run python orchestrator/scripts/train.py --config configs/experiments/exp_orchestrator_detectron2_faster_rcnn.yaml
```

Standalone help:
```bash
uv run --project frameworks/detectron2 python scripts/train.py --help
```
