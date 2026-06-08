#!/usr/bin/env python3
"""Run YOLO instance segmentation on one image and save bbox/mask crops."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.core.instance_segmenter import YoloSegmentationRuntime


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Input image path")
    parser.add_argument(
        "--model",
        default="assets/best_foreigh_segment_yolov8m_seg.pt",
        help="YOLO segmentation model path, relative to backend/ or absolute",
    )
    parser.add_argument("--output-dir", default="data/segmentation_outputs")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def resolve_backend_path(path_str: str) -> Path:
    p = Path(path_str).expanduser()
    if p.is_absolute():
        return p
    return ROOT / p


def main() -> None:
    args = parse_args()
    image_path = resolve_backend_path(args.image)
    model_path = resolve_backend_path(args.model)
    output_dir = resolve_backend_path(args.output_dir)

    frame = cv2.imread(str(image_path))
    if frame is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    runtime = YoloSegmentationRuntime(
        model_path=model_path,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        device=args.device,
        output_dir=output_dir,
        save_artifacts=True,
    )
    result = runtime.process_frame(frame, frame_id=image_path.stem)
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
