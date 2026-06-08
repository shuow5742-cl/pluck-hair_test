#!/usr/bin/env python3
"""
Run tiled YOLO inference on large images and emit AnyLabeling-compatible JSON files.

For each image, the script:
1. splits the image into overlapping tiles,
2. runs Ultralytics YOLO inference per tile,
3. merges detections back to the original resolution, and
4. writes `<image_stem>.json` following AnyLabeling 3.x schema.

By default the JSON lives alongside the source image so XAnyLabeling can pick it up directly.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from PIL import Image

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - optional dependency
    def tqdm(iterable, **kwargs):  # type: ignore
        return iterable

try:
    from ultralytics import YOLO
except Exception as exc:  # pragma: no cover - runtime guard
    raise SystemExit(
        "Ultralytics is required for inference. Install it via 'pip install ultralytics'."
    ) from exc


Image.MAX_IMAGE_PIXELS = None  # allow huge hair/debris frames


@dataclass
class Detection:
    bbox: Sequence[float]  # (x1, y1, x2, y2) absolute pixels
    score: float
    cls: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict detections on large images and export AnyLabeling JSON files."
    )
    parser.add_argument("--model", required=True, help="Path to YOLO model checkpoint (.pt/.onnx).")
    parser.add_argument(
        "--images", required=True, help="Directory containing the raw high-resolution images."
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Optional directory for JSON files. "
            "If omitted, JSON is saved next to each source image."
        ),
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=640,
        help="Square tile size passed to the model (default: 640).",
    )
    parser.add_argument(
        "--overlap",
        type=float,
        default=0.2,
        help="Overlap ratio between tiles (0.0-0.9, default: 0.2).",
    )
    parser.add_argument(
        "--conf-thres",
        type=float,
        default=0.1,
        help="Confidence threshold for Ultralytics predict() (default: 0.1).",
    )
    parser.add_argument(
        "--iou-thres",
        type=float,
        default=0.45,
        help="Tile-level NMS IoU threshold inside Ultralytics (default: 0.45).",
    )
    parser.add_argument(
        "--merge-iou",
        type=float,
        default=0.6,
        help="IoU threshold for merging overlapping boxes across tiles (default: 0.6).",
    )
    parser.add_argument(
        "--max-det",
        type=int,
        default=1000,
        help="Maximum detections to keep per image after merging (default: 1000).",
    )
    parser.add_argument(
        "--device",
        default="",
        help="Device string for Ultralytics (e.g. '0' or 'cpu'). Leave empty for auto.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Scan image directory recursively instead of only the top level.",
    )
    parser.add_argument(
        "--extensions",
        nargs="+",
        default=[".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"],
        help="Valid image extensions (case-insensitive).",
    )
    parser.add_argument(
        "--class-map",
        type=str,
        default=None,
        help="Optional JSON file describing class names. Accepts list or {id: name} dict.",
    )
    parser.add_argument(
        "--existing-label-dir",
        type=str,
        default=None,
        help=(
            "Directory holding existing AnyLabeling JSON files. "
            "Defaults to the output location (next to images unless --output-dir is set)."
        ),
    )
    parser.add_argument(
        "--existing-iou-thres",
        type=float,
        default=0.6,
        help="IoU threshold to consider a prediction already covered by an existing label (default: 0.6).",
    )
    parser.add_argument(
        "--incremental-prefix",
        type=str,
        default="predict_",
        help="Prefix to apply to new incremental labels (default: predict_).",
    )
    parser.add_argument(
        "--backup",
        action="store_true",
        help="If set, create a .bak alongside each JSON before overwriting.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run inference and report counts but do not write JSON files.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip images that already have a JSON file at the destination.",
    )
    parser.add_argument(
        "--index-range",
        type=str,
        default=None,
        help=(
            "Optional 1-based inclusive index range to limit images, e.g. '101-200' or '150'. "
            "Indexes are applied after sorting."
        ),
    )
    return parser.parse_args()


def load_class_names(model: YOLO, class_map_path: str | None) -> List[str]:
    if not class_map_path:
        names = model.names
        if isinstance(names, dict):
            return [names[k] for k in sorted(names.keys())]
        return list(names)

    mapping_file = Path(class_map_path)
    with mapping_file.open("r", encoding="utf-8") as f:
        mapping = json.load(f)

    if isinstance(mapping, list):
        return mapping
    if isinstance(mapping, dict):
        max_idx = max(int(k) for k in mapping.keys())
        names = [""] * (max_idx + 1)
        for key, value in mapping.items():
            names[int(key)] = str(value)
        return names
    raise ValueError(f"Unsupported class map format in {mapping_file}")


def _grid_positions(length: int, tile_size: int, stride: int) -> List[int]:
    if length <= tile_size:
        return [0]
    starts = [0]
    pos = 0
    while pos + tile_size < length:
        pos += stride
        if pos + tile_size >= length:
            starts.append(max(length - tile_size, 0))
            break
        starts.append(pos)
    return sorted(set(max(0, min(s, length - tile_size)) for s in starts))


def generate_tiles(width: int, height: int, tile_size: int, overlap: float) -> Iterable[tuple[int, int, int, int]]:
    overlap = max(0.0, min(overlap, 0.95))
    stride = max(1, int(round(tile_size * (1 - overlap))))
    x_positions = _grid_positions(width, tile_size, stride)
    y_positions = _grid_positions(height, tile_size, stride)
    for y0 in y_positions:
        for x0 in x_positions:
            x1 = min(x0 + tile_size, width)
            y1 = min(y0 + tile_size, height)
            yield x0, y0, x1, y1


def run_model_on_tile(
    model: YOLO,
    tile: Image.Image,
    conf_thres: float,
    iou_thres: float,
    device: str,
) -> List[Detection]:
    """Run YOLO on a tile and return detections in tile coordinates."""
    preds = model.predict(
        tile, conf=conf_thres, iou=iou_thres, device=device, verbose=False
    )
    detections: List[Detection] = []
    for result in preds:
        if result.boxes is None or result.boxes.shape[0] == 0:
            continue
        boxes = result.boxes
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        classes = boxes.cls.cpu().numpy().astype(int)
        for idx in range(xyxy.shape[0]):
            detections.append(
                Detection(
                    bbox=(float(xyxy[idx, 0]), float(xyxy[idx, 1]), float(xyxy[idx, 2]), float(xyxy[idx, 3])),
                    score=float(confs[idx]),
                    cls=int(classes[idx]),
                )
            )
    return detections


def py_nms(boxes: np.ndarray, scores: np.ndarray, iou_thres: float) -> List[int]:
    """Lightweight NMS implementation (x1,y1,x2,y2)."""
    if boxes.size == 0:
        return []

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep: List[int] = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        denom = areas[i] + areas[order[1:]] - inter
        iou = np.where(denom > 0, inter / denom, 0)
        inds = np.where(iou <= iou_thres)[0]
        order = order[inds + 1]
    return keep


def merge_detections(
    detections: List[Detection],
    merge_iou: float,
    max_det: int,
) -> List[Detection]:
    if not detections:
        return []

    merged: List[Detection] = []
    by_class: Dict[int, List[Detection]] = {}
    for det in detections:
        by_class.setdefault(det.cls, []).append(det)

    for cls_id, dets in by_class.items():
        boxes = np.array([det.bbox for det in dets], dtype=np.float32)
        scores = np.array([det.score for det in dets], dtype=np.float32)
        keep_idx = py_nms(boxes, scores, merge_iou)
        for idx in keep_idx:
            merged.append(dets[idx])

    merged.sort(key=lambda d: d.score, reverse=True)
    if len(merged) > max_det:
        merged = merged[:max_det]
    return merged


def detections_to_shapes(
    detections: List[Detection],
    class_names: Sequence[str],
) -> List[Dict[str, object]]:
    shapes: List[Dict[str, object]] = []
    for det in detections:
        x1, y1, x2, y2 = det.bbox
        cls_id = det.cls
        label = class_names[cls_id] if 0 <= cls_id < len(class_names) else str(cls_id)
        shapes.append(
            {
                "label": label,
                "score": float(det.score),
                "points": [
                    [float(x1), float(y1)],
                    [float(x2), float(y1)],
                    [float(x2), float(y2)],
                    [float(x1), float(y2)],
                ],
                "group_id": None,
                "description": "",
                "difficult": False,
                "shape_type": "rectangle",
                "flags": {},
                "attributes": {},
                "kie_linking": [],
            }
        )
    return shapes


def load_existing_shapes(path: Path) -> List[Dict[str, object]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload.get("shapes", [])


def _shape_to_rect(shape: Dict[str, object]) -> Tuple[float, float, float, float] | None:
    points = shape.get("points")
    if not isinstance(points, list) or len(points) < 2:
        return None
    try:
        xs = [float(pt[0]) for pt in points]
        ys = [float(pt[1]) for pt in points]
    except Exception:
        return None
    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def iou(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    if inter_area == 0.0:
        return 0.0
    a_area = (ax2 - ax1) * (ay2 - ay1)
    b_area = (bx2 - bx1) * (by2 - by1)
    denom = a_area + b_area - inter_area
    return inter_area / denom if denom > 0 else 0.0


def filter_incremental_detections(
    detections: List[Detection],
    class_names: Sequence[str],
    existing_shapes: List[Dict[str, object]],
    iou_thres: float,
    label_prefix: str,
) -> Tuple[List[Dict[str, object]], int]:
    """Return new shapes (prefixed labels) and count of filtered boxes."""
    existing_rects = []
    for sh in existing_shapes:
        rect = _shape_to_rect(sh)
        if rect:
            existing_rects.append(rect)

    kept_shapes: List[Dict[str, object]] = []
    filtered = 0

    for det in detections:
        label = class_names[det.cls] if 0 <= det.cls < len(class_names) else str(det.cls)
        rect = det.bbox
        max_iou = max((iou(rect, r) for r in existing_rects), default=0.0)
        if max_iou >= iou_thres:
            filtered += 1
            continue

        x1, y1, x2, y2 = rect
        kept_shapes.append(
            {
                "label": f"{label_prefix}{label}",
                "score": float(det.score),
                "points": [
                    [float(x1), float(y1)],
                    [float(x2), float(y1)],
                    [float(x2), float(y2)],
                    [float(x1), float(y2)],
                ],
                "group_id": None,
                "description": "",
                "difficult": False,
                "shape_type": "rectangle",
                "flags": {},
                "attributes": {},
                "kie_linking": [],
            }
        )

    return kept_shapes, filtered


def collect_images(images_dir: Path, recursive: bool, extensions: Sequence[str]) -> List[Path]:
    valid_ext = {ext.lower() for ext in extensions}
    if recursive:
        paths = [
            p
            for p in images_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in valid_ext
        ]
    else:
        paths = [
            p
            for p in images_dir.iterdir()
            if p.is_file() and p.suffix.lower() in valid_ext
        ]
    return sorted(paths)


def apply_index_range(image_paths: List[Path], range_str: str) -> List[Path]:
    """Select a 1-based inclusive slice of image paths."""
    raw = range_str.strip()
    if not raw:
        raise SystemExit("Empty --index-range value.")

    if "-" in raw:
        start_str, end_str = raw.split("-", 1)
    else:
        start_str, end_str = raw, raw

    try:
        start = int(start_str)
        end = int(end_str)
    except ValueError as exc:
        raise SystemExit(f"Invalid --index-range '{range_str}', expected integers like '101-200'.") from exc

    if start <= 0 or end <= 0:
        raise SystemExit(f"--index-range must be positive (1-based): '{range_str}'.")
    if start > end:
        raise SystemExit(f"--index-range start must be <= end: '{range_str}'.")

    total = len(image_paths)
    if total == 0:
        return []

    start_idx = start - 1  # convert to 0-based
    end_idx = min(end, total)  # inclusive end in 1-based -> exclusive in slice
    if start_idx >= total:
        return []

    return image_paths[start_idx:end_idx]


def save_anylabel_json(
    dest: Path,
    image_path: Path,
    width: int,
    height: int,
    shapes: List[Dict[str, object]],
) -> None:
    payload = {
        "version": "3.3.1",
        "flags": {},
        "shapes": shapes,
        "imagePath": image_path.name,
        "imageData": None,
        "imageHeight": height,
        "imageWidth": width,
    }
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def main() -> int:
    args = parse_args()
    images_dir = Path(args.images).expanduser().resolve()
    if not images_dir.exists():
        raise SystemExit(f"Image directory not found: {images_dir}")

    output_root = Path(args.output_dir).expanduser().resolve() if args.output_dir else None
    existing_root = Path(args.existing_label_dir).expanduser().resolve() if args.existing_label_dir else None

    print(f"[INFO] Loading model: {args.model}")
    model = YOLO(args.model)
    class_names = load_class_names(model, args.class_map)
    print(f"[INFO] Class names: {class_names}")

    image_paths = collect_images(images_dir, args.recursive, args.extensions)
    if not image_paths:
        print(f"[WARN] No images found in {images_dir}")
        return 0

    if args.index_range:
        image_paths = apply_index_range(image_paths, args.index_range)
        if not image_paths:
            print(f"[WARN] --index-range {args.index_range!r} produced no images in list.")
            return 0
        print(f"[INFO] Applying index range {args.index_range} (1-based). Using {len(image_paths)} images.")

    print(f"[INFO] Found {len(image_paths)} images. Starting inference...")

    for img_path in tqdm(image_paths, desc="Predicting"):
        target_json = (
            (output_root / (img_path.stem + ".json"))
            if output_root is not None
            else img_path.with_suffix(".json")
        )
        existing_json = (
            (existing_root / (img_path.stem + ".json"))
            if existing_root is not None
            else target_json
        )
        if args.skip_existing and target_json.exists():
            continue

        with Image.open(img_path) as img:
            img = img.convert("RGB")
            width, height = img.size
            tiles = list(generate_tiles(width, height, args.tile_size, args.overlap))

            detections: List[Detection] = []
            for x0, y0, x1, y1 in tiles:
                tile_img = img.crop((x0, y0, x1, y1))
                tile_dets = run_model_on_tile(
                    model=model,
                    tile=tile_img,
                    conf_thres=args.conf_thres,
                    iou_thres=args.iou_thres,
                    device=args.device,
                )
                for det in tile_dets:
                    gx1 = max(0.0, min(det.bbox[0] + x0, width))
                    gy1 = max(0.0, min(det.bbox[1] + y0, height))
                    gx2 = max(0.0, min(det.bbox[2] + x0, width))
                    gy2 = max(0.0, min(det.bbox[3] + y0, height))
                    if gx2 <= gx1 or gy2 <= gy1:
                        continue
                    detections.append(
                        Detection(
                            bbox=(gx1, gy1, gx2, gy2),
                            score=det.score,
                            cls=det.cls,
                        )
                    )

        merged = merge_detections(detections, merge_iou=args.merge_iou, max_det=args.max_det)
        existing_shapes = load_existing_shapes(existing_json)
        incremental_shapes, filtered = filter_incremental_detections(
            merged,
            class_names=class_names,
            existing_shapes=existing_shapes,
            iou_thres=args.existing_iou_thres,
            label_prefix=args.incremental_prefix,
        )
        combined_shapes = existing_shapes + incremental_shapes

        print(
            f"[INFO] {img_path.name}: existing={len(existing_shapes)}, "
            f"new_kept={len(incremental_shapes)}, filtered={filtered}"
        )

        if args.dry_run:
            continue

        if args.backup and target_json.exists():
            bak = target_json.with_suffix(target_json.suffix + ".bak")
            if not bak.exists():
                bak.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(target_json, bak)

        save_anylabel_json(target_json, img_path, width, height, combined_shapes)

    print("[INFO] Completed inference. JSON files ready for XAnyLabeling.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# python scripts/predict_xanylabeling.py \
#   --model experiments/pluck/full/yolo8l/yolo_run3/weights/best.pt \
#   --images /media/xinyuan/新加卷1/project/pluck/data/11-18 \
#   --tile-size 640 \
#   --overlap 0.2 \
#   --index-range 127-150 \
#   --conf-thres 0.1 \
#   --merge-iou 0.6

# python scripts/predict_xanylabeling.py \
#   --model experiments/pluck/full/yolov8s/yolo_run5/weights/best.pt \
#   --images /media/xinyuan/新加卷1/project/pluck/data/11-18 \
#   --index-range 1-150 \
#   --existing-iou-thres 0.2 \
#   --incremental-prefix predict_debris \
#   --conf-thres 0.2 \
#   --backup