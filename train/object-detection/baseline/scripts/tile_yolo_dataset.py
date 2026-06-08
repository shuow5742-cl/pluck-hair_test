#!/usr/bin/env python3
"""
Tile a YOLO-format dataset into fixed-size slices.

If SAHI exposes `slice_yolo`, the script delegates to it; otherwise, a manual
tiler is used. The resulting dataset is written to <dataset>/tiles_<size>/ by default.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

from PIL import Image

try:  # SAHI >=0.12
    from sahi.slicing import slice_yolo  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    slice_yolo = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tile a YOLO dataset into fixed-size patches.")
    parser.add_argument(
        "--dataset",
        required=True,
        help="Path to YOLO dataset root containing images/ and labels/ folders.",
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=640,
        help="Height/width of each square tile (pixels).",
    )
    parser.add_argument(
        "--overlap",
        type=float,
        default=0.2,
        help="Overlap ratio between adjacent tiles (0-1).",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val"],
        help="Dataset splits to slice (subfolders under images/ and labels/).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Destination directory for sliced dataset. Defaults to <dataset>/tiles_<tile-size>.",
    )
    parser.add_argument(
        "--keep-empty",
        action="store_true",
        help="Keep tiles without annotations (default: drop empty tiles).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce console output.",
    )
    return parser.parse_args()


def validate_dirs(root: Path) -> None:
    if not root.exists():
        raise FileNotFoundError(f"Dataset root not found: {root}")
    if not (root / "images").exists() or not (root / "labels").exists():
        raise FileNotFoundError(f"Dataset must contain 'images' and 'labels' under {root}")


def ensure_output_dir(dataset_root: Path, output_dir: str | None, tile_size: int) -> Path:
    if output_dir:
        out = Path(output_dir)
    else:
        out = dataset_root / f"tiles_{tile_size}"
    out.mkdir(parents=True, exist_ok=True)
    return out


def _load_yolo_labels(label_path: Path, width: int, height: int) -> List[Tuple[int, float, float, float, float]]:
    boxes: List[Tuple[int, float, float, float, float]] = []
    if not label_path.exists():
        return boxes
    with label_path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls = int(float(parts[0]))
            x, y, w, h = map(float, parts[1:])
            bw = w * width
            bh = h * height
            cx = x * width
            cy = y * height
            x1 = max(0.0, cx - bw / 2)
            y1 = max(0.0, cy - bh / 2)
            x2 = min(width, cx + bw / 2)
            y2 = min(height, cy + bh / 2)
            if x2 <= x1 or y2 <= y1:
                continue
            boxes.append((cls, x1, y1, x2, y2))
    return boxes


def _get_positions(length: int, tile_size: int, stride: int) -> List[int]:
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
    # Normalize starts to stay within valid range
    return sorted(set(max(0, min(s, length - tile_size)) for s in starts))


def _manual_slice_split(
    image_dir: Path,
    label_dir: Path,
    output_root: Path,
    split: str,
    tile_size: int,
    overlap: float,
    keep_empty: bool,
) -> Tuple[int, int]:
    image_output_dir = output_root / "images" / split
    label_output_dir = output_root / "labels" / split
    image_output_dir.mkdir(parents=True, exist_ok=True)
    label_output_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    skipped = 0
    stride = max(1, int(tile_size * (1 - overlap)))
    valid_ext = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

    image_paths = sorted(
        [p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in valid_ext]
    )
    for img_path in image_paths:
        with Image.open(img_path) as img:
            width, height = img.size
            boxes = _load_yolo_labels(label_dir / f"{img_path.stem}.txt", width, height)
            x_positions = _get_positions(width, tile_size, stride)
            y_positions = _get_positions(height, tile_size, stride)

            for y0 in y_positions:
                for x0 in x_positions:
                    x1 = min(x0 + tile_size, width)
                    y1 = min(y0 + tile_size, height)
                    tile_w = x1 - x0
                    tile_h = y1 - y0
                    if tile_w <= 0 or tile_h <= 0:
                        continue

                    tile_boxes: List[str] = []
                    for cls, bx1, by1, bx2, by2 in boxes:
                        ix1 = max(bx1, x0)
                        iy1 = max(by1, y0)
                        ix2 = min(bx2, x1)
                        iy2 = min(by2, y1)
                        if ix2 <= ix1 or iy2 <= iy1:
                            continue
                        bw = ix2 - ix1
                        bh = iy2 - iy1
                        if bw < 1 or bh < 1:
                            continue
                        cx = ((ix1 + ix2) / 2 - x0) / tile_w
                        cy = ((iy1 + iy2) / 2 - y0) / tile_h
                        nw = bw / tile_w
                        nh = bh / tile_h
                        tile_boxes.append(f"{cls} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

                    if not tile_boxes and not keep_empty:
                        skipped += 1
                        continue

                    tile_name = f"{img_path.stem}_x{x0}_y{y0}"
                    img_out_path = image_output_dir / f"{tile_name}{img_path.suffix}"
                    label_out_path = label_output_dir / f"{tile_name}.txt"
                    tile = img.crop((x0, y0, x1, y1))
                    tile.save(img_out_path)
                    label_out_path.write_text("\n".join(tile_boxes))
                    written += 1
    return written, skipped


def run_slice_for_split(
    dataset_root: Path,
    output_root: Path,
    split: str,
    tile_size: int,
    overlap: float,
    keep_empty: bool,
    quiet: bool,
) -> None:
    image_dir = dataset_root / "images" / split
    label_dir = dataset_root / "labels" / split
    if not image_dir.exists():
        print(f"[WARN] images/{split} not found, skipping.", file=sys.stderr)
        return
    if not label_dir.exists():
        print(f"[WARN] labels/{split} not found, skipping.", file=sys.stderr)
        return

    print(f"[INFO] Slicing split '{split}' -> {output_root}", flush=True)
    if slice_yolo is not None:
        slice_yolo(
            image_dir=str(image_dir),
            label_dir=str(label_dir),
            output_dir=str(output_root),
            image_output_dir=f"images/{split}",
            label_output_dir=f"labels/{split}",
            slice_height=tile_size,
            slice_width=tile_size,
            overlap_height_ratio=overlap,
            overlap_width_ratio=overlap,
            ignore_negative_samples=not keep_empty,
            verbose=not quiet,
        )
    else:
        written, skipped = _manual_slice_split(
            image_dir=image_dir,
            label_dir=label_dir,
            output_root=output_root,
            split=split,
            tile_size=tile_size,
            overlap=overlap,
            keep_empty=keep_empty,
        )
        print(
            f"[INFO] Split '{split}' tiled manually: saved {written} tiles, skipped {skipped} empty tiles."
        )


def main() -> int:
    args = parse_args()
    dataset_root = Path(args.dataset).expanduser().resolve()
    validate_dirs(dataset_root)
    output_root = ensure_output_dir(dataset_root, args.output_dir, args.tile_size)

    print(f"[INFO] Dataset root: {dataset_root}")
    print(f"[INFO] Output root:  {output_root}")
    print(f"[INFO] Tile size:    {args.tile_size}")
    print(f"[INFO] Overlap:      {args.overlap}")
    print(f"[INFO] Splits:       {', '.join(args.splits)}")
    print(f"[INFO] Using {'SAHI slice_yolo' if slice_yolo else 'manual tiler'} backend.")

    for split in args.splits:
        run_slice_for_split(
            dataset_root=dataset_root,
            output_root=output_root,
            split=split,
            tile_size=args.tile_size,
            overlap=args.overlap,
            keep_empty=args.keep_empty,
            quiet=args.quiet,
        )

    print("[INFO] Tiling complete.")
    print(f"[INFO] Update your dataset YAML to use path: {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# python scripts/tile_yolo_dataset.py \
#     --dataset dataset/pluck_11-18_subset150_yolo \
#     --tile-size 640 \
#     --overlap 0.2 \
#     --output-dir dataset/pluck_11-18_subset150_yolo/tiles_640
