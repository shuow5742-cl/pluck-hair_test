#!/usr/bin/env python3
"""Batch segmentation pipeline runner on a directory of PNGs.

Drives the full configured VisionPipeline (yolo_seg → abstain_near_metal →
…) on each PNG and writes a single annotated visualization that reflects
the *post-filter* detection list. Picks the picker output too (post-abstain
detections, their masks, picks).

Output per frame:
  <output-dir>/<frame_id>/annotated.png   ← drawn from final ctx.detections
  <output-dir>/<frame_id>/result.json     ← matching JSON
  <output-dir>/_all_annotated/<frame_id>.png   ← flat-folder copy for browsing
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Register all hub-side step types BEFORE building the pipeline.
import src.steps  # noqa: F401

from autoweaver.pipeline import VisionPipeline  # noqa: E402

from src.types import SegDetection  # noqa: E402
from src.core.pick_point_estimator import _detect_metal_plate, _dilate_binary  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--image-dir", required=True)
    p.add_argument("--output-dir", default="data/segmentation_outputs")
    p.add_argument("--model", default="assets/best_foreigh_segment_yolov8m_seg.pt")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.45)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", default="auto")
    p.add_argument("--safety-margin-px", type=float, default=78.0)
    p.add_argument("--limit", type=int, default=None)
    return p.parse_args()


def resolve(path_str: str) -> Path:
    p = Path(path_str).expanduser()
    return p if p.is_absolute() else (ROOT / p).resolve()


def build_pipeline(args: argparse.Namespace, output_dir: Path) -> VisionPipeline:
    return VisionPipeline.from_config({
        "pipeline": {
            "steps": [
                {
                    "name": "segment",
                    "type": "yolo_seg",
                    "params": {
                        "model": str(resolve(args.model)),
                        "conf": args.conf,
                        "iou": args.iou,
                        "imgsz": args.imgsz,
                        "device": args.device,
                        "output_dir": str(output_dir),
                        "save_artifacts": False,  # we render after filtering
                    },
                },
                {
                    "name": "abstain",
                    "type": "abstain_near_metal",
                    "params": {"safety_margin_px": args.safety_margin_px},
                },
            ]
        }
    })


def render_annotated(
    frame_bgr: np.ndarray,
    detections: list[SegDetection],
    metal_mask: np.ndarray | None = None,
    forbidden_mask: np.ndarray | None = None,
) -> np.ndarray:
    out = frame_bgr.copy()
    overlay = out.copy()

    # Mask polygons (orange).
    for det in detections:
        if det.polygon_xy:
            poly = np.asarray(det.polygon_xy, dtype=np.int32).reshape(-1, 1, 2)
            cv2.fillPoly(overlay, [poly], (0, 128, 255))
    out = cv2.addWeighted(overlay, 0.35, out, 0.65, 0)

    # Safety zone overlay BEFORE bbox/pick markers so they sit on top.
    if forbidden_mask is not None and forbidden_mask.any():
        safety_overlay = out.copy()
        # Translucent magenta over the safety buffer around metal.
        safety_overlay[forbidden_mask > 0] = (180, 60, 200)
        out = cv2.addWeighted(safety_overlay, 0.30, out, 0.70, 0)
    if metal_mask is not None and metal_mask.any():
        # Solid red for the metal plate itself.
        metal_overlay = out.copy()
        metal_overlay[metal_mask > 0] = (40, 40, 220)
        out = cv2.addWeighted(metal_overlay, 0.55, out, 0.45, 0)

    for det in detections:
        x1, y1, x2, y2 = [int(round(v)) for v in (det.bbox.x1, det.bbox.y1, det.bbox.x2, det.bbox.y2)]
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{det.object_type} {det.confidence:.2f} area={det.mask_area}"
        cv2.putText(out, label, (x1, max(0, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        _draw_pick(out, det)
    return out


def _draw_pick(out: np.ndarray, det: SegDetection) -> None:
    if not det.pick_point_xy:
        return
    px, py = [int(round(v)) for v in det.pick_point_xy]
    color = (0, 0, 255)
    cv2.drawMarker(out, (px, py), color, markerType=cv2.MARKER_CROSS, markerSize=18, thickness=2)
    cv2.circle(out, (px, py), 5, color, thickness=2)
    if det.pick_angle_deg is not None:
        theta = np.deg2rad(det.pick_angle_deg)
        dx = int(round(np.cos(theta) * 30))
        dy = int(round(np.sin(theta) * 30))
        cv2.line(out, (px - dx, py - dy), (px + dx, py + dy), (255, 0, 255), 2, cv2.LINE_AA)
    text = det.pick_method or "pick"
    if det.pick_score is not None:
        text += f" {det.pick_score:.2f}"
    if det.distance_to_metal_px is not None:
        text += f" metal={det.distance_to_metal_px:.0f}px"
    cv2.putText(out, text, (px + 8, max(0, py - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)


def det_to_dict(d: SegDetection) -> dict[str, Any]:
    base = {
        "detection_id": d.detection_id,
        "object_type": d.object_type,
        "confidence": d.confidence,
        "bbox_xyxy": [d.bbox.x1, d.bbox.y1, d.bbox.x2, d.bbox.y2],
        "center_xy": d.center_xy,
        "mask_area": d.mask_area,
        "polygon_xy": d.polygon_xy,
        "pick_point_xy": d.pick_point_xy,
        "pick_angle_deg": d.pick_angle_deg,
        "pick_method": d.pick_method,
        "pick_score": d.pick_score,
        "shape_class": d.shape_class,
        "extent": d.extent,
        "solidity": d.solidity,
        "object_aspect_ratio": d.object_aspect_ratio,
        "distance_to_metal_px": d.distance_to_metal_px,
        "distance_to_edge_px": d.distance_to_edge_px,
    }
    return base


def main() -> None:
    args = parse_args()
    image_dir = resolve(args.image_dir)
    output_dir = resolve(args.output_dir)

    pngs = sorted(image_dir.glob("*.png"))
    if args.limit:
        pngs = pngs[: args.limit]
    if not pngs:
        raise SystemExit(f"No PNGs in {image_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    flat_dir = output_dir / "_all_annotated"
    flat_dir.mkdir(parents=True, exist_ok=True)

    pipeline = build_pipeline(args, output_dir)

    class_counts: Counter[str] = Counter()
    method_counts: Counter[str] = Counter()
    total_dets_post_filter = 0
    zero_det_files = 0
    abstained_total = 0

    for i, png in enumerate(pngs, start=1):
        frame = cv2.imread(str(png))
        if frame is None:
            print(f"[{i:3d}/{len(pngs)}] SKIP {png.name}: cannot read")
            continue

        pipeline_result = pipeline.run(frame)
        dets: list[SegDetection] = [d for d in pipeline_result.detections if isinstance(d, SegDetection)]
        n = len(dets)
        abstained = pipeline_result.metadata.get("abstained_count", 0)

        total_dets_post_filter += n
        abstained_total += abstained
        if n == 0:
            zero_det_files += 1
        for d in dets:
            class_counts[d.shape_class or "None"] += 1
            method_counts[d.pick_method or "None"] += 1

        print(f"[{i:3d}/{len(pngs)}] {png.name}: {n} det (+{abstained} abstained)")

        # Recompute the same metal / forbidden masks the picker uses so the
        # visualization shows exactly what the abstain decision was against.
        # Pass the union of all post-filter YOLO masks as exclude_mask — pixels
        # YOLO claims as foreign matter are never plate.
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h_full, w_full = gray.shape[:2]
        yolo_union = np.zeros((h_full, w_full), dtype=np.uint8)
        for d in dets:
            if d.polygon_xy:
                poly = np.asarray(d.polygon_xy, dtype=np.int32).reshape(-1, 1, 2)
                cv2.fillPoly(yolo_union, [poly], 1)
        metal_mask = _detect_metal_plate(gray, exclude_mask=yolo_union)
        forbidden_mask = _dilate_binary(metal_mask, int(args.safety_margin_px))

        # Render annotated PNG from POST-FILTER detections + safety overlay.
        annotated = render_annotated(frame, dets, metal_mask=metal_mask, forbidden_mask=forbidden_mask)
        frame_dir = output_dir / png.stem
        frame_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(frame_dir / "annotated.png"), annotated)
        cv2.imwrite(str(frame_dir / "original.png"), frame)
        shutil.copy(frame_dir / "annotated.png", flat_dir / f"{png.stem}.png")

        result_payload = {
            "frame_id": png.stem,
            "image_width": int(frame.shape[1]),
            "image_height": int(frame.shape[0]),
            "abstained_count": abstained,
            "abstained_ids": pipeline_result.metadata.get("abstained_ids", []),
            "detections": [det_to_dict(d) for d in dets],
        }
        (frame_dir / "result.json").write_text(
            json.dumps(result_payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    print("\n=== Summary ===")
    print(f"  images:           {len(pngs)}")
    print(f"  post-filter dets: {total_dets_post_filter}")
    print(f"  abstained dets:   {abstained_total}")
    print(f"  zero-det files:   {zero_det_files}")
    print("\n  shape_class (post-filter):")
    for cls, n in class_counts.most_common():
        pct = 100.0 * n / max(total_dets_post_filter, 1)
        print(f"    {cls:14s} {n:4d}  ({pct:5.1f}%)")
    print("\n  pick_method (post-filter):")
    for m, n in method_counts.most_common():
        pct = 100.0 * n / max(total_dets_post_filter, 1)
        print(f"    {m:38s} {n:4d}  ({pct:5.1f}%)")


if __name__ == "__main__":
    main()
