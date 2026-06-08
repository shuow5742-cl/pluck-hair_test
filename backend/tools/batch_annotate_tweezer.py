#!/usr/bin/env python3
"""Batch-run the tweezer detector on a folder and save annotated images."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import AppConfig
from src.core.tweezer_detector import TweezerConfig, TweezerDetector


def _iter_images(root: Path) -> list[Path]:
    exts = {".bmp", ".png", ".jpg", ".jpeg"}
    return sorted(p for p in root.iterdir() if p.is_file() and p.suffix.lower() in exts)


def _clear_output(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for p in root.iterdir():
        if p.is_file():
            p.unlink()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="backend settings YAML")
    parser.add_argument("--input", required=True, help="input image directory")
    parser.add_argument("--output", required=True, help="annotated image directory")
    args = parser.parse_args()

    cfg = AppConfig.from_yaml(args.config)
    tw_cfg = TweezerConfig.from_dict(cfg.tweezer.params)
    tw_cfg.enabled = True
    tw_cfg.debug_dump_dir = None
    detector = TweezerDetector(tw_cfg)

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    _clear_output(output_dir)

    for path in _iter_images(input_dir):
        image = cv2.imread(str(path))
        if image is None:
            continue
        detector.reset()
        result = detector.detect(image)
        annotated = image.copy()
        detector.draw_overlay(annotated, result, None)
        state = (
            "OPEN" if result.is_open else "CLOSED" if result.is_open is not None else "UNKNOWN"
        )
        cv2.putText(
            annotated,
            f"{path.name} | {state} | conf={result.confidence:.2f}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 0, 0),
            5,
            cv2.LINE_AA,
        )
        cv2.putText(
            annotated,
            f"{path.name} | {state} | conf={result.confidence:.2f}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        if result.tip_xy is not None:
            x, y = result.tip_xy
            cv2.putText(
                annotated,
                f"target=({x:.1f}, {y:.1f})",
                (20, 78),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 0),
                5,
                cv2.LINE_AA,
            )
            cv2.putText(
                annotated,
                f"target=({x:.1f}, {y:.1f})",
                (20, 78),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 0, 255),
                2,
                cv2.LINE_AA,
            )
        out_path = output_dir / f"{path.stem}_det.png"
        cv2.imwrite(str(out_path), annotated)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
