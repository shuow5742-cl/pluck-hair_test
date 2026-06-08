from __future__ import annotations

import argparse

from src.engine.trainer import Trainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train detector via unified trainer.")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to experiment YAML config.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    trainer = Trainer(args.config)
    trainer.train()


if __name__ == "__main__":
    main()
