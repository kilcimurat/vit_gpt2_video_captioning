from __future__ import annotations

import argparse
from pathlib import Path

from src.config import ProjectConfig
from src.data.preprocess import prepare_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare MSVD manifests")
    parser.add_argument(
        "--root",
        type=str,
        default=None,
        help="Optional dataset root (overrides default data directory)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for deterministic train/val/test split",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ProjectConfig()
    if args.root is not None:
        base = Path(args.root)
        config.dataset.root_dir = base
        config.dataset.raw_dir = base / "raw"
        config.dataset.processed_dir = base / "processed"
    config.ensure_directories()

    manifest_paths = prepare_dataset(config.dataset, seed=args.seed)
    print("Prepared manifests:")
    for split, path in manifest_paths.items():
        print(f"  {split}: {path}")


if __name__ == "__main__":
    main()
