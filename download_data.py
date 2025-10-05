from __future__ import annotations

import argparse

from pathlib import Path

from src.config import ProjectConfig
from src.data.downloader import download_dataset
from src.data.metadata import ensure_captions_file
from src.data.preprocess import prepare_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download the MSVD dataset")
    parser.add_argument(
        "--root",
        type=str,
        default=None,
        help="Optional path to store dataset files",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ProjectConfig()
    if args.root is not None:
        config.dataset.root_dir = Path(args.root)
        config.dataset.raw_dir = Path(args.root) / "raw"
        config.dataset.processed_dir = Path(args.root) / "processed"
    config.ensure_directories()

    location = download_dataset(config.dataset)
    print(f"Dataset extracted to {location}")

    captions = ensure_captions_file(config.dataset)
    print(f"Captions available at {captions}")

    manifest_paths = prepare_dataset(config.dataset, seed=config.training.seed)
    print("Prepared manifests:")
    for split, path in manifest_paths.items():
        print(f"  {split}: {path}")


if __name__ == "__main__":
    main()
