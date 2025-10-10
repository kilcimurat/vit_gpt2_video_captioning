from __future__ import annotations

import argparse

from src.data.downloader import download_dataset
from src.data.metadata import ensure_captions_file
from src.data.preprocess import prepare_dataset
from train import build_arg_parser, configure_from_args, run_training


def parse_args() -> argparse.Namespace:
    parser = build_arg_parser("Download, prepare, and train the model")
    parser.set_defaults(
        epochs=20,
        batch_size=16,
        learning_rate=3e-5,
        device="cuda",
        num_workers=8,
        gradient_accumulation=1,
        num_frames=16,
        gpu_ids="0,2",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = configure_from_args(args)

    if config.training.gpu_ids:
        print(f"Using GPU(s): {', '.join(map(str, config.training.gpu_ids))}")
    else:
        print(f"Using device: {config.training.device}")

    dataset_location = download_dataset(config.dataset)
    print(f"Dataset ready at {dataset_location}")

    captions_path = ensure_captions_file(config.dataset)
    print(f"Captions ready at {captions_path}")

    manifests = prepare_dataset(config.dataset, seed=config.training.seed)
    print("Prepared manifests:")
    for split, path in manifests.items():
        print(f"  {split}: {path}")

    run_training(config)


if __name__ == "__main__":
    main()
