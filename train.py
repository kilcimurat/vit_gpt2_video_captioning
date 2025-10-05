from __future__ import annotations

import argparse
from pathlib import Path

from torch.utils.data import DataLoader
from transformers import GPT2TokenizerFast

from src.config import ProjectConfig
from src.data.dataset import VideoCaptionCollator, VideoCaptionDataset
from src.models.video_captioning import VideoCaptioningModel
from src.training.engine import Trainer


def build_arg_parser(description: str = "Train the video captioning model") -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--data-root", type=str, default=None, help="Dataset root directory")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Where to store checkpoints")
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Mini-batch size")
    parser.add_argument(
        "--learning-rate", type=float, default=None, help="Optimizer learning rate"
    )
    parser.add_argument("--device", type=str, default=None, help="Execution device (cuda or cpu)")
    parser.add_argument(
        "--num-workers", type=int, default=None, help="DataLoader worker processes"
    )
    parser.add_argument(
        "--gradient-accumulation",
        type=int,
        default=None,
        help="Steps to accumulate gradients",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=None,
        help="Number of frames sampled per video",
    )
    parser.add_argument(
        "--gpu-ids",
        type=str,
        default=None,
        help="Comma-separated GPU device indices for DataParallel",
    )
    return parser


def parse_args() -> argparse.Namespace:
    parser = build_arg_parser()
    parser.set_defaults(gpu_ids="0,2")
    return parser.parse_args()


def configure_from_args(args: argparse.Namespace) -> ProjectConfig:
    config = ProjectConfig()
    if args.data_root is not None:
        base = Path(args.data_root)
        config.dataset.root_dir = base
        config.dataset.raw_dir = base / "raw"
        config.dataset.processed_dir = base / "processed"
    if args.output_dir is not None:
        config.training.output_dir = Path(args.output_dir)
    if args.epochs is not None:
        config.training.epochs = args.epochs
    if args.batch_size is not None:
        config.training.batch_size = args.batch_size
    if args.learning_rate is not None:
        config.training.learning_rate = args.learning_rate
    if args.device is not None:
        config.training.device = args.device
    if args.num_workers is not None:
        config.training.num_workers = args.num_workers
    if args.gradient_accumulation is not None:
        config.training.gradient_accumulation_steps = args.gradient_accumulation
    if args.num_frames is not None:
        config.dataset.num_frames = args.num_frames
    if args.gpu_ids is not None:
        gpu_ids = tuple(int(part.strip()) for part in args.gpu_ids.split(",") if part.strip())
        config.training.gpu_ids = gpu_ids
        if gpu_ids:
            config.training.device = f"cuda:{gpu_ids[0]}"
    config.ensure_directories()
    return config


def run_training(config: ProjectConfig) -> None:
    manifest_dir = config.dataset.processed_dir / config.dataset.dataset_name
    train_manifest = manifest_dir / "train.json"
    val_manifest = manifest_dir / "val.json"
    if not train_manifest.exists() or not val_manifest.exists():
        raise FileNotFoundError(
            "Prepared manifests missing. Run prepare_data.py before training."
        )

    tokenizer = GPT2TokenizerFast.from_pretrained(config.model.gpt2_checkpoint)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
    tokenizer.padding_side = "right"

    train_dataset = VideoCaptionDataset(train_manifest, cfg=config.dataset, seed=config.training.seed)
    val_dataset = VideoCaptionDataset(val_manifest, cfg=config.dataset, seed=config.training.seed)
    collator = VideoCaptionCollator(tokenizer, config.dataset.max_caption_tokens)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.training.num_workers,
        pin_memory=True,
        collate_fn=collator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.training.num_workers,
        pin_memory=True,
        collate_fn=collator,
    )

    model = VideoCaptioningModel(
        cfg=config.model,
        vocab_size=len(tokenizer),
        tokenizer_pad_token_id=tokenizer.pad_token_id,
    )

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        train_loader=train_loader,
        val_loader=val_loader,
        cfg=config.training,
        generation_cfg=config.generation,
        output_dir=Path(config.training.output_dir),
    )
    trainer.fit()


def main() -> None:
    args = parse_args()
    config = configure_from_args(args)
    run_training(config)


if __name__ == "__main__":
    main()
