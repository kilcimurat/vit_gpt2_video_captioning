from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from typing import Dict, List

from tqdm.auto import tqdm

from ..config import DatasetConfig
from .metadata import ensure_captions_file


def _collect_video_files(root: Path) -> Dict[str, Path]:
    video_map: Dict[str, Path] = {}
    search_patterns = ("*.mp4", "*.avi", "*.flv", "*.mov", "*.mpeg", "*.mpg")
    for extension in search_patterns:
        for path in tqdm(
            root.rglob(extension),
            desc=f"Scanning {extension} files",
            leave=False,
        ):
            video_map[path.stem.lower()] = path
    return video_map


def _resolve_captions_file(cfg: DatasetConfig, root: Path) -> Path:
    candidates = [
        root / "video_corpus.csv",
        root / "data" / "video_corpus.csv",
        root / "MSR Video Description Corpus.csv",
        cfg.raw_dir / cfg.captions_filename,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("Could not locate caption metadata in MSVD archive")


def _load_samples(annotation_file: Path, videos: Dict[str, Path]) -> List[dict]:
    samples: Dict[str, dict] = {}
    with open(annotation_file, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
        for row in tqdm(rows, desc="Parsing captions", leave=False):
            video_id = row.get("VideoID", "").strip().lower()
            youtube_id = row.get("YouTubeID", "").strip().lower()
            sentence = row.get("Sentence", "").strip()
            if not sentence:
                continue

            video_path = videos.get(video_id) or videos.get(youtube_id)
            if video_path is None:
                continue

            key = video_path.stem
            if key not in samples:
                samples[key] = {"video_path": str(video_path), "captions": []}
            samples[key]["captions"].append(sentence)
    return list(samples.values())


def _split_samples(samples: List[dict], seed: int) -> dict:
    rng = random.Random(seed)
    rng.shuffle(samples)
    total = len(samples)
    train_end = int(total * 0.8)
    val_end = train_end + int(total * 0.1)
    return {
        "train": samples[:train_end],
        "val": samples[train_end:val_end],
        "test": samples[val_end:],
    }


def prepare_dataset(cfg: DatasetConfig, seed: int = 42) -> dict:
    """Create train/val/test JSON manifests for the MSVD dataset."""

    archive_root = cfg.raw_dir / "YouTubeClips"
    if not archive_root.exists():
        raise FileNotFoundError(
            "YouTubeClips directory missing. Run download_data.py first."
        )

    ensure_captions_file(cfg)

    video_dir = archive_root
    captions_file = _resolve_captions_file(cfg, archive_root)

    videos = _collect_video_files(video_dir)
    samples = _load_samples(captions_file, videos)
    splits = _split_samples(samples, seed)

    manifest_dir = cfg.processed_dir / cfg.dataset_name
    manifest_dir.mkdir(parents=True, exist_ok=True)

    for split_name, split_samples in splits.items():
        output_path = manifest_dir / f"{split_name}.json"
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(split_samples, handle, indent=2)

    return {key: str(cfg.processed_dir / cfg.dataset_name / f"{key}.json") for key in splits}
