from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List

import torch
from torch.utils.data import Dataset
from torchvision.io import read_video
from torchvision.transforms import functional as F
from transformers import PreTrainedTokenizerBase

from ..config import DatasetConfig

_IMAGE_MEAN = [0.485, 0.456, 0.406]
_IMAGE_STD = [0.229, 0.224, 0.225]


class VideoCaptionDataset(Dataset):
    """Loads MSVD clips and captions, sampling frames on the fly."""

    def __init__(
        self,
        manifest_path: Path,
        cfg: DatasetConfig,
        seed: int = 42,
    ) -> None:
        self.cfg = cfg
        self.manifest_path = manifest_path
        with open(manifest_path, encoding="utf-8") as handle:
            self.samples: List[Dict[str, Any]] = json.load(handle)
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.samples)

    def _sample_indices(self, total_frames: int) -> torch.Tensor:
        if total_frames <= 0:
            return torch.zeros(self.cfg.num_frames, dtype=torch.long)
        indices = torch.linspace(0, max(total_frames - 1, 0), steps=self.cfg.num_frames)
        return indices.long().clamp(max=max(total_frames - 1, 0))

    def _load_frames(self, video_path: str) -> torch.Tensor:
        video, _, _ = read_video(video_path, pts_unit="sec")
        if video.numel() == 0:
            frames = torch.zeros(self.cfg.num_frames, 3, *self.cfg.frame_shape)
            return frames
        frames = video.float() / 255.0
        indices = self._sample_indices(frames.shape[0])
        selected = frames[indices]
        selected = selected.permute(0, 3, 1, 2)  # T, C, H, W
        resized = torch.stack(
            [F.resize(frame, self.cfg.frame_shape, antialias=True) for frame in selected]
        )
        normalized = F.normalize(resized, _IMAGE_MEAN, _IMAGE_STD)
        return normalized

    def __getitem__(self, index: int) -> Dict[str, Any]:
        sample = self.samples[index]
        caption = self.rng.choice(sample["captions"])
        frames = self._load_frames(sample["video_path"])
        return {"frames": frames, "caption": caption, "video_path": sample["video_path"]}


class VideoCaptionCollator:
    """Tokenizes captions and batches tensors for training."""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        max_length: int,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        frames = torch.stack([item["frames"] for item in batch])
        captions = [item["caption"] for item in batch]
        tokenized = self.tokenizer(
            captions,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        labels = tokenized["input_ids"].clone()
        return {
            "pixel_values": frames,
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "labels": labels,
        }
