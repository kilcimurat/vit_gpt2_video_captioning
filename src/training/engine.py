from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerBase
from tqdm.auto import tqdm

from ..config import GenerationConfig, TrainingConfig
from ..models.video_captioning import VideoCaptioningModel


@dataclass
class EpochResult:
    loss: float
    perplexity: float


class Trainer:
    """Simple training loop for the video captioner."""

    def __init__(
        self,
        model: VideoCaptioningModel,
        tokenizer: PreTrainedTokenizerBase,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        cfg: TrainingConfig,
        generation_cfg: GenerationConfig,
        output_dir: Path,
    ) -> None:
        self.tokenizer = tokenizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = cfg
        self.generation_cfg = generation_cfg
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if cfg.gpu_ids and not torch.cuda.is_available():
            raise RuntimeError("GPU IDs configured but CUDA is not available")

        if cfg.gpu_ids:
            primary_gpu = cfg.gpu_ids[0]
            self.device = torch.device(f"cuda:{primary_gpu}")
            if len(cfg.gpu_ids) > 1:
                model = nn.DataParallel(model, device_ids=list(cfg.gpu_ids))
        else:
            self.device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

        self.model = model.to(self.device)

        self.optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
        )
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.device.type == "cuda")

    def _base_model(self) -> VideoCaptioningModel:
        if isinstance(self.model, nn.DataParallel):
            return self.model.module  # type: ignore[return-value]
        return self.model

    def _step(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        pixel_values = batch["pixel_values"].to(self.device)
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        labels = batch["labels"].to(self.device)

        with torch.cuda.amp.autocast(enabled=self.device.type == "cuda"):
            outputs = self.model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss / self.cfg.gradient_accumulation_steps
        return loss

    def _run_epoch(self, epoch: int) -> EpochResult:
        self.model.train()
        total_loss = 0.0
        progress = tqdm(
            enumerate(self.train_loader, start=1),
            total=len(self.train_loader),
            desc=f"Epoch {epoch} [train]",
            leave=False,
        )
        for step, batch in progress:
            loss = self._step(batch)
            self.scaler.scale(loss).backward()

            if step % self.cfg.gradient_accumulation_steps == 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.cfg.max_grad_norm
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)

            total_loss += loss.item()

            if step % self.cfg.log_every == 0:
                current_loss = total_loss / step
                progress.set_postfix({"loss": f"{current_loss:.4f}"}, refresh=False)

        remainder = len(self.train_loader) % self.cfg.gradient_accumulation_steps
        if remainder != 0:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.cfg.max_grad_norm
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)

        mean_loss = total_loss / max(len(self.train_loader), 1)
        perplexity = float(torch.exp(torch.tensor(mean_loss)))
        return EpochResult(loss=mean_loss, perplexity=perplexity)

    def _evaluate(self, epoch: int) -> Optional[EpochResult]:
        if self.val_loader is None:
            return None
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            progress = tqdm(
                self.val_loader,
                total=len(self.val_loader),
                desc=f"Epoch {epoch} [val]",
                leave=False,
            )
            for batch in progress:
                pixel_values = batch["pixel_values"].to(self.device)
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                outputs = self.model(
                    pixel_values=pixel_values,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                total_loss += outputs.loss.item()
                progress.set_postfix(
                    {"loss": f"{total_loss / max(progress.n, 1):.4f}"},
                    refresh=False,
                )

        mean_loss = total_loss / max(len(self.val_loader), 1)
        perplexity = float(torch.exp(torch.tensor(mean_loss)))
        return EpochResult(loss=mean_loss, perplexity=perplexity)

    def _save_checkpoint(self, epoch: int, metrics: Dict[str, float]) -> None:
        checkpoint_dir = self.output_dir / f"epoch_{epoch:03d}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self._base_model().state_dict(), checkpoint_dir / "model.pt")
        torch.save(self.optimizer.state_dict(), checkpoint_dir / "optimizer.pt")
        with open(checkpoint_dir / "metrics.json", "w", encoding="utf-8") as handle:
            json.dump(metrics, handle, indent=2)

    def _qualitative_sample(self, batch: Dict[str, torch.Tensor], epoch: int) -> None:
        self.model.eval()
        with torch.no_grad():
            pixel_values = batch["pixel_values"][:2].to(self.device)
            generation_kwargs = {
                "max_length": self.generation_cfg.max_length,
                "num_beams": self.generation_cfg.num_beams,
                "temperature": self.generation_cfg.temperature,
                "pad_token_id": self.tokenizer.pad_token_id,
                "bos_token_id": self.tokenizer.bos_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
            }
            generated = self._base_model().generate(pixel_values, generation_kwargs)
            captions = self.tokenizer.batch_decode(generated, skip_special_tokens=True)

        sample_file = self.output_dir / f"epoch_{epoch:03d}_samples.json"
        with open(sample_file, "w", encoding="utf-8") as handle:
            json.dump(captions, handle, indent=2)

    def fit(self) -> None:
        self.optimizer.zero_grad(set_to_none=True)
        for epoch in range(1, self.cfg.epochs + 1):
            train_metrics = self._run_epoch(epoch)
            val_metrics = self._evaluate(epoch)

            metrics = {
                "train_loss": train_metrics.loss,
                "train_perplexity": train_metrics.perplexity,
            }
            if val_metrics is not None:
                metrics.update(
                    {
                        "val_loss": val_metrics.loss,
                        "val_perplexity": val_metrics.perplexity,
                    }
                )

            print(f"Epoch {epoch} metrics: {metrics}")
            if epoch % self.cfg.save_every == 0:
                self._save_checkpoint(epoch, metrics)

            self._qualitative_sample(next(iter(self.train_loader)), epoch)
