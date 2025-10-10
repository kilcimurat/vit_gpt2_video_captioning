from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerBase
from tqdm.auto import tqdm

from ..config import GenerationConfig, TrainingConfig
from ..models.video_captioning import VideoCaptioningModel

try:  # pragma: no cover - optional dependency for language metrics
    import evaluate
except ImportError:  # pragma: no cover - metrics remain disabled if unavailable
    evaluate = None


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

    def _build_generation_kwargs(self) -> Dict[str, Union[int, float]]:
        """Create generation kwargs shared across evaluation helpers."""

        kwargs: Dict[str, Union[int, float]] = {
            "max_length": self.generation_cfg.max_length,
            "num_beams": self.generation_cfg.num_beams,
            "temperature": self.generation_cfg.temperature,
        }
        if self.tokenizer.pad_token_id is not None:
            kwargs["pad_token_id"] = self.tokenizer.pad_token_id
        if self.tokenizer.bos_token_id is not None:
            kwargs["bos_token_id"] = self.tokenizer.bos_token_id
        if self.tokenizer.eos_token_id is not None:
            kwargs["eos_token_id"] = self.tokenizer.eos_token_id
        return kwargs

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
            loss = outputs.loss
            if isinstance(loss, torch.Tensor):
                loss = loss.mean()
            loss = loss / self.cfg.gradient_accumulation_steps
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
                loss = outputs.loss
                if isinstance(loss, torch.Tensor):
                    loss = loss.mean()
                total_loss += loss.item()
                progress.set_postfix(
                    {"loss": f"{total_loss / max(progress.n, 1):.4f}"},
                    refresh=False,
                )

        mean_loss = total_loss / max(len(self.val_loader), 1)
        perplexity = float(torch.exp(torch.tensor(mean_loss)))
        return EpochResult(loss=mean_loss, perplexity=perplexity)

    def _collect_predictions(
        self, epoch: int
    ) -> Tuple[List[str], List[List[str]], List[Optional[str]]]:
        if self.val_loader is None:
            return [], [], []
        dataset = getattr(self.val_loader, "dataset", None)
        if dataset is None or not hasattr(dataset, "samples"):
            return [], [], []

        model = self._base_model()
        model.eval()
        generation_kwargs = self._build_generation_kwargs()

        predictions: List[str] = []
        references: List[List[str]] = []
        video_paths: List[Optional[str]] = []

        sample_offset = 0
        with torch.no_grad():
            progress = tqdm(
                self.val_loader,
                total=len(self.val_loader),
                desc=f"Epoch {epoch} [gen]",
                leave=False,
            )
            for batch in progress:
                pixel_values = batch["pixel_values"].to(self.device)
                generated = model.generate(pixel_values, generation_kwargs)
                decoded = self.tokenizer.batch_decode(
                    generated, skip_special_tokens=True
                )
                batch_size = len(decoded)

                for idx_in_batch, pred in enumerate(decoded):
                    dataset_index = sample_offset + idx_in_batch
                    if dataset_index >= len(dataset.samples):
                        continue
                    sample = dataset.samples[dataset_index]
                    predictions.append(pred.strip())
                    references.append(
                        [caption.strip() for caption in sample.get("captions", [])]
                        or [""]
                    )
                    video_paths.append(sample.get("video_path"))

                sample_offset += batch_size

        return predictions, references, video_paths

    def _compute_language_metrics(
        self, predictions: List[str], references: List[List[str]]
    ) -> Dict[str, float]:
        if not predictions or not references or evaluate is None:
            return {}

        metrics: Dict[str, float] = {}

        try:
            bleu = evaluate.load("bleu")
            bleu_result = bleu.compute(
                predictions=predictions, references=references
            )
            precisions = bleu_result.get("precisions", [])
            for n, score in enumerate(precisions, start=1):
                metrics[f"bleu_{n}"] = float(score)
        except Exception as exc:  # pragma: no cover - metric backend optional
            print(f"[warn] BLEU metric unavailable: {exc}")

        try:
            cider = evaluate.load("cider")
            cider_result = cider.compute(
                predictions=predictions, references=references
            )
            metrics["cider"] = float(cider_result.get("cider", 0.0))
        except Exception as exc:  # pragma: no cover
            print(f"[warn] CIDEr metric unavailable: {exc}")

        single_reference = [refs[0] if refs else "" for refs in references]
        try:
            meteor = evaluate.load("meteor")
            meteor_result = meteor.compute(
                predictions=predictions, references=single_reference
            )
            metrics["meteor"] = float(meteor_result.get("meteor", 0.0))
        except Exception as exc:  # pragma: no cover
            print(f"[warn] METEOR metric unavailable: {exc}")

        try:
            rouge = evaluate.load("rouge")
            rouge_result = rouge.compute(
                predictions=predictions, references=single_reference
            )
            metrics["rouge_l"] = float(rouge_result.get("rougeL", 0.0))
        except Exception as exc:  # pragma: no cover
            print(f"[warn] ROUGE-L metric unavailable: {exc}")

        try:
            spice = evaluate.load("spice")
            spice_result = spice.compute(
                predictions=predictions, references=references
            )
            metrics["spice"] = float(spice_result.get("spice", 0.0))
        except Exception as exc:  # pragma: no cover
            print(f"[warn] SPICE metric unavailable: {exc}")

        return metrics

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
            generation_kwargs = self._build_generation_kwargs()
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

                if evaluate is not None:
                    predictions, references, _ = self._collect_predictions(epoch)
                    language_metrics = self._compute_language_metrics(
                        predictions, references
                    )
                    metrics.update(language_metrics)

            print(f"Epoch {epoch} metrics: {metrics}")
            if epoch % self.cfg.save_every == 0:
                self._save_checkpoint(epoch, metrics)

            self._qualitative_sample(next(iter(self.train_loader)), epoch)
