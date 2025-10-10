from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from transformers import GPT2Config, GPT2LMHeadModel

from ..config import ModelConfig

try:
    from torchvision.models import vit_b_16
    from torchvision.models.vision_transformer import ViT_B_16_Weights
except ImportError as exc:  # pragma: no cover
    raise ImportError("torchvision is required for the video encoder") from exc


class VideoEncoder(nn.Module):
    """Wraps a ViT backbone to encode sampled frames."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        # Use ImageNet-pretrained weights that expect 224x224 inputs to match the
        # default frame preprocessing in DatasetConfig.
        weights = ViT_B_16_Weights.IMAGENET1K_V1
        self.backbone = vit_b_16(weights=weights)
        self.hidden_size = self.backbone.hidden_dim
        self.backbone.heads = nn.Identity()
        if not cfg.train_encoder:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        batch, frames, channels, height, width = pixel_values.shape
        flattened = pixel_values.view(batch * frames, channels, height, width)
        # The ViT forward (with heads replaced by Identity) already returns the
        # CLS embeddings, ensuring positional encodings and class token handling
        # stay in sync with the pretrained configuration.
        cls_tokens = self.backbone(flattened)
        return cls_tokens.view(batch, frames, -1)


class VideoCaptioningModel(nn.Module):
    """Combines a ViT encoder with a GPT-2 decoder using cross-attention."""

    def __init__(
        self,
        cfg: ModelConfig,
        vocab_size: Optional[int] = None,
        tokenizer_pad_token_id: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.encoder = VideoEncoder(cfg)

        decoder_config = GPT2Config.from_pretrained(cfg.gpt2_checkpoint)
        decoder_config.add_cross_attention = True
        decoder_config.cross_attention_hidden_size = decoder_config.n_embd
        self.decoder = GPT2LMHeadModel.from_pretrained(
            cfg.gpt2_checkpoint, config=decoder_config
        )
        if vocab_size is not None:
            self.decoder.resize_token_embeddings(vocab_size)
        if not cfg.train_decoder:
            for param in self.decoder.parameters():
                param.requires_grad = False

        if tokenizer_pad_token_id is not None:
            self.decoder.config.pad_token_id = tokenizer_pad_token_id
            self.decoder.config.bos_token_id = tokenizer_pad_token_id
            self.decoder.config.eos_token_id = tokenizer_pad_token_id

        self.bridge = nn.Sequential(
            nn.LayerNorm(self.encoder.hidden_size),
            nn.Linear(self.encoder.hidden_size, self.decoder.config.n_embd),
            nn.Dropout(cfg.dropout),
        )

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        encoder_hidden = self.encoder(pixel_values)
        encoder_hidden = self.bridge(encoder_hidden)
        encoder_attention_mask = torch.ones(
            encoder_hidden.size()[:2], dtype=attention_mask.dtype, device=attention_mask.device
        )
        outputs = self.decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden,
            encoder_attention_mask=encoder_attention_mask,
            labels=labels,
            return_dict=True,
        )
        return outputs

    @torch.no_grad()
    def generate(
        self,
        pixel_values: torch.Tensor,
        generation_kwargs: Optional[dict] = None,
    ) -> torch.Tensor:
        generation_kwargs = generation_kwargs or {}
        encoder_hidden = self.encoder(pixel_values)
        encoder_hidden = self.bridge(encoder_hidden)
        encoder_attention_mask = torch.ones(
            encoder_hidden.size()[:2], device=pixel_values.device, dtype=torch.long
        )
        outputs = self.decoder.generate(
            encoder_hidden_states=encoder_hidden,
            encoder_attention_mask=encoder_attention_mask,
            **generation_kwargs,
        )
        return outputs
