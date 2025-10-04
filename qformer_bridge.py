import torch
from torch import nn
from typing import Optional


class QFormerLayer(nn.Module):
    """Single Q-Former layer with self- and cross-attention blocks."""

    def __init__(self, hidden_size: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size),
            nn.Dropout(dropout),
        )
        self.norm_self = nn.LayerNorm(hidden_size)
        self.norm_cross = nn.LayerNorm(hidden_size)
        self.norm_ffn = nn.LayerNorm(hidden_size)

    def forward(
        self,
        query_states: torch.Tensor,
        encoder_states: torch.Tensor,
        encoder_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Self-attention over query tokens
        residual = query_states
        q_states, _ = self.self_attn(query_states, query_states, query_states)
        q_states = self.norm_self(residual + q_states)

        # Cross-attention: queries attend to vision tokens
        residual = q_states
        cross_output, _ = self.cross_attn(
            q_states,
            encoder_states,
            encoder_states,
            key_padding_mask=encoder_padding_mask,
        )
        q_states = self.norm_cross(residual + cross_output)

        # Feed-forward block
        residual = q_states
        q_states = self.ffn(q_states)
        q_states = self.norm_ffn(residual + q_states)
        return q_states


class QFormerBridge(nn.Module):
    """Bridges ViT frame tokens to GPT-2 cross-attention space via learnable query tokens."""

    def __init__(
        self,
        hidden_size: int = 768,
        num_query_tokens: int = 16,
        num_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.1,
        vision_hidden_size: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.query_tokens = nn.Parameter(
            torch.randn(num_query_tokens, hidden_size) * 0.02
        )
        self.layers = nn.ModuleList(
            [QFormerLayer(hidden_size, num_heads, dropout) for _ in range(num_layers)]
        )
        if vision_hidden_size is not None and vision_hidden_size != hidden_size:
            self.vision_proj = nn.Linear(vision_hidden_size, hidden_size)
        else:
            self.vision_proj = nn.Linear(hidden_size, hidden_size)
        self.output_norm = nn.LayerNorm(hidden_size)
        self.summary = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(
        self,
        vision_embeddings: torch.Tensor,
        vision_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            vision_embeddings: Tensor of shape (batch, frames, dim).
            vision_padding_mask: Optional bool mask where True marks padded frames.
        Returns:
            Tensor of shape (batch, num_query_tokens + 1, hidden_size).
        """
        batch_size = vision_embeddings.size(0)
        dtype = vision_embeddings.dtype
        device = vision_embeddings.device

        vision_embeddings = self.vision_proj(vision_embeddings)

        query_tokens = self.query_tokens.unsqueeze(0).expand(batch_size, -1, -1).to(device=device, dtype=dtype)

        encoder_mask = vision_padding_mask
        if encoder_mask is not None and encoder_mask.dtype != torch.bool:
            encoder_mask = encoder_mask.to(torch.bool)

        hidden_states = query_tokens
        for layer in self.layers:
            hidden_states = layer(hidden_states, vision_embeddings, encoder_mask)

        hidden_states = self.output_norm(hidden_states)
        global_summary = vision_embeddings.mean(dim=1, keepdim=True)
        global_summary = self.summary(global_summary)
        return torch.cat([hidden_states, global_summary], dim=1)
