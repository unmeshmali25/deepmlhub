from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch
import torch.nn as nn
from torch import Tensor

if TYPE_CHECKING:
    from src.models.gpt2.config import GPT2Config


class MultiHeadAttention(nn.Module):
    def __init__(
        self, n_embd: int, n_head: int, bias: bool = True, dropout: float = 0.0
    ):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.n_embd = n_embd
        self.head_dim = n_embd // n_head
        self.scale = self.head_dim**-0.5

        self.qkv_proj = nn.Linear(n_embd, 3 * n_embd, bias=bias)
        self.out_proj = nn.Linear(n_embd, n_embd, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        B, T, C = x.shape
        qkv = self.qkv_proj(x)
        qkv = qkv.view(B, T, 3, self.n_head, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        y = torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=mask,
            dropout_p=self.dropout.p if self.training else 0.0,
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        return self.out_proj(y)


class FeedForward(nn.Module):
    def __init__(
        self, n_embd: int, hidden_dim: int, bias: bool = True, dropout: float = 0.0
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, hidden_dim, bias=bias),
            nn.GELU(),
            nn.Linear(hidden_dim, n_embd, bias=bias),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        n_embd: int,
        n_head: int,
        hidden_dim: int,
        bias: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.attn = MultiHeadAttention(n_embd, n_head, bias, dropout)
        self.ffn = FeedForward(n_embd, hidden_dim, bias, dropout)
        self.ln1 = nn.LayerNorm(n_embd, bias=bias)
        self.ln2 = nn.LayerNorm(n_embd, bias=bias)

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        x = x + self.attn(self.ln1(x), mask=mask)
        x = x + self.ffn(self.ln2(x))
        return x


class GPT2Model(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        n_layer: int,
        n_head: int,
        n_embd: int,
        hidden_dim: int,
        dropout: float = 0.0,
        bias: bool = True,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.n_embd = n_embd

        self.transformer = nn.ModuleDict(
            {
                "wte": nn.Embedding(vocab_size, n_embd),
                "wpe": nn.Embedding(context_length, n_embd),
                "drop": nn.Dropout(dropout),
                "h": nn.ModuleList(
                    [
                        TransformerBlock(n_embd, n_head, hidden_dim, bias, dropout)
                        for _ in range(n_layer)
                    ]
                ),
                "ln_f": nn.LayerNorm(n_embd, bias=bias),
            }
        )
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: Tensor,
        targets: Optional[Tensor] = None,
    ) -> tuple[Tensor, Optional[Tensor]]:
        B, T = input_ids.shape
        assert T <= self.context_length, (
            f"Sequence length {T} exceeds context length {self.context_length}"
        )

        pos = torch.arange(0, T, dtype=torch.long, device=input_ids.device)
        pos_emb = self.transformer["wpe"](pos)
        tok_emb = self.transformer["wte"](input_ids)
        x = self.transformer["drop"](tok_emb + pos_emb)

        for block in self.transformer["h"]:
            x = block(x)

        x = self.transformer["ln_f"](x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=0,
            )

        return logits, loss

    @classmethod
    def from_config(cls, config: "GPT2Config") -> "GPT2Model":
        return cls(
            vocab_size=config.vocab_size,
            context_length=config.context_length,
            n_layer=config.n_layer,
            n_head=config.n_head,
            n_embd=config.n_embd,
            hidden_dim=config.hidden_dim,
            dropout=config.dropout,
            bias=config.bias,
        )
