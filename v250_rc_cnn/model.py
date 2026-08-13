"""Reverse-complement aware raw-sequence residual expert for Tiara2 v2.5.0."""
from __future__ import annotations

import torch
from torch import nn


HEADS = ("root", "euk", "prok", "organelle")


class ResidualTCNBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, 7, padding=3 * dilation,
                      dilation=dilation, groups=channels, bias=False),
            nn.GroupNorm(16, channels),
            nn.GELU(),
            nn.Conv1d(channels, channels, 1, bias=False),
            nn.GroupNorm(16, channels),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return x + self.net(x)


class RCShortResidualExpert(nn.Module):
    """Small multiscale CNN/TCN that emits residual hierarchical logits.

    Token convention is A=0, C=1, G=2, T=3 and padding/ambiguous=4.
    The same tower is used for forward and reverse-complement inputs.
    """

    def __init__(self, head_sizes, embed_dim=32, channels=192, dropout=0.15):
        super().__init__()
        if channels % 16:
            raise ValueError("channels must be divisible by 16")
        self.head_sizes = dict(head_sizes)
        self.embedding = nn.Embedding(5, embed_dim, padding_idx=4)
        self.stem = nn.Sequential(
            nn.Conv1d(embed_dim, 96, 7, stride=2, padding=3, bias=False),
            nn.GroupNorm(16, 96),
            nn.GELU(),
        )
        branch = channels // 3
        self.multiscale = nn.ModuleList([
            nn.Conv1d(96, branch, kernel, stride=2, padding=kernel // 2, bias=False)
            for kernel in (7, 15, 31)
        ])
        self.project = nn.Sequential(
            nn.GroupNorm(16, channels), nn.GELU(),
            nn.Conv1d(channels, channels, 1, bias=False),
        )
        self.tcn = nn.Sequential(*[
            ResidualTCNBlock(channels, dilation, dropout)
            for dilation in (1, 2, 4)
        ])
        self.pool = nn.Sequential(
            nn.Linear(channels * 2, 512), nn.GELU(), nn.Dropout(dropout)
        )
        self.heads = nn.ModuleDict({name: nn.Linear(512, int(size))
                                    for name, size in self.head_sizes.items()})
        for head in self.heads.values():
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    @staticmethod
    def reverse_complement(tokens, lengths):
        positions = torch.arange(tokens.shape[1], device=tokens.device)[None, :]
        valid = positions < lengths[:, None]
        source = (lengths[:, None] - 1 - positions).clamp_min(0)
        reversed_tokens = tokens.gather(1, source)
        complemented = torch.where(reversed_tokens < 4, 3 - reversed_tokens, reversed_tokens)
        return torch.where(valid, complemented, torch.full_like(complemented, 4))

    def _tower(self, tokens):
        x = self.embedding(tokens).transpose(1, 2)
        x = self.stem(x)
        x = torch.cat([layer(x) for layer in self.multiscale], dim=1)
        x = self.tcn(self.project(x))
        pooled = torch.cat([x.mean(dim=2), x.amax(dim=2)], dim=1)
        z = self.pool(pooled)
        return {name: head(z) for name, head in self.heads.items()}

    def forward(self, tokens, lengths, rc_average=True):
        forward = self._tower(tokens)
        if not rc_average:
            return forward
        reverse = self._tower(self.reverse_complement(tokens, lengths))
        return {name: 0.5 * (forward[name] + reverse[name]) for name in HEADS}
