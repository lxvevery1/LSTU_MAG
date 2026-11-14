#!/usr/bin/env python3
"""Text + numeric fusion model for WSB direction prediction."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable

import torch
import torch.nn as nn
from transformers import AutoModel

from common import CLASSES


class DirectionFusionModel(nn.Module):
    def __init__(
        self,
        encoder_path: str,
        numeric_dim: int,
        n_tickers: int,
        n_flairs: int,
        horizons: list[int],
        hidden_dim: int = 128,
        ticker_dim: int = 16,
        flair_dim: int = 8,
        dropout: float = 0.2,
        freeze_layers: int = 8,
        use_aux_class: bool = True,
        use_magnitude: bool = False,
    ):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(encoder_path)
        self.text_dim = int(self.encoder.config.hidden_size)
        self.horizons = list(horizons)
        self.use_aux_class = use_aux_class
        self.use_magnitude = use_magnitude
        self.freeze_layers = int(freeze_layers)
        self.n_layers = int(getattr(self.encoder.config, "num_hidden_layers", 12))

        if self.freeze_layers > 0:
            for p in self.encoder.embeddings.parameters():
                p.requires_grad = False
        for i, layer in enumerate(self.encoder.encoder.layer):
            if i < self.freeze_layers:
                for p in layer.parameters():
                    p.requires_grad = False

        self.ticker_embedding = nn.Embedding(n_tickers, ticker_dim)
        self.flair_embedding = nn.Embedding(n_flairs, flair_dim)
        fusion_dim = self.text_dim + numeric_dim + ticker_dim + flair_dim
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.heads = nn.ModuleDict({f"h{h}": nn.Linear(hidden_dim, 3) for h in self.horizons})
        self.aux_class = nn.Linear(hidden_dim, len(CLASSES)) if use_aux_class else None
        self.magnitude = nn.Linear(hidden_dim, 1) if use_magnitude else None

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_layers >= self.n_layers:
            # A fully frozen encoder must stay in eval mode so dropout does not
            # perturb the cached representation distribution.
            self.encoder.eval()
        return self

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        numeric: torch.Tensor,
        ticker_id: torch.Tensor,
        flair_id: torch.Tensor,
    ) -> dict:
        enc = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        text = enc.last_hidden_state[:, 0, :]
        tick = self.ticker_embedding(ticker_id)
        flair = self.flair_embedding(flair_id)
        z = self.fusion(torch.cat([text, numeric, tick, flair], dim=-1))
        out = {"logits": {f"h{h}": self.heads[f"h{h}"](z) for h in self.horizons}, "pooled": z}
        if self.aux_class is not None:
            out["class_logits"] = self.aux_class(z)
        if self.magnitude is not None:
            out["magnitude"] = self.magnitude(z).squeeze(-1)
        return out

    def encoder_parameters(self) -> Iterable[nn.Parameter]:
        return (p for p in self.encoder.parameters() if p.requires_grad)

    def head_parameters(self) -> Iterable[nn.Parameter]:
        for name, p in self.named_parameters():
            if not name.startswith("encoder.") and p.requires_grad:
                yield p


def save_model_card(run_dir: Path, obj: dict) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "training_summary.json").write_text(
        json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8"
    )
