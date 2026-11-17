"""Model-construction tests that do not require a real pretrained checkpoint."""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from model import DirectionFusionModel


def _tiny_bert(tmp_path):
    config = transformers.BertConfig(
        vocab_size=64,
        hidden_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=32,
    )
    transformers.BertModel(config).save_pretrained(tmp_path)
    return tmp_path


def _build(path, freeze_layers: int):
    return DirectionFusionModel(
        encoder_path=str(path),
        numeric_dim=3,
        n_tickers=2,
        n_flairs=2,
        horizons=[5],
        hidden_dim=8,
        ticker_dim=4,
        flair_dim=2,
        dropout=0.0,
        freeze_layers=freeze_layers,
    )


def test_fully_frozen_encoder_freezes_every_encoder_parameter(tmp_path):
    model = _build(_tiny_bert(tmp_path), freeze_layers=99)
    assert not any(p.requires_grad for p in model.encoder.parameters())
    assert any(p.requires_grad for p in model.head_parameters())


def test_partial_freeze_keeps_last_block_trainable(tmp_path):
    model = _build(_tiny_bert(tmp_path), freeze_layers=1)
    assert not model.encoder.embeddings.word_embeddings.weight.requires_grad
    assert not next(model.encoder.encoder.layer[0].parameters()).requires_grad
    assert any(p.requires_grad for p in model.encoder.encoder.layer[1].parameters())
