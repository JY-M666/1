"""Smoke tests for the 60-day Alpha360 StockMixer adaptation."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from model import StockMixer  # noqa: E402


def test_alpha360_forward_and_backward():
    """A 60x6 Alpha360 cross-section must preserve the official output shape."""
    model = StockMixer(stocks=735, time_steps=60, channels=6, market=20, scale=3)
    features = torch.randn(735, 60, 6)

    prediction = model(features)

    assert prediction.shape == (735, 1)
    prediction.mean().backward()
    assert model.stock_mixer.dense1.weight.grad is not None
