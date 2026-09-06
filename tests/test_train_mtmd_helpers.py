"""Dependency-light checks for MTMD adapter helpers."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from train_mtmd import build_stockmixer_input  # noqa: E402


def test_fixed_slot_layout():
    features = torch.arange(3 * 360, dtype=torch.float32).reshape(3, 360)
    slots = torch.tensor([1, 3, 4])
    result = build_stockmixer_input(features, slots, stock_num=5)

    assert result.shape == (5, 60, 6)
    assert torch.equal(result[1, :, 0], features[0, :60])
    assert torch.count_nonzero(result[0]) == 0


def test_duplicate_slots_are_rejected():
    features = torch.zeros(2, 360)
    slots = torch.tensor([1, 1])
    try:
        build_stockmixer_input(features, slots, stock_num=5)
    except ValueError as error:
        assert "same stock slot" in str(error)
    else:
        raise AssertionError("duplicate stock slots must fail")
