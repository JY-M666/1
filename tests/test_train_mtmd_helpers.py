"""Dependency-light checks for MTMD adapter helpers."""

import copy
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import train_mtmd  # noqa: E402
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


class TrackingModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([0.0]))
        self.load_history = []

    def load_state_dict(self, state_dict, *args, **kwargs):
        self.load_history.append(float(state_dict["weight"].item()))
        return super().load_state_dict(state_dict, *args, **kwargs)


def run_smoothing_loop(output_dir, validation_ics, early_stop):
    model = TrackingModel()
    train_starts = []
    evaluation_weights = []
    saved_states = []

    def fake_train_epoch(current_model, _optimizer, _loader, _stock_num, _args):
        train_starts.append(float(current_model.weight.item()))
        with torch.no_grad():
            current_model.weight.add_(10.0)
        return float(current_model.weight.item())

    def fake_evaluate(current_model, _loader, _stock_num, _args):
        evaluation_weights.append(float(current_model.weight.item()))
        validation_index = len(evaluation_weights) - 1
        valid_ic = (
            validation_ics[validation_index]
            if validation_index < len(validation_ics)
            else 0.0
        )
        metrics = {"IC": valid_ic, "RankIC": 0.0}
        predictions = pd.DataFrame({"score": [0.0], "label": [0.0]})
        return 0.0, metrics, predictions

    args = SimpleNamespace(
        batch_size=-1,
        device="cpu",
        seed=0,
        smoke_test=False,
        outdir=str(output_dir),
        data_set="csi300",
        overwrite=False,
        time_steps=60,
        d_feat=6,
        market=20,
        scale=3,
        lr=2e-4,
        smooth_steps=3,
        n_epochs=5,
        early_stop=early_stop,
    )
    with mock.patch.object(
        train_mtmd,
        "StockMixer",
        side_effect=lambda *_args: model,
    ), mock.patch.object(
        train_mtmd,
        "create_loaders",
        side_effect=lambda _args, _device: (
            object(),
            object(),
            object(),
            1,
        ),
    ), mock.patch.object(
        train_mtmd,
        "train_epoch",
        side_effect=fake_train_epoch,
    ), mock.patch.object(
        train_mtmd,
        "evaluate",
        side_effect=fake_evaluate,
    ), mock.patch.object(
        train_mtmd.torch,
        "save",
        side_effect=lambda state, _path: saved_states.append(
            copy.deepcopy(state)
        ),
    ):
        train_mtmd.main(args)
    return model, train_starts, evaluation_weights, saved_states


class ParameterSmoothingTest(unittest.TestCase):
    def test_smoothing_does_not_change_next_training_start(self):
        with tempfile.TemporaryDirectory() as directory:
            model, starts, evaluations, saved_states = run_smoothing_loop(
                Path(directory),
                validation_ics=[1.0, 2.0, 3.0, 4.0, 5.0],
                early_stop=10,
            )

        self.assertEqual(starts, [0.0, 10.0, 20.0, 30.0, 40.0])
        self.assertEqual(evaluations[:5], [10.0, 15.0, 20.0, 30.0, 40.0])
        self.assertEqual(float(saved_states[-1]["weight"].item()), 40.0)
        self.assertEqual(model.load_history[-1], 40.0)

    def test_smoothing_restores_before_early_stop(self):
        with tempfile.TemporaryDirectory() as directory:
            model, starts, evaluations, _ = run_smoothing_loop(
                Path(directory),
                validation_ics=[2.0, 1.0],
                early_stop=1,
            )

        self.assertEqual(starts, [0.0, 10.0])
        self.assertEqual(evaluations[:2], [10.0, 15.0])
        self.assertEqual(model.load_history[:4], [10.0, 10.0, 15.0, 20.0])
