"""Train the official StockMixer architecture under the MTMD/Qlib protocol.

This script deliberately leaves ``src/train.py`` untouched.  It replaces only
the original dataset and experiment loop with MTMD's Alpha360 handler,
temporal split, MSE objective, validation-IC checkpoint selection, and daily
cross-sectional evaluation.
"""

from __future__ import annotations

import argparse
import collections
import copy
import datetime as dt
import json
import random
from pathlib import Path
from typing import Dict, Iterator, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from model import StockMixer


DEFAULT_PROVIDER_URI = "~/.qlib/qlib_data/cn_data"
DEFAULT_STOCK_INDEX = "../mtmd-fixed/data/csi300_stock_index.npy"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def average_params(params_list: collections.deque) -> collections.OrderedDict:
    """Match the moving parameter average used by MTMD's baseline script."""
    if not params_list:
        raise ValueError("cannot average an empty parameter list")
    n_params = len(params_list)
    averaged = collections.OrderedDict()
    for name in params_list[0]:
        averaged[name] = sum(params[name] for params in params_list) / n_params
    return averaged


class DailyDataLoader:
    """Keep each training item as one complete trading-day cross-section."""

    def __init__(
        self,
        features: pd.DataFrame,
        labels: pd.DataFrame,
        stock_slots: pd.Series,
        device: torch.device,
    ) -> None:
        if not (features.index.equals(labels.index) and labels.index.equals(stock_slots.index)):
            raise ValueError("features, labels, and stock slots must share the same index")
        if features.index.nlevels != 2:
            raise ValueError("Qlib data must use a datetime/instrument MultiIndex")

        daily_count = labels.groupby(level=0).size().to_numpy()
        if len(daily_count) == 0:
            raise ValueError("dataset split contains no daily cross-sections")

        self.features = torch.as_tensor(features.to_numpy(), dtype=torch.float32, device=device)
        self.labels = torch.as_tensor(labels.to_numpy()[:, 0], dtype=torch.float32, device=device)
        self.stock_slots = torch.as_tensor(stock_slots.to_numpy(), dtype=torch.long, device=device)
        self.index = labels.index
        self.daily_count = daily_count
        self.daily_start = np.concatenate(([0], np.cumsum(daily_count)[:-1]))

    @property
    def daily_length(self) -> int:
        return len(self.daily_count)

    def iter_daily(self, shuffle: bool = False) -> Iterator[slice]:
        day_indices = np.arange(self.daily_length)
        if shuffle:
            np.random.shuffle(day_indices)
        for day in day_indices:
            start = self.daily_start[day]
            yield slice(start, start + self.daily_count[day])

    def get(self, slc: slice) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, pd.MultiIndex]:
        return self.features[slc], self.labels[slc], self.stock_slots[slc], self.index[slc]


def build_stockmixer_input(
    features: torch.Tensor,
    stock_slots: torch.Tensor,
    stock_num: int,
    d_feat: int = 6,
    time_steps: int = 60,
) -> torch.Tensor:
    """Place one Qlib trading day in StockMixer's fixed global stock slots."""
    if features.ndim != 2 or features.shape[1] != d_feat * time_steps:
        raise ValueError(
            f"expected features shaped [N, {d_feat * time_steps}], got {tuple(features.shape)}"
        )
    if stock_slots.ndim != 1 or stock_slots.numel() != features.shape[0]:
        raise ValueError("stock_slots must be one-dimensional and align with features")
    if stock_slots.numel() == 0:
        raise ValueError("a trading day cannot be empty")
    if torch.any(stock_slots < 0) or torch.any(stock_slots >= stock_num):
        raise ValueError("stock slot lies outside the fixed StockMixer universe")
    if torch.unique(stock_slots).numel() != stock_slots.numel():
        raise ValueError("one trading day maps multiple instruments to the same stock slot")

    dense = features.new_zeros((stock_num, d_feat * time_steps))
    dense.index_copy_(0, stock_slots, features)
    return dense.reshape(stock_num, d_feat, time_steps).permute(0, 2, 1).contiguous()


def masked_mse(prediction: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
    mask = ~torch.isnan(label)
    if not torch.any(mask):
        raise ValueError("a daily cross-section has no valid labels")
    return nn.functional.mse_loss(prediction[mask], label[mask])


def _summarize_daily(values: list[float]) -> Dict[str, float]:
    valid = np.asarray(values, dtype=float)
    valid = valid[np.isfinite(valid)]
    if len(valid) == 0:
        return {"mean": float("nan"), "ir": float("nan"), "count": 0}
    mean = float(np.mean(valid))
    std = float(np.std(valid, ddof=1)) if len(valid) > 1 else float("nan")
    ir = float(mean / std) if np.isfinite(std) and std > 0 else float("nan")
    return {"mean": mean, "ir": ir, "count": int(len(valid))}


def calculate_metrics(predictions: pd.DataFrame) -> Dict[str, float]:
    """Return day-mean IC/RankIC and their unannualized information ratios."""
    if not {"score", "label"}.issubset(predictions.columns):
        raise ValueError("predictions must contain score and label columns")

    daily_ic: list[float] = []
    daily_rank_ic: list[float] = []
    for _, frame in predictions.dropna(subset=["label"]).groupby(level=0):
        if len(frame) < 2:
            continue
        daily_ic.append(frame["label"].corr(frame["score"], method="pearson"))
        daily_rank_ic.append(frame["label"].corr(frame["score"], method="spearman"))

    ic = _summarize_daily(daily_ic)
    rank_ic = _summarize_daily(daily_rank_ic)
    return {
        "IC": ic["mean"],
        "ICIR": ic["ir"],
        "RankIC": rank_ic["mean"],
        "RankICIR": rank_ic["ir"],
        "ic_days": ic["count"],
        "rank_ic_days": rank_ic["count"],
    }


def evaluate(
    model: StockMixer,
    loader: DailyDataLoader,
    stock_num: int,
    args: argparse.Namespace,
) -> Tuple[float, Dict[str, float], pd.DataFrame]:
    model.eval()
    losses: list[float] = []
    outputs: list[pd.DataFrame] = []
    with torch.no_grad():
        for slc in loader.iter_daily():
            feature, label, stock_slots, index = loader.get(slc)
            model_input = build_stockmixer_input(
                feature, stock_slots, stock_num, args.d_feat, args.time_steps
            )
            prediction = model(model_input)[stock_slots].squeeze(-1)
            losses.append(masked_mse(prediction, label).item())
            outputs.append(
                pd.DataFrame(
                    {"score": prediction.cpu().numpy(), "label": label.cpu().numpy()}, index=index
                )
            )
    predictions = pd.concat(outputs)
    return float(np.mean(losses)), calculate_metrics(predictions), predictions


def train_epoch(
    model: StockMixer,
    optimizer: optim.Optimizer,
    loader: DailyDataLoader,
    stock_num: int,
    args: argparse.Namespace,
) -> float:
    model.train()
    losses: list[float] = []
    for slc in loader.iter_daily(shuffle=True):
        feature, label, stock_slots, _ = loader.get(slc)
        model_input = build_stockmixer_input(
            feature, stock_slots, stock_num, args.d_feat, args.time_steps
        )
        prediction = model(model_input)[stock_slots].squeeze(-1)
        loss = masked_mse(prediction, label)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(model.parameters(), 3.0)
        optimizer.step()
        losses.append(loss.item())
    return float(np.mean(losses))


def _slots_for_frame(frame: pd.DataFrame, stock_map: Dict[str, int], split_name: str) -> pd.Series:
    instruments = frame.index.get_level_values("instrument")
    slots = instruments.map(stock_map)
    missing = pd.isna(slots)
    if missing.any():
        sample = sorted(set(instruments[missing]))[:10]
        raise ValueError(
            f"{split_name} contains {int(missing.sum())} rows absent from stock_index; sample={sample}"
        )
    return pd.Series(slots.astype(np.int64), index=frame.index, name="stock_slot")


def create_loaders(args: argparse.Namespace, device: torch.device) -> Tuple[DailyDataLoader, DailyDataLoader, DailyDataLoader, int]:
    """Load exactly the MTMD Alpha360 handler and temporal splits."""
    try:
        import qlib
        from qlib.config import REG_CN
        from qlib.data.dataset import DatasetH
        from qlib.data.dataset.handler import DataHandlerLP
    except ImportError as error:
        raise RuntimeError("Qlib is required for data loading; install pyqlib first") from error

    provider_uri = str(Path(args.provider_uri).expanduser())
    qlib.init(provider_uri=provider_uri, region=REG_CN)

    start_time = dt.datetime.strptime(args.train_start_date, "%Y-%m-%d")
    train_end_time = dt.datetime.strptime(args.train_end_date, "%Y-%m-%d")
    end_time = dt.datetime.strptime(args.test_end_date, "%Y-%m-%d")
    handler = {
        "class": "Alpha360",
        "module_path": "qlib.contrib.data.handler",
        "kwargs": {
            "start_time": start_time,
            "end_time": end_time,
            "fit_start_time": start_time,
            "fit_end_time": train_end_time,
            "instruments": args.data_set,
            "infer_processors": [
                {"class": "RobustZScoreNorm", "kwargs": {"fields_group": "feature", "clip_outlier": True}},
                {"class": "Fillna", "kwargs": {"fields_group": "feature"}},
            ],
            "learn_processors": [
                {"class": "DropnaLabel"},
                {"class": "CSRankNorm", "kwargs": {"fields_group": "label"}},
            ],
            "label": ["Ref($close, -1) / $close - 1"],
        },
    }
    segments = {
        "train": (args.train_start_date, args.train_end_date),
        "valid": (args.valid_start_date, args.valid_end_date),
        "test": (args.test_start_date, args.test_end_date),
    }
    dataset = DatasetH(handler, segments)
    frames = dataset.prepare(
        ["train", "valid", "test"], col_set=["feature", "label"], data_key=DataHandlerLP.DK_L
    )

    stock_map = np.load(args.stock_index, allow_pickle=True).item()
    if not stock_map or min(stock_map.values()) != 0:
        raise ValueError("stock_index must be a non-empty, zero-based global mapping")
    stock_num = max(stock_map.values()) + 1
    if len(set(stock_map.values())) != stock_num:
        raise ValueError("stock_index values must be contiguous and unique")

    loaders = []
    for split_name, frame in zip(("train", "valid", "test"), frames):
        frame = frame.sort_index()
        loaders.append(
            DailyDataLoader(
                frame["feature"], frame["label"], _slots_for_frame(frame, stock_map, split_name), device
            )
        )
    return *loaders, stock_num


def run_smoke_test(args: argparse.Namespace, device: torch.device) -> None:
    """Run a synthetic daily batch without Qlib or market data."""
    stock_num = 7
    active_slots = torch.tensor([0, 1, 3, 5, 6], dtype=torch.long, device=device)
    features = torch.randn(len(active_slots), args.d_feat * args.time_steps, device=device)
    labels = torch.randn(len(active_slots), device=device)
    model = StockMixer(stock_num, args.time_steps, args.d_feat, args.market, args.scale).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    model_input = build_stockmixer_input(features, active_slots, stock_num, args.d_feat, args.time_steps)
    prediction = model(model_input)[active_slots].squeeze(-1)
    loss = masked_mse(prediction, labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if prediction.shape != labels.shape:
        raise AssertionError(f"prediction shape {prediction.shape} does not match labels {labels.shape}")
    print(f"smoke test passed: input={tuple(model_input.shape)}, loss={loss.item():.6f}")


def main(args: argparse.Namespace) -> None:
    if args.batch_size > 0:
        raise ValueError("StockMixer needs a complete daily cross-section; use --batch_size -1")
    device = torch.device(args.device if args.device else ("cuda:0" if torch.cuda.is_available() else "cpu"))
    seed_everything(args.seed)
    if args.smoke_test:
        run_smoke_test(args, device)
        return

    output_dir = Path(args.outdir or f"output/{args.data_set}_seed{args.seed}")
    if output_dir.exists() and (output_dir / "metrics.json").exists() and not args.overwrite:
        raise FileExistsError(f"{output_dir} already contains a run; use --overwrite or choose --outdir")
    output_dir.mkdir(parents=True, exist_ok=True)

    train_loader, valid_loader, test_loader, stock_num = create_loaders(args, device)
    model = StockMixer(stock_num, args.time_steps, args.d_feat, args.market, args.scale).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    parameter_history: collections.deque = collections.deque(maxlen=args.smooth_steps)
    best_score = -np.inf
    best_state = None
    best_epoch = -1
    stale_epochs = 0

    for epoch in range(args.n_epochs):
        train_loss = train_epoch(model, optimizer, train_loader, stock_num, args)
        parameter_history.append(copy.deepcopy(model.state_dict()))
        model.load_state_dict(average_params(parameter_history))
        valid_loss, valid_metrics, _ = evaluate(model, valid_loader, stock_num, args)
        valid_ic = valid_metrics["IC"]
        print(
            f"epoch={epoch:03d} train_mse={train_loss:.6f} valid_mse={valid_loss:.6f} "
            f"valid_IC={valid_ic:.6f} valid_RankIC={valid_metrics['RankIC']:.6f}",
            flush=True,
        )

        if np.isfinite(valid_ic) and valid_ic > best_score:
            best_score = valid_ic
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            stale_epochs = 0
            torch.save(best_state, output_dir / "best_model.pt")
        else:
            stale_epochs += 1
            if stale_epochs >= args.early_stop:
                print(f"early stopping at epoch {epoch}", flush=True)
                break

    if best_state is None:
        raise RuntimeError("validation IC was never finite; cannot select a checkpoint")
    model.load_state_dict(best_state)
    train_loss, train_metrics, _ = evaluate(model, train_loader, stock_num, args)
    valid_loss, valid_metrics, _ = evaluate(model, valid_loader, stock_num, args)
    test_loss, test_metrics, test_predictions = evaluate(model, test_loader, stock_num, args)
    test_predictions.to_pickle(output_dir / "test_predictions.pkl")

    results = {
        "best_epoch": best_epoch,
        "stock_num": stock_num,
        "train": {"MSE": train_loss, **train_metrics},
        "valid": {"MSE": valid_loss, **valid_metrics},
        "test": {"MSE": test_loss, **test_metrics},
        "args": vars(args),
    }
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as stream:
        json.dump(results, stream, indent=2, ensure_ascii=False)
    print(json.dumps(results, indent=2, ensure_ascii=False), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_set", default="csi300")
    parser.add_argument("--provider_uri", default=DEFAULT_PROVIDER_URI)
    parser.add_argument("--stock_index", default=DEFAULT_STOCK_INDEX)
    parser.add_argument("--d_feat", type=int, default=6)
    parser.add_argument("--time_steps", type=int, default=60)
    parser.add_argument("--market", type=int, default=20)
    parser.add_argument("--scale", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=-1)
    parser.add_argument("--n_epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--early_stop", type=int, default=30)
    parser.add_argument("--smooth_steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="")
    parser.add_argument("--outdir", default="")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--smoke_test", action="store_true")
    parser.add_argument("--train_start_date", default="2007-01-01")
    parser.add_argument("--train_end_date", default="2014-12-31")
    parser.add_argument("--valid_start_date", default="2015-01-01")
    parser.add_argument("--valid_end_date", default="2016-12-31")
    parser.add_argument("--test_start_date", default="2017-01-01")
    parser.add_argument("--test_end_date", default="2020-12-31")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
