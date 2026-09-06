"""Train StockMixer on MTMD/Qlib Alpha360 data with the official StockMixer objective.

The data source and temporal split are adapted to MTMD/Qlib, while the training
objective and main optimizer defaults follow the official StockMixer code:

* raw next-day return labels (no CSRankNorm on labels)
* regression MSE + alpha * pairwise ranking loss, alpha=0.1
* Adam with lr=1e-3
* 100 epochs by default
* checkpoint selected by minimum validation total loss
* no parameter smoothing and no gradient clipping

Because Alpha360 features are scale-free and do not carry an absolute stock-price
level, the model output is interpreted directly as a return score.  The exact
official ``get_loss`` implementation is reused through an algebraically equivalent
unit-base-price transform: predicted_price = 1 + predicted_return,
base_price = 1.  Therefore the regression and pairwise ranking terms are exactly
the same functions of predicted/ground-truth returns as in the official code.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import random
from pathlib import Path
from typing import Dict, Iterator, Tuple

import numpy as np
import pandas as pd
import torch
import torch.optim as optim

from model import StockMixer, get_loss


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


class DailyDataLoader:
    """Keep each item as one complete trading-day cross-section."""

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


def author_loss(
    prediction_return: torch.Tensor,
    ground_truth_return: torch.Tensor,
    alpha: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reuse the official StockMixer loss exactly, in return space.

    Official StockMixer computes ``return_ratio=(prediction_price-base_price)/base_price``
    and then applies MSE + alpha * pairwise ranking loss.  Alpha360 does not
    preserve absolute price scale, so we set base_price=1 and
    prediction_price=1+prediction_return.  This makes ``return_ratio`` exactly
    equal to ``prediction_return`` while preserving the official loss formula.
    """
    if prediction_return.ndim != 1 or ground_truth_return.ndim != 1:
        raise ValueError("prediction and ground truth must be one-dimensional")
    if prediction_return.shape != ground_truth_return.shape:
        raise ValueError("prediction and ground truth must have the same shape")

    valid = torch.isfinite(ground_truth_return)
    if not torch.any(valid):
        raise ValueError("a daily cross-section has no valid labels")

    pred = prediction_return[valid].unsqueeze(-1)
    gt = ground_truth_return[valid].unsqueeze(-1)
    base_price = torch.ones_like(pred)
    predicted_price = base_price + pred
    mask = torch.ones_like(pred)

    return get_loss(
        predicted_price,
        gt,
        base_price,
        mask,
        pred.shape[0],
        alpha,
    )


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
) -> Tuple[Dict[str, float], Dict[str, float], pd.DataFrame]:
    model.eval()
    total_losses: list[float] = []
    reg_losses: list[float] = []
    rank_losses: list[float] = []
    outputs: list[pd.DataFrame] = []

    with torch.no_grad():
        for slc in loader.iter_daily():
            feature, label, stock_slots, index = loader.get(slc)
            model_input = build_stockmixer_input(
                feature, stock_slots, stock_num, args.d_feat, args.time_steps
            )
            prediction = model(model_input)[stock_slots].squeeze(-1)
            total, reg, rank, return_ratio = author_loss(prediction, label, args.alpha)
            total_losses.append(total.item())
            reg_losses.append(reg.item())
            rank_losses.append(rank.item())

            valid = torch.isfinite(label)
            outputs.append(
                pd.DataFrame(
                    {
                        "score": return_ratio.squeeze(-1).cpu().numpy(),
                        "label": label[valid].cpu().numpy(),
                    },
                    index=index[valid.cpu().numpy()],
                )
            )

    predictions = pd.concat(outputs)
    losses = {
        "Loss": float(np.mean(total_losses)),
        "MSE": float(np.mean(reg_losses)),
        "RankLoss": float(np.mean(rank_losses)),
    }
    return losses, calculate_metrics(predictions), predictions


def train_epoch(
    model: StockMixer,
    optimizer: optim.Optimizer,
    loader: DailyDataLoader,
    stock_num: int,
    args: argparse.Namespace,
) -> Dict[str, float]:
    model.train()
    total_losses: list[float] = []
    reg_losses: list[float] = []
    rank_losses: list[float] = []

    for slc in loader.iter_daily(shuffle=True):
        feature, label, stock_slots, _ = loader.get(slc)
        model_input = build_stockmixer_input(
            feature, stock_slots, stock_num, args.d_feat, args.time_steps
        )
        prediction = model(model_input)[stock_slots].squeeze(-1)
        total, reg, rank, _ = author_loss(prediction, label, args.alpha)

        optimizer.zero_grad()
        total.backward()
        optimizer.step()

        total_losses.append(total.item())
        reg_losses.append(reg.item())
        rank_losses.append(rank.item())

    return {
        "Loss": float(np.mean(total_losses)),
        "MSE": float(np.mean(reg_losses)),
        "RankLoss": float(np.mean(rank_losses)),
    }


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


def create_loaders(
    args: argparse.Namespace,
    device: torch.device,
) -> Tuple[DailyDataLoader, DailyDataLoader, DailyDataLoader, int]:
    """Load Alpha360 features with raw next-day return labels."""
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
                {
                    "class": "RobustZScoreNorm",
                    "kwargs": {"fields_group": "feature", "clip_outlier": True},
                },
                {"class": "Fillna", "kwargs": {"fields_group": "feature"}},
            ],
            # Official StockMixer trains against raw returns.  Keep only
            # DropnaLabel here; do NOT cross-sectionally rank-normalize labels.
            "learn_processors": [{"class": "DropnaLabel"}],
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
        ["train", "valid", "test"],
        col_set=["feature", "label"],
        data_key=DataHandlerLP.DK_L,
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
                frame["feature"],
                frame["label"],
                _slots_for_frame(frame, stock_map, split_name),
                device,
            )
        )
    return *loaders, stock_num


def run_smoke_test(args: argparse.Namespace, device: torch.device) -> None:
    """Run a synthetic daily batch without Qlib or market data."""
    stock_num = 7
    active_slots = torch.tensor([0, 1, 3, 5, 6], dtype=torch.long, device=device)
    features = torch.randn(len(active_slots), args.d_feat * args.time_steps, device=device)
    labels = torch.randn(len(active_slots), device=device) * 0.02
    model = StockMixer(stock_num, args.time_steps, args.d_feat, args.market, args.scale).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    model_input = build_stockmixer_input(
        features,
        active_slots,
        stock_num,
        args.d_feat,
        args.time_steps,
    )
    prediction = model(model_input)[active_slots].squeeze(-1)
    total, reg, rank, _ = author_loss(prediction, labels, args.alpha)
    optimizer.zero_grad()
    total.backward()
    optimizer.step()
    if prediction.shape != labels.shape:
        raise AssertionError(f"prediction shape {prediction.shape} does not match labels {labels.shape}")
    print(
        f"smoke test passed: input={tuple(model_input.shape)}, "
        f"loss={total.item():.6f}, mse={reg.item():.6f}, rank={rank.item():.6f}"
    )


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

    best_valid_loss = np.inf
    best_state = None
    best_epoch = -1
    stale_epochs = 0

    for epoch in range(args.n_epochs):
        train_losses = train_epoch(model, optimizer, train_loader, stock_num, args)
        valid_losses, valid_metrics, _ = evaluate(model, valid_loader, stock_num, args)

        print(
            f"epoch={epoch:03d} "
            f"train_loss={train_losses['Loss']:.6f} "
            f"train_mse={train_losses['MSE']:.6f} "
            f"train_rank={train_losses['RankLoss']:.6f} "
            f"valid_loss={valid_losses['Loss']:.6f} "
            f"valid_mse={valid_losses['MSE']:.6f} "
            f"valid_rank={valid_losses['RankLoss']:.6f} "
            f"valid_IC={valid_metrics['IC']:.6f} "
            f"valid_RankIC={valid_metrics['RankIC']:.6f}",
            flush=True,
        )

        if np.isfinite(valid_losses["Loss"]) and valid_losses["Loss"] < best_valid_loss:
            best_valid_loss = valid_losses["Loss"]
            best_state = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
            best_epoch = epoch
            stale_epochs = 0
            torch.save(best_state, output_dir / "best_model.pt")
        else:
            stale_epochs += 1

        # The official code trains all requested epochs.  Early stopping is
        # disabled by default; --early_stop N is retained only as an optional
        # experiment convenience.
        if args.early_stop > 0 and stale_epochs >= args.early_stop:
            print(f"early stopping at epoch {epoch}", flush=True)
            break

    if best_state is None:
        raise RuntimeError("validation loss was never finite; cannot select a checkpoint")

    model.load_state_dict(best_state)
    train_losses, train_metrics, _ = evaluate(model, train_loader, stock_num, args)
    valid_losses, valid_metrics, _ = evaluate(model, valid_loader, stock_num, args)
    test_losses, test_metrics, test_predictions = evaluate(model, test_loader, stock_num, args)
    test_predictions.to_pickle(output_dir / "test_predictions.pkl")

    results = {
        "best_epoch": best_epoch,
        "stock_num": stock_num,
        "train": {**train_losses, **train_metrics},
        "valid": {**valid_losses, **valid_metrics},
        "test": {**test_losses, **test_metrics},
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
    parser.add_argument("--n_epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument(
        "--early_stop",
        type=int,
        default=0,
        help="0 disables early stopping (official behavior); N enables patience N",
    )
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
