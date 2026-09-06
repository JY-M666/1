# StockMixer-MTMD

本分支在官方 [StockMixer](https://github.com/SJTU-DMTai/StockMixer) 代码基础上，提供独立的 Qlib/CSI300 适配管线；官方 `src/train.py` 不受影响，兄弟目录的 `mtmd-fixed` 也不需要修改。

为保持仓库轻量，本分支不包含官方 NASDAQ、NYSE、SP500 的 `dataset/` 文件。

## 当前适配原则

目标是先尽可能恢复 StockMixer 原作者的训练逻辑，只替换数据源与输入窗口：

- 数据：Qlib `Alpha360`，过去 60 日 × 6 个基础序列，共 360 维；
- 标签：`Ref($close, -1) / $close - 1` 的原始下一日收益率；
- 不再对 label 使用 `CSRankNorm`；只使用 `DropnaLabel`；
- Loss：恢复官方 `MSE + alpha * RankLoss`，默认 `alpha=0.1`；
- Optimizer：Adam；
- 学习率：恢复官方默认 `lr=0.001`；
- epoch：恢复官方默认 `100`；
- checkpoint：按最低 validation total loss 选择，与官方 `train.py` 一致；
- 默认不 early stop；`--early_stop 0` 表示关闭；
- 删除 MTMD 适配版中的参数滑动平均与 gradient clipping。

官方 StockMixer 的 loss 先把预测价格转成 return，再对 return 做 MSE + pairwise ranking loss。Alpha360 输入经过归一化，不携带绝对股价尺度，因此本适配将模型输出直接解释为预测 return，并通过 `base_price=1, prediction_price=1+predicted_return` 调用官方 `get_loss`。这样得到的 return_ratio 恰好等于模型输出，MSE 与 RankLoss 的数学形式与官方实现保持一致。

## 与官方实现仍然存在的必要差异

- 官方输入默认 16 日 × 5 特征；本分支使用 Qlib Alpha360 的 60 日 × 6 特征；
- 原模型 `scale_dim=8` 只对应 16 日输入，本分支按 `Conv1d(kernel=2, stride=2)` 的输出长度自动计算，60 日时为 30；
- 数据源从 NASDAQ/NYSE/SP500 改为 CSI300/Qlib；
- 评价结果额外报告 IC、ICIR、RankIC、RankICIR；
- 股票采用固定全局槽位，以适配 StockMixer 原有固定 stock dimension。

因此这仍然是“StockMixer on CSI300/Alpha360”，不是官方美股实验的逐字复现；但训练 objective、主要 optimizer 超参数和 validation-loss checkpoint 逻辑已恢复到官方实现。

## 环境与数据

准备 Qlib 中国市场数据，并提供覆盖实验期间所有 CSI300 成分的、编号从 0 连续开始的股票映射 `.npy` 文件。

AutoDL 当前建议：

```text
/root/.qlib/qlib_data/cn_data
/root/autodl-tmp/StockMixer-MTMD/data/csi300_stock_index_full_2007_2020.npy
```

## 运行

先 smoke test：

```bash
python src/train_mtmd.py --smoke_test
```

正式 seed0：

```bash
python src/train_mtmd.py \
  --data_set csi300 \
  --provider_uri /root/.qlib/qlib_data/cn_data \
  --stock_index /root/autodl-tmp/StockMixer-MTMD/data/csi300_stock_index_full_2007_2020.npy \
  --seed 0 \
  --outdir output/csi300_author_seed0 \
  --overwrite
```

如需显式使用官方默认值：

```bash
--n_epochs 100 --lr 0.001 --alpha 0.1 --early_stop 0
```

输出：

- `best_model.pt`
- `metrics.json`
- `test_predictions.pkl`

默认时间切分仍保持当前 MTMD 比较协议：Train 2007--2014，Valid 2015--2016，Test 2017--2020。
