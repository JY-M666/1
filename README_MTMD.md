# StockMixer-MTMD

本分支在官方 [StockMixer](https://github.com/SJTU-DMTai/StockMixer) 代码基础上，提供独立的 MTMD/Qlib baseline 管线；官方 `src/train.py` 不受影响，兄弟目录的 `mtmd-fixed` 也不需要修改。

## 与官方实现的差异

- 数据与标签：Qlib `Alpha360`，标签为 `Ref($close, -1) / $close - 1`，并复用 MTMD 的 `DropnaLabel`、`CSRankNorm` 处理。
- 输入：每个交易日固定全局股票槽位，Alpha360 从 `[N, 360]` 重排为 `[stock_num, 60, 6]`。
- 最小模型适配：原实现将 scale 分支长度固定为 8（仅适合 16 日输入）。本分支按卷积 `kernel_size=2, stride=2` 的输出长度计算 `scale_dim`，故 60 日输入得到 30。
- 模型不接收 `active_mask`，不改 `forward()` 接口。无效槽位以零填充并仍进入官方横截面 mixing；只在选回当天有效股票后计算 MSE 和评价指标。
- 训练：按交易日 batch、MSE、验证集 IC 选 checkpoint、MTMD 默认时间切分。结果报告 IC、ICIR、RankIC、RankICIR；IR 是日度均值除日度样本标准差，未年化。

这定义为“采用 MTMD 实验协议的 StockMixer”，而不是原论文 16×5 输入和原始排序损失的逐字复现。

## 环境与数据

安装依赖：

```bash
pip install -r requirements.txt
```

准备 Qlib 中国市场数据，并提供覆盖实验期间所有 CSI300 成分的、编号从 0 连续开始的股票映射 `.npy` 文件。默认路径是相邻 MTMD 工作目录的：

```text
../mtmd-fixed/data/csi300_stock_index.npy
```

若你的 AutoDL 目录不同，请显式传入 `--provider_uri` 与 `--stock_index`。映射缺失、越界或同一交易日重复槽位会使训练立即失败，防止静默覆盖股票特征。

## 运行

先执行不依赖 Qlib 数据的 smoke test：

```bash
python src/train_mtmd.py --smoke_test
```

执行一个正式 seed：

```bash
python src/train_mtmd.py --data_set csi300 --seed 0
```

依次运行三个 seed：

```bash
for seed in 0 1 2; do
  python src/train_mtmd.py --data_set csi300 --seed "$seed"
done
```

每次运行会在 `output/csi300_seed<seed>/` 写入：

- `best_model.pt`：按验证 IC 选择的权重；
- `metrics.json`：训练、验证、测试的 MSE、IC、ICIR、RankIC、RankICIR；
- `test_predictions.pkl`：测试集逐股票预测与标签。

默认时间切分与 MTMD 一致：训练 2007--2014，验证 2015--2016，测试 2017--2020。StockMixer 的横截面 mixing 不能使用随机小 batch；脚本会拒绝 `--batch_size > 0`。
