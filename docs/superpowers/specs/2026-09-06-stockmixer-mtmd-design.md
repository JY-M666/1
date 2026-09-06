# StockMixer-MTMD 设计说明

## 目标

在官方 StockMixer 仓库的 `mtmd-alpha360` 分支中，新增一条独立、可复现实验管线。该管线使用 MTMD 的 Qlib Alpha360 数据、标签、时间切分、模型选择和评价协议；`mtmd-fixed` 不作任何修改。

## 架构边界

- 保留官方 `src/model.py` 的 TriU、时间/通道 mixer、NoGraphMixer、激活函数、残差路径及网络深度。
- 唯一模型结构修改：将 StockMixer 中固定的 `scale_dim = 8` 改为卷积输出的时间长度 `(time_steps - 2) // 2 + 1`。当 `time_steps=60` 时，该值为 30。
- 新增 `src/train_mtmd.py`，不改官方 `src/train.py`。它负责 Qlib 数据读取、固定股票槽位、训练、验证、检查点和指标。
- 新增 `README_MTMD.md`，说明官方代码差异、环境、数据文件和运行命令。

## 数据流

1. 复用 MTMD 的 Alpha360 `DatasetH` handler：标签为 `Ref($close, -1) / $close - 1`，学习端处理器包含 `DropnaLabel` 与 `CSRankNorm`。
2. 采用 CSI300 及 MTMD 固定切分：训练 2007--2014，验证 2015--2016，测试 2017--2020。
3. 读取全局 `stock_index` 映射，把每个交易日的有效股票放入固定槽位，得到 `[stock_num, 360]`，再变换为 `[stock_num, 60, 6]`。
4. 对零填充的无效槽位仍执行官方 `StockMixer.forward(x)`。模型本身不接收 mask；仅从输出中取回当天有效股票，用于 MSE、预测存档及指标计算。
5. 模型输出统一压缩为一维，再与同一交易日的 label 对齐。映射缺失、越界或同日重复槽位属于数据错误，训练必须立即失败。

## 训练与评估

- StockMixer 只允许按交易日训练，即 `batch_size <= 0`。
- 优化目标为 MTMD 的带 NaN 过滤 MSE；这定义为“采用 MTMD 协议的 StockMixer”，并非原论文损失的逐字复现。
- 以验证集平均 IC 选择 checkpoint。
- 对训练、验证和测试按日计算 IC、RankIC；汇总平均值及 ICIR、RankICIR。ICIR 的定义为日度均值除日度标准差；标准差为零时报告 NaN。
- 支持 `--seed 0 1 2` 的独立运行，避免脚本在单次运行中隐式复用随机状态。

## 文件与提交

1. 提交一：时间维度适配和相应 smoke test。
2. 提交二：Qlib/MTMD 训练评估脚本、说明文档与依赖声明。

## 验证

- 模型 smoke test 验证 `[735, 60, 6] -> [735, 1]`，并确认反向传播可用。
- 训练脚本的纯函数测试覆盖槽位构造、重复索引与越界索引拒绝、预测形状。
- 在没有本地 Qlib 数据时，提供 `--smoke_test`，它仅构造合成日度数据，验证一轮前向、MSE 和指标，不声称为实验结果。
