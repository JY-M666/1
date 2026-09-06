# MTMD 参数平滑训练轨迹修复设计

## 背景

`src/train_mtmd.py` 当前在每轮训练后，将原始模型参数加入滑动窗口并加载平均参数进行验证，但验证结束后没有恢复该轮训练得到的原始参数。因此下一轮训练从平均参数开始，而 Adam 的内部状态仍对应原始训练轨迹。这与既定 MTMD baseline 中“参数平滑仅用于评估和 checkpoint 选择”的协议不一致。

## 目标

恢复既定训练协议：滑动平均参数只用于验证和 best checkpoint 选择，不改变下一轮训练的起点。

## 修复范围

只修改 `src/train_mtmd.py` 的 epoch 循环，并新增针对该行为的回归测试。以下内容保持不变：

- StockMixer 模型结构与 Alpha360 时间维适配；
- Qlib 数据处理、固定股票槽位及标签协议；
- MSE、IC、ICIR、RankIC、RankICIR 的计算方式；
- Adam 优化器、学习率、参数平滑窗口和 early-stop 规则；
- best checkpoint 仍保存滑动平均参数。

## 训练流程

每个 epoch 按以下顺序执行：

1. 使用当前原始训练参数执行 `train_epoch`。
2. 深拷贝本轮训练后的参数为 `params_ckpt`。
3. 将 `params_ckpt` 加入 `parameter_history`。
4. 计算并临时加载滑动平均参数 `avg_state`。
5. 使用 `avg_state` 计算验证损失和指标。
6. 若 Valid IC 改善，将 `avg_state` 保存为 `best_state` 和 `best_model.pt`。
7. 恢复 `params_ckpt`，使下一轮训练延续原始训练轨迹。
8. 恢复完成后再执行 early-stop 的 `break`。

循环结束后仍加载 `best_state`，再计算最终 Train、Valid 和 Test 指标。

## 异常与边界

- `smooth_steps=1` 时平均参数与 `params_ckpt` 相同，流程仍保持一致。
- 达到 early-stop 时也必须先恢复 `params_ckpt`；随后循环外加载 `best_state` 用于最终评估。
- 参数历史保存深拷贝，避免后续训练原地更新历史状态。
- 不保存或恢复 optimizer state，因为评估阶段不执行 optimizer 更新；optimizer 应继续保持本轮训练后的状态。

## 测试与验收

新增回归测试验证：

1. 连续两个 epoch、`smooth_steps > 1` 时，第二轮 `train_epoch` 接收到的是第一轮原始训练参数，而不是平均参数。
2. 验证阶段接收到的是滑动平均参数。
3. best checkpoint 保存的仍是滑动平均参数。
4. early-stop 分支在退出训练循环前完成原始参数恢复。
5. 现有 Alpha360 shape、固定股票槽位和 smoke tests 继续通过。

验收标准是修复前新增回归测试失败、修复后全部测试通过，且改动不超出上述范围。
