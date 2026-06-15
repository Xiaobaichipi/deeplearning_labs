---
title: 时序预测：滑动窗口 + 时间序列拆分
status: needs-triage
---

## What to build

当前系统只支持经典回归和分类，RNN/LSTM/GRU/Transformer 只是将扁平特征 reshape 成序列形状，不是真正的时序预测。需要增加：

1. **新任务类型：时序预测** — 与"经典回归"并列
2. **滑动窗口配置** — seq_len（输入步长）、pred_len（预测步长）、label_len（decoder 标签前缀，Transformer 用）
3. **时间列选择** — 用户从 CSV 列中指定哪列是时间列
4. **时序安全的数据拆分** — 按时间顺序切分训练/验证/测试集（不随机 shuffle）
5. **模型改造** — 序列模型接收 `(batch, seq_len, n_features)` 形数据；解码器输出 `(batch, pred_len)`

## 设计决策（需讨论）

- 滑动窗口变换发生在数据预处理阶段还是训练数据加载阶段？
- 多目标预测是否一起做？（当前对话中已推迟）
- `features='S'/'M'/'MS'` 模式是否支持？（单变量/多变量）
- 时间编码（month/day/hour 等额外特征）是否需要？

## Acceptance criteria

- [ ] Step 1 新增"任务类型"选择：经典回归 vs 时序预测
- [ ] 选时序预测后显示 seq_len/pred_len/label_len 输入框
- [ ] 选时序预测后显示"时间列"下拉选择
- [ ] `split_data()` 对时序任务做时间顺序切分
- [ ] RNN/LSTM/GRU 模型接收 `(batch, seq_len, n_features)` 输入
- [ ] 训练和评估管线与新数据结构兼容
- [ ] 测试覆盖滑动窗口正确性、时间切分正确性
- [ ] 回归模型和分类模型不受影响

## Blocked by

None - can start immediately (requires design approval first)
