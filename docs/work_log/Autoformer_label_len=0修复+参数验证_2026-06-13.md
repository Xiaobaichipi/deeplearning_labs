# Autoformer label_len=0 修复 + 前后端参数验证

## 日期
2026-06-13

## 背景

### Bug: Tensor size mismatch 54 vs 18
训练完成后，点击 "Evaluation" → "Run Evaluation" 报错：
```
Error: The size of tensor a (36) must match the size of tensor b (18) at non-singleton dimension 1
```

**根因**: `label_len=0` 时，Python 的 `-0 == 0`，导致 Autoformer forward 中：
```python
trend_init[:, -self.label_len:, :]  # -0 → 0 → 选择所有 timesteps
```
selects all 36 timesteps instead of none, `cat` 后产生 54 维的 decoder 输入，但 `x_mark_dec` 只有 18 个 timesteps（label_len + pred_len = 0 + 18），shape 不匹配。

### 验证: 参数越界场景分析
| 场景 | 影响 |
|------|------|
| pred_len=0 | 空 tensor → training 崩溃 |
| seq_len=1 | 无 temporal signal，attention/分解无意义 |
| label_len=0 | 修复前 tensor size mismatch；修复后可正常工作 |
| label_len=0 + seq_len=1 | 同上，但 seq_len=1 无实际意义 |
| seq_len <= pred_len | split_data 可能生成空训练集 → 崩溃 |

## 修改内容

### 1. `utils/models/autoformer.py` — label_len=0 guard（**bug fix**）
- **位置**: `_RawAutoformer.forward()`，lines 136-143
- **修改**: 添加 `if self.label_len > 0:` 显式检查，防止 Python `-0` 行为
- **效果**: label_len=0 时直接使用 mean/zeros，不再做 slice+cat

### 2. `routes/training.py` — 后端参数验证
- **位置**: `_setup_training()`，在构建 `ts_params` 之前
- **验证项**：
  - `seq_len >= 2`
  - `pred_len >= 1`
  - `label_len >= 0`
  - `seq_len > pred_len`
- **报错方式**: 抛出 `RouteError`，由 `@handle_errors` 转为 JSON 错误响应

### 3. `routes/data.py` — 后端参数验证
- **位置**: `set_task_config()`，在构建 `new_config` 之前
- **验证项**: 同上（仅在 `task_type == "time_series"` 时生效）
- **报错方式**: 同上

### 4. `templates/index.html` — 前端输入约束（上次会话中已完成）
- `seqLenInput`: `min="2"`
- `predLenInput`: `min="1"`
- `labelLenInput`: `min="0"`

## 验证结果

1. label_len=0 的 Autoformer train/evaluate/predict/CV 全部通过
2. 前端输入框已有 min 约束，防止非法值
3. 后端双重验证（set_task_config + _setup_training）防止逻辑绕过
4. 非法参数返回清晰的中文错误信息

## 影响范围

| 文件 | 修改类型 |
|------|---------|
| `utils/models/autoformer.py` | Bug fix |
| `routes/training.py` | 新增验证 |
| `routes/data.py` | 新增验证 |
| `templates/index.html` | 前端约束（上次会话已完成） |
