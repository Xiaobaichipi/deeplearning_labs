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

### Bug: label_len 未持久化到模型 meta（项目激活/加载模型时崩溃）

训练 Autoformer（seq_len=36, pred_len=12, label_len=1）后，通过项目激活或加载模型再 Evaluation 报错：
```
The size of tensor a (30) must match the size of tensor b (13) at non-singleton dimension 1
```

**根因**: `_run_and_persist()` 存储模型 meta 时漏掉了 `label_len`。`_reconstruct_model()` 取不到 `label_len` 时回退到 `seq_len // 2 = 18`，与原始训练的 `label_len=1` 不匹配。重建后 `self.label_len = 18` → slice 取末尾 18 个 timesteps + pred_len=12 个 zeros = **30** 个 timesteps，但数据只有 1+12=**13** 个。

**修复**:
- `routes/training.py` — meta 新增 `"label_len": split_result.get("label_len")`
- `routes/projects.py:_reconstruct_model()` — 三优先级：① meta["label_len"]（新模型）② seq_len // 2（旧模型，含 warn）
- `routes/projects.py:load_model_into_session/activate_project` — task_config 补充 label_len

## 验证结果

5. label_len 持久化后，训练→项目激活→加载模型→Evaluation 全链路通过，MSE 一致

## 影响范围（追加）

| 文件 | 修改类型 |
|------|---------|
| `routes/training.py` | Bug fix（meta 持久化 label_len） |
| `routes/projects.py` | Bug fix（_reconstruct_model 兜底 + task_config 补充） |
