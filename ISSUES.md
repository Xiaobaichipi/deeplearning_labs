# Issues Log

## 2026-06-16: 前端测试全覆盖 — 全局挂载方案 + 46 个 Vitest 测试 (feat/informer-integration 分支)

### 概述

在前一阶段（filterUtils 纯函数提取 + 5 测试）基础上，将测试覆盖扩展到所有 DOM 操作函数。采用 **全局挂载方案**：JS 文件末尾添加 `window.xxx = xxx` 导出，`vitest.setup.js` 通过 `(0, globalThis.eval)()` 注入源码到 jsdom 全局作用域，测试通过 `window` 访问函数。

### 设计决策

| 决策 | 选择 | 理由 |
|------|------|------|
| 模块模式 | 全局挂载（非 ES module） | ESLint 检测通过，不修改 HTML `<script>` 标签，`onclick` 兼容 |
| 源码加载 | `(0, globalThis.eval)()` indirect eval | 函数声明提升为全局变量，`const`/`let` 也通过 eval 作用域链路可访问 |
| 加载方式 | setupFiles + 静态 HTML 骨架 | 避免测试文件重复注入 DOM，一次 setup 全局可用 |

### 新增文件

| 文件 | 内容 |
|------|------|
| `vitest.setup.js` | DOM 骨架（200+ 元素匹配 index.html）+ 4 个 JS 文件 indirect eval 加载 |
| `static/js/__tests__/ui.test.js` | 22 个测试（toggleModelParams/populateModelDropdown/esc/showBadge/goToStep 等） |
| `static/js/__tests__/app.test.js` | 9 个测试（updateModelOptions/applyTaskConfig/startTraining 参数收集） |
| `static/js/__tests__/api.test.js` | 10 个测试（API URL 正确性/错误处理） |

### 修改文件

| 文件 | 变更 |
|------|------|
| `static/js/api.js` | 末尾添加 18 个 `window.xxx = xxx` 导出 |
| `static/js/app.js` | 末尾添加 21 个 `window.xxx = xxx` 导出 |
| `static/js/ui.js` | 末尾添加 37 个 `window.xxx = xxx` 导出 |
| `vitest.config.js` | 新增 `setupFiles: ["./vitest.setup.js"]` |

### 测试覆盖

| 函数 | 测试数 | 覆盖场景 |
|------|--------|---------|
| `toggleModelParams` | 7 | mlp、rnn/lstm/gru、transformer、autoformer、informer、crossformer、dlinear |
| `populateModelDropdown` | 5 | TS 过滤、General 过滤、空列表、无匹配、select 缺失 |
| `esc` | 3 | HTML 转义、空字符串、纯文本 |
| `showLoadedModelBadge` | 2 | 显示/隐藏 |
| `showTaskConfigSaved` | 1 | 3 秒后自动隐藏 |
| `onTaskTypeChange` | 2 | TS 显示配置、General 隐藏 |
| `goToStep` | 1 | 激活目标 step |
| `showTrainError` | 1 | 错误信息展示 |
| `updateModelOptions` | 2 | General 全显示、TS 过滤非时序 |
| `applyTaskConfig` | 2 | DOM 值收集、空值回退默认 |
| `startTraining` | 5 | 缺 target 提前返回、MLP 参数、Autoformer 参数、Crossformer 参数、DLinear 参数 |
| API URL 正确性 | 9 | 8 个端点 URL + method 验证 |
| 错误处理 | 1 | error 响应抛异常 |

### 验证

```
JS:  46 passed (1.54s)
Python: 200 passed (100.50s)
Total: 246 passed
```

---

## 2026-06-16: 引入 JS 单元测试 — Vitest + filterUtils 提取 (feat/informer-integration 分支)

### 概述

引入轻量 JS 测试框架 Vitest，覆盖前端纯逻辑函数。采用"提取纯函数到独立文件"策略，不改动现有 JS 加载方式。

### 新增文件

| 文件 | 内容 |
|------|------|
| `package.json` | JS 开发依赖清单（private=true, type=module） |
| `package-lock.json` | 版本锁定 |
| `vitest.config.js` | Vitest 配置（jsdom 环境） |
| `static/js/filterUtils.js` | 纯函数工具文件（当前仅 `filterModelsByTask`） |
| `static/js/__tests__/filterUtils.test.js` | 5 个 Vitest 测试 |

### 修改文件

| 文件 | 变更 |
|------|------|
| `static/js/ui.js` | 删除 `filterModelsByTask` 定义（已移至 `filterUtils.js`） |
| `templates/index.html` | 新增 `<script src="filterUtils.js">` 加载（在 `ui.js` 之前） |
| `.gitignore` | 新增 `node_modules/` |

### 设计决策

| 决策 | 选择 | 理由 |
|------|------|------|
| 测试文件代码复制 | 测试文件复制 `filterModelsByTask` 源码 | 源文件无 `export`，移动前端到 ES module 需要改 HTML `onclick`，风险高 |
| `package.json` 位置 | 项目根目录 | 标准做法，CI 配置简单 |
| Vitest 环境 | jsdom | 后续可测 DOM 操作函数 |
| 测试范围 | 纯函数起步 | 覆盖最易出 bug 的边界条件（undefined、空数组、TS 切换） |

### 验证

```
JS:  5 passed (1.55s)
Python: 59 route tests passed (无变化)
```

### 后续扩展路径

1. 提取更多纯函数到 `filterUtils.js` 或新建 `paramUtils.js`
2. 积累 10+ 纯函数后，考虑 `module.exports` 消除测试文件中的代码复制
3. 如需测试 DOM 操作函数，再走混合模式（`window.xxx` 挂载）

---

## 2026-06-16: Bug 修复 — einops 缺失导致 Flask 服务拒绝连接 (feat/informer-integration 分支)

### Bug: Crossformer 集成后 localhost:5000 拒绝连接

**症状**: 浏览器访问 `localhost:5000` 显示「拒绝了连接请求」，服务启动后立即退出。

**根因**: Crossformer 模型在 `utils/models/crossformer.py:13` 中使用 `from einops import rearrange, repeat`，但 `einops` 包从未加入 `requirements.txt`。`start.sh` 使用的 Python 环境的 pip 列表中没有 `einops`，`main.py` 在导入阶段即崩溃：

```
File "/data/wj/programmer/learning/deeplearning_labs/utils/models/crossformer.py", line 13, in <module>
    from einops import rearrange, repeat
ModuleNotFoundError: No module named 'einops'
```

旧进程（PID 1995182）因启动崩溃退出，`.server.pid` 指向已终止的进程。

**修复**:
1. `requirements.txt` — 新增 `einops>=0.8`
2. 在服务器 Python 环境中执行 `pip install einops`

**验证**: `curl http://localhost:5000` → `200 OK`

---

## 2026-06-15: Bug 修复 — 模型对比图多步预测延时重叠 (feat/informer-integration 分支)

### Bug: 模型对比图多步预测延时重叠

**症状**: Compare Selected 生成模型预测对比图时，真实值和模型预测值都显示为 pred_len 条时间上存在延迟的线条叠加在一起，图形混乱难以观察。

**根因**: 时间序列任务中 `predict()` 返回 `(n_samples, pred_len)` 形状的 2-D 数组，每行是一个滑动窗口的 pred_len 步预测。`plot_model_comparison()` 直接将所有点绘制到图中，未对多步预测做降维处理，导致每个样本的 pred_len 个预测值（时间上连续偏移）全部重叠显示。

**修复**: `utils/plot_utils.py:210` — `plot_model_comparison()` 在绘图前检查输入维度，对 2-D 数组取 `[:, -1]`（每个窗口的最后时间步），将数据降为 1-D 后正常绘图。

```python
if y_true.ndim > 1:
    y_true = y_true[:, -1]
predictions_dict = {
    k: (v[:, -1] if v.ndim > 1 else v)
    for k, v in predictions_dict.items()
}
```

**验证**: 2-D 输入（n_samples=10, pred_len=4）和 1-D 输入均通过，返回合法 base64 PNG。

---

## 2026-06-15: Bug 修复 — Train / Evaluate & Predict 页面丢失 + GPU 不可用 (feat/informer-integration 分支)

### Bug 1: Train / Evaluate & Predict 页面丢失

**根因**: Informer 集成时 `templates/index.html:267` 的 `<div class="input-group">` 闭合标签被误删。导致 `<div class="form-row">` 一直未闭合，后续所有参数面板被吞入 form-row，模型架构 card 的 HTML 结构被破坏，浏览器无法正确渲染 Step 5（Training）和 Step 6（Evaluate & Predict）。

**修复**: 在 `</select>` 后补回缺失的 `</div>`。

| 文件 | 变更 |
|------|------|
| `templates/index.html` | 第 267 行补回 `</div>`，恢复 input-group → form-row → card 的正确嵌套 |

### Bug 2: PyTorch 升级导致 GPU 不可用

**症状**: `torch.cuda.is_available()` 返回 `False`，所有训练在 CPU 上运行，速度极慢。

**根因**: 项目 conda 环境的 PyTorch 被升级到 `2.11.0+cu130`（需 CUDA 13.0 驱动），但系统 NVIDIA 驱动 `535.309.01` 最高支持 CUDA 12.2。`requirements.txt` 中 `torch>=2.0` 无上限锁定，导致 `pip install` 时自动拉取不兼容的版本。

**修复**: 降级 PyTorch 到 `2.5.1+cu121`，锁定 `requirements.txt`。

| 文件 | 变更 |
|------|------|
| `requirements.txt` | `torch>=2.0` → `torch>=2.0,<2.6` |

### 验证

| 检查项 | 结果 |
|--------|------|
| 页面渲染（6 个 step-item） | ✅ |
| `torch.cuda.is_available()` | ✅ True |
| GPU 数量 | ✅ 2 × RTX 3090 |
| GRU GPU 前向传播 | ✅ 通过 |

---

## 2026-06-15: Informer 模型集成 (feat/informer-integration 分支)

### 概述

将 Informer 模型（ProbSparse Attention）集成到 DeepLearning Labs，作为第二个 "large pipeline" 模型。Informer 使用 ProbSparse 自注意力机制（O(L log L) 复杂度），通过 ConvLayer 蒸馏在编码器层间进行下采样，适用于长序列时间序列预测。

### 新增文件

| 文件 | 内容 |
|------|------|
| `utils/models/informer.py` | InformerWrapper + _RawInformer（含 ProbSparse 注意力 + 蒸馏编码器 + 稀疏解码器） |
| `utils/models/informer_layers/__init__.py` | 包初始化 |
| `utils/models/informer_layers/SelfAttention_Family.py` | ProbAttention（Top-k 查询采样）+ AttentionLayer（QKV 投影包装） |
| `utils/models/shared_layers/__init__.py` | 共享层包初始化，导出 TokenEmbedding / PositionalEmbedding / DataEmbedding / Encoder / Decoder 等 |
| `utils/models/shared_layers/Embed.py` | TokenEmbedding、PositionalEmbedding、DataEmbedding（n_time_features 自适应 Linear） |
| `utils/models/shared_layers/Transformer_EncDec.py` | Encoder（支持 ConvLayer 蒸馏蒸馏）、Decoder、EncoderLayer、DecoderLayer、ConvLayer |
| `utils/models/shared_layers/masking.py` | TriangularCausalMask、ProbMask |

### 架构亮点

| 设计决策 | 说明 |
|----------|------|
| **共享层提取** | Encoder/Decoder/Embedding 等标准 Transformer 组件提取到 `shared_layers/`，后续其他 Transformer 变体可直接复用 |
| **蒸馏模式** | `distil=True` 时，编码器首个 EncoderLayer 后插入 ConvLayer（3×1 conv + 1×1 maxpool），序列长度减半 |
| **ProbSparse Attention** | Top-k 查询采样替代全注意力（O(L log L) 而非 O(L²)），k = c * ln(L) 从均匀分布采样 M 个 key 对 |
| **DataEmbedding** | 继承 Autoformer 的 n_time_features 自适应设计，position 使用 sin/cos 位置编码 |

### 后端修改

| 文件 | 变更 |
|------|------|
| `utils/models/__init__.py` | MODEL_REGISTRY 注册 InformerWrapper（params: d_model, n_heads, e_layers, d_layers, d_ff, factor, distil, dropout, activation）；__all__ 新增 |
| `utils/config.py` | MODEL 字典新增 informer 默认超参数（d_model=256, n_heads=8, e_layers=3, d_layers=3, d_ff=32, factor=3, distil=True, dropout=0.1, activation="gelu"） |

### 前端修改

| 文件 | 变更 |
|------|------|
| `templates/index.html` | modelType 新增 Informer 选项；新增 informerParams 配置区域（d_model, n_heads, e_layers, d_layers, d_ff, factor, distil toggle, activation） |
| `static/js/app.js` | allOptions 新增 Informer；tsModels 数组新增；startTraining() 读取 informer 参数 |
| `static/js/ui.js` | toggleModelParams() 显示/隐藏 informer 参数区 |

### 测试验证

| 功能 | 状态 |
|------|------|
| 训练流（2 epochs, loss 0.8805） | ✅ |
| 评估（MSE 0.8390 + 可视化） | ✅ |
| 预测（40 条预测 + 散点图/折线图） | ✅ |
| 交叉验证（2-fold, scores -0.43/-0.34） | ✅ |

### 技术债务

| 项目 | 说明 | 状态 |
|------|------|------|
| Autoformer 未迁移到 shared_layers | `autoformer_layers/Autoformer_EncDec.py` 包含独立副本的 Encoder/Decoder/EncoderLayer/DecoderLayer，与 `shared_layers/Transformer_EncDec.py` 高度重复。后续 `feat/unify-shared-layers` 分支应将 Autoformer 改为引用共享层并删除冗余代码。 | ⏳ 待处理 |

---

## 2026-06-09: Autoformer 大模型流水线集成 (feat/large-model-pipeline 分支)

### 概述

将 Autoformer 作为首个 "large pipeline" 模型集成到 DeepLearning Labs。引入双流水线架构（small/large），解决 Autoformer 4 参数 forward 签名与现有 model(X) 模式的不兼容问题。

### 双流水线架构

| Pipeline | 模型 | forward 签名 | 数据处理 |
|----------|------|-------------|---------|
| small | MLP/CNN/RNN/LSTM/GRU/Transformer | `model(X)` → output | 标准 2D/3D 数据 |
| large | Autoformer（样板） | `model(x_enc, x_mark_enc, x_dec, x_mark_dec)` → output | 5 元组 window（含时间特征） |

### 新增文件

| 文件 | 内容 |
|------|------|
| `utils/models/autoformer.py` | AutoformerWrapper + _RawAutoformer + 自定义 _DataEmbeddingWoPos |
| `utils/models/autoformer_layers/__init__.py` | 包初始化 |
| `utils/models/autoformer_layers/Embed.py` | 值嵌入 + 时间特征嵌入 |
| `utils/models/autoformer_layers/AutoCorrelation.py` | Autoformer 核心注意力机制 |
| `utils/models/autoformer_layers/Autoformer_EncDec.py` | Encoder/Decoder/Decomp |

### 后端修改

| 文件 | 变更 |
|------|------|
| `utils/models/base.py` | BaseModel 新增 `pipeline = "small"` 类属性 |
| `utils/models/__init__.py` | MODEL_REGISTRY 新增 `pipeline` 字段；注册 AutoformerWrapper；新增 `get_model_pipeline()` / `get_large_model_types()` |
| `utils/config.py` | MODEL 字典新增 autoformer 默认超参数 |
| `utils/data_utils.py` | 新增 `_create_large_windows()` 5 元组窗口；`split_data()` 新增 large pipeline 分支；新增 `normalize_data_apply()` |
| `utils/model_utils.py` | `train_model()`/`predict()`/`evaluate()`/`cross_validate_model()` 增加 large pipeline 分支 |
| `routes/training.py` | `_run_and_persist()` 透传 `extra_model_kw`/`large_kw`；`_setup_training()` 支持 large pipeline 归一化；meta 持久化 pipeline/n_time_features |
| `routes/evaluation.py` | evaluate/predict/validate 透传 `x_mark_test`/`dec_inp_test`/`y_mark_test`；validate 传递 `extra_model_kw` 给模型创建 |
| `routes/projects.py` | `_reconstruct_model()` 补充 n_time_features；`activate_project()` 加载最新模型到主槽位 |

### 前端修改

| 文件 | 变更 |
|------|------|
| `templates/index.html` | modelType 新增 autoformer 选项；新增 autoformerParams 配置区域（d_model, n_heads, e_layers 等） |
| `static/js/app.js` | startTraining() 读取 autoformer 参数 |
| `static/js/ui.js` | toggleModelParams() 显示/隐藏 autoformer 参数区 |

### 关键修复

1. **trend_proj** — trend_init 从 enc_in 维投影到 c_out=1，防止 Autoformer 输出 (batch, pred_len, enc_in) 而非 (batch, pred_len, 1)
2. **n_time_features** — 自定义 `_DataEmbeddingWoPos` 替代硬编码 freq→dim 映射，适配每小时 5 维时间特征
3. **CV 模型创建** — 交叉验证时传递 `extra_model_kw`（n_time_features, seq_len, label_len）到 `create_model()`
4. **模型重建** — `_reconstruct_model()` 和 model meta 均保存/恢复 n_time_features

### 测试验证

| 功能 | 状态 |
|------|------|
| SSE 训练流（train_loss 16.43→2.82, avg_epoch_time 11.75s） | ✅ |
| 评估（MSE/R2 + 可视化） | ✅ |
| 预测（101 条 + 折线图） | ✅ |
| 交叉验证（2-fold，原 n_time_features 维度 Bug 已修复） | ✅ |
| 项目激活 + 模型重建（序列化/反序列化完整链路） | ✅ |

### 后续修复

| 日期 | 问题 | 修复 |
|------|------|------|
| 2026-06-09 | 时间序列模式下 Autoformer 从模型类型下拉菜单消失 | `static/js/app.js:updateModelOptions()` 中 `tsModels` 数组和 `allOptions` 对象均添加 `"autoformer"` |

---

## 2026-06-13: Autoformer label_len=0 修复 + 前后端参数验证 (feat/large-model-pipeline 分支)

### Bug: label_len=0 导致 tensor size mismatch

**症状**: 训练 Autoformer（pred_len=12, label_len=0）后，Evaluation → Run Evaluation 报错 `The size of tensor a (36) must match the size of tensor b (18) at non-singleton dimension 1`

**根因**: Python 的 `-0 == 0`，`trend_init[:, -self.label_len:, :]` 选择所有 36 个 timesteps 而非 0 个，cat 后 decoder 输入变为 54 维，但 `x_mark_dec` 只有 18 个 timesteps（label_len + pred_len = 0 + 18）。

**修复** (`utils/models/autoformer.py`):
```python
if self.label_len > 0:
    trend_init = torch.cat([trend_init[:, -self.label_len:, :], mean], dim=1)
    seasonal_init = torch.cat([seasonal_init[:, -self.label_len:, :], zeros], dim=1)
else:
    trend_init = mean
    seasonal_init = zeros
```

### 参数验证（防止越界）

| 位置 | 文件 | 验证项 |
|------|------|--------|
| 后端（训练前） | `routes/training.py:_setup_training()` | seq_len≥2, pred_len≥1, label_len≥0, seq_len>pred_len |
| 后端（配置时） | `routes/data.py:set_task_config()` | 同上 |
| 前端（输入约束） | `templates/index.html` | seqLenInput min=2, predLenInput min=1, labelLenInput min=0 |

### Bug: label_len 未持久化到模型 meta（项目激活/加载模型时崩溃）

**症状**: 训练 Autoformer（seq_len=36, pred_len=12, label_len=1）后，通过项目激活或加载模型再 Evaluation 报错 `The size of tensor a (30) must match the size of tensor b (13) at non-singleton dimension 1`

**根因**: `_run_and_persist()` 存储模型 meta 时漏掉了 `label_len`。`_reconstruct_model()` 取不到 `label_len` 时回退到 `seq_len // 2 = 18`，与原始训练的 `label_len=1` 不匹配：
- 模型重建后 `self.label_len = 18` → slice 取末尾 18 个 timesteps + pred_len=12 个 zeros → **30** 个 timesteps
- 但原始数据中 `dec_inp` 只有 `label_len + pred_len = 1 + 12 = **13**` 个 timesteps
- `dec_embedding(seasonal_init, x_mark_dec)` 时 value_embedding 输出 30 维，temporal_embedding 输出 13 维，崩溃

**修复**:
1. **`routes/training.py`** — meta 新增 `"label_len": split_result.get("label_len")`
2. **`routes/projects.py:_reconstruct_model()`** — 新增三优先级兜底：
   - ① `meta["label_len"]`（新模型）
   - ② `seq_len // 2`（旧模型）并输出 `warnings.warn()`
3. **`routes/projects.py:load_model_into_session/activate_project`** — 恢复 task_config 时补充 `label_len`

**验证结果**: 训练→项目激活→加载模型→Evaluation 全链路通过，MSE 一致。

---

## 2026-06-13: Large Pipeline 归一化修复 — y_train 未归一化导致 loss 膨胀 (feat/large-model-pipeline 分支)

### Bug: Large pipeline 回归任务中 y_train 未归一化

**症状**: 选择归一化 (minmax/mean) 训练 Autoformer，train_loss 高达 ~5000（与不归一化相同量级），归一化"没有效果"。

**根因**: `_setup_training()` 中 large pipeline 的 y_train/y_test 是与 X_train 分开的独立数组（通过 `_create_large_windows()` 生成的 `(n_windows, pred_len)` 窗口）。X 归一化使用 `normalize_data()` 自动处理所有特征列，但 y 归一化逻辑在 small pipeline 路径中通过 `normalize_target()` 处理，large pipeline 路径从未触及 y。

### 修复

**`routes/training.py:_setup_training()`** — 在 large pipeline 分支中，从 `norm_params` 提取目标列的 minmax/mean 统计量显式归一化 y：

| 归一化方法 | 目标列统计量提取 | y 变换 |
|-----------|----------------|--------|
| minmax | `norm_params["min"][target_idx]` / `norm_params["max"][target_idx]` | `(y - min) / (max - min)` |
| mean | `norm_params["mean"][target_idx]` / `norm_params["std"][target_idx]` | `(y - mean) / std` |

**`utils/model_utils.py:evaluate()`** — large pipeline 评估时，在计算 MSE/RMSE/MAE/R² 之前反归一化 preds 和 y_true，确保指标反映原始数据尺度。

**`routes/evaluation.py:api_evaluate()`** — 传递 `y_scaler=split_result.get("y_scaler")` 到 evaluate()。

### 验证结果

| 归一化 | Train Loss (归一化尺度) | Eval MSE (原始尺度) |
|--------|----------------------|--------------------|
| none   | 13.97 (原始尺度)      | 11.59             |
| minmax | 0.06 (归一化尺度)      | 14.80             |
| mean   | 0.88 (归一化尺度)      | 60.72             |

归一化后 train_loss 从 ~5000 降至 0.06，确认 y 归一化正常工作。不同归一化方法的 eval_mse 差异源于优化景观变化导致模型收敛到不同局部极小值，属于预期行为。

---

## 2026-06-09: 时间粒度选择 + cuDNN 兼容 + Cross Validation 输出维度修复 (v2-project-system 分支)

### 1. Bug — split_data 时间序列路径 time_col==target_col 崩溃

**症状**: 训练时显示 `could not convert string to float: '2024-01-01 00:00:00'`

**根因**: `split_data()` 时间序列路径中，当 `time_col == target_col` 时，条件 `if time_col != target_col` 阻止了时间列的删除，随后 `y = df_enc[target_col].values.astype(np.float32)` 试图将日期字符串转成 float 导致崩溃。

**修复** (`utils/data_utils.py`):
- 将 y 的提取提前到列删除之前，使其独立于 time_col 处理
- 自动检测目标列数据类型——非数值（如日期字符串）时使用 LabelEncoder 转为分类任务

### 2. Feature — 时间粒度选择

**变更**: Step 2 时间序列配置新增 Granularity 下拉菜单（Auto/Year/Month/Day/Hour/Minute）。

| 粒度 | 生成的时间特征 |
|------|----------------|
| Auto | 根据数据采样频率自动推断 |
| Year | year (归一化) |
| Month | year, month |
| Day | year, month, day, weekday |
| Hour | year, month, day, weekday, hour |
| Minute | year, month, day, weekday, hour, minute |

Auto 模式计算中位数时间间隔：≥28d → month、≥1d → day、≥1h → hour、否则 minute。

**涉及文件**:
- `utils/data_utils.py` — `_infer_granularity()`, `time_encoding()` granularity 参数, `split_data()` time_granularity 参数
- `routes/training.py` — 传递 time_granularity
- `routes/projects.py` — 恢复 time_granularity
- `routes/data.py` — task-config 接受 time_granularity
- `templates/index.html` — Granularity 下拉框
- `static/js/app.js` — save/load granularity

### 3. Bug — cuDNN CUDNN_STATUS_NOT_INITIALIZED

**症状**: 训练 RNN/LSTM/GRU/CNN 时显示 `RuntimeError: cuDNN error: CUDNN_STATUS_NOT_INITIALIZED`。

**根因**: Ollama 安装的 cuDNN 9.21 覆盖了 PyTorch 自带的 nvidia-cudnn-cu12 9.1.0。PyTorch 2.5.1+cu121 与 cuDNN 9.21 的 RNN/conv API 不兼容，`nn.Linear` (BLAS) 和 Transformer 不受影响。

**修复** (`utils/config.py`):
- 启动时运行运行时探测：创建微型 GRU，移至 CUDA，前向传播验证
- 失败时全局禁用 cuDNN：`torch.backends.cudnn.enabled = False`
- PyTorch 退化到原生 CUDA 实现（速度略慢但数值结果一致）
- 设备下拉框自动追加 `(cuDNN disabled)` 标签

### 4. Bug — Cross Validation 时序多步输出维度不匹配

**症状**: 训练时序模型后点击 Cross Validation，显示 `The size of tensor a (32) must match the size of tensor b (12) at non-singleton dimension 1`。

**根因**: `routes/evaluation.py:api_validate()` 硬编码 `output_dim = 1`（第166行），但时序多步预测（如 pred_len=12）的 y 是 `(batch, pred_len)` 即 `(32, 12)`。MSE Loss 广播时 shape 不匹配。

**修复**:
- `routes/evaluation.py` — 时序任务时 `output_dim = split_result["pred_len"]`
- `routes/projects.py` — `compare_models()` 同理修复

**回归测试**: `tests/test_routes.py` — 新增 `test_validate_time_series_multi_step`

### 测试

160/160 测试通过，0 failed。

---

### 背景

Code Review 发现 `v2-project-system` 分支存在模型输入 shape 不匹配的问题：滑动窗口生成的数据被拍平为 2D，而 RNN/LSTM/GRU/Transformer 需要原生 3D `(batch, seq_len, n_features)` 输入。

### 修复列表

#### 1. 滑动窗口返回 3D (核心修复)
- **`utils/data_utils.py:_create_sliding_windows()`** — 返回 `(n_windows, seq_len, n_features)` 而非拍平的 2D
- **`utils/data_utils.py:split_data()`** — `input_dim = n_features`（而非 `n_windows * seq_len`）
- **`utils/data_utils.py:normalize_data()`** — 增加 3D→2D→归一化→3D 回程逻辑

#### 2. 模型适配 3D 输入
- **RNN/LSTM/GRU** (`rnn.py`, `lstm.py`, `gru.py`) — `input_size=1` → `input_size=input_dim`，移除 `view()` 拍平
- **Transformer** (`transformer.py`) — 移除 `unsqueeze(1)`/`squeeze(1)`，改用 `mean(dim=1)` 池化
- **MLP** (`mlp.py`) — 3D 输入时 per-timestep 推理 + mean 池化
- **CNN** (`cnn.py`) — 已正确处理 3D（`view(x.size(0), 1, -1)`），无需修改

#### 3. label_len 字段
- **`utils/config.py`** — `TIME_SERIES` 新增 `label_len: 0`
- **`routes/data.py`** — task-config 端点读写 label_len
- **`templates/index.html`** — 时间序列配置区新增 Label Length 输入框
- **`static/js/app.js`** — config 读写 label_len

#### 4. Evaluate/Predict Device 硬编码
- **`routes/evaluation.py`** — `device="cpu"` 改为从 `model_config` 读取训练时使用的 device

#### 5. 测试
- **`tests/test_data_utils.py`** — 新增 7 个时间序列测试（3D 形状、chronological 顺序、label_len、normalize 3D 保形、不足数据异常）

### 测试结果

159/159 测试通过，0 failed。

---

## 2026-06-08: 新增 Device 选择功能 (v2-project-system 分支)

Hyperparameters 中增加 Device 下拉框，列出 CPU 和所有可用 GPU，支持多显卡并行训练。

### 后端

- **`utils/config.py`** — 新增 `get_available_devices()` 检测并返回所有可用设备列表（含 GPU 型号名）；新增 `parse_device()` 解析用户选择的设备字符串（支持 `"cuda:0  (RTX 3080)"` → `"cuda:0"`，`"all  (DataParallel multi-GPU)"` → `["cuda:0", "cuda:1", ...]`）
- **`main.py`** — 将 `devices` 列表传入模板 cfg
- **`utils/model_utils.py`** — `train_model()` 支持 `device` 参数为 list 时使用 `nn.DataParallel` 多 GPU 训练；训练结束后自动 unwrap DataParallel 确保 state_dict 兼容性；`predict()` 和 `evaluate()` 兼容 list 类型 device 参数
- **`routes/training.py`** — `_build_config()` 从请求参数读取 device，使用 `parse_device()` 解析

### 前端

- **`templates/index.html`** — Hyperparameters 新增 Device `<select>` 下拉框，选项由后端动态生成
- **`static/js/app.js`** — `startTraining()` 将 `device` 字段加入请求参数

### 测试

- 152 测试全部通过

---

## 2026-06-08: Bug — 项目激活后 Cross Validation 报 KeyError 'model_type' (v2-project-system 分支)

### 症状

训练完模型后，点击 Step 6 → Cross Validation → Run，页面显示 `Error: 'model_type'`。

### 根因

`routes/projects.py` 中 `load_model_into_session()` 和 `activate_project()` 调用 `sm.set_model_config()` 时只存储了 `meta.get("model_params", {})`。而 `routes/evaluation.py:api_validate()` 需要从 model_config 读取 `model_type` 和 `model_params` 等多个字段。

对比训练流程：`routes/training.py:203` 存储的是 `_build_config()` 返回的完整 dict（含 `model_type`、`model_params`、`learning_rate`、`batch_size`、`device`），因此训练后直接 Cross Validation 正常，但项目激活/加载模型后调用 `api_validate()` 就报了 KeyError。

### 修复

`routes/projects.py` 两处 `sm.set_model_config()` 改为存储完整配置字典：

```python
sm.set_model_config(data_id, {
    "model_type": meta.get("model_type"),
    "model_params": meta.get("model_params", {}),
    "learning_rate": meta.get("learning_rate", config.TRAINING["learning_rate"]),
    "batch_size": meta.get("batch_size", config.TRAINING["batch_size"]),
    "device": meta.get("device", config.DEVICE),
})
```

### 涉及文件

- `routes/projects.py` — 两处 set_model_config 调用修复，新增 `from utils import config`

### 测试

58 个 route 测试全部通过。

---

## 2026-06-08: 项目系统 (Projects) — 持久化存储 + 前端项目列表 (jiagou_youhua 分支)

### 概述

新增项目系统，每个数据集成为一个"项目"（Project），上传的数据集、训练好的模型均持久化存储到磁盘，支持随时重新激活。

### 后端

**新增 `utils/project_manager.py`** — `ProjectManager` 类，文件级持久化：

- 项目目录结构：`projects/<project_id>/`
  - `config.json` — name / timestamps / model_count
  - `dataset/data.csv` — 上传的数据集（内部统一 CSV 格式）
  - `splits/latest.json` — 最近一次训练数据拆分（numpy→JSON 序列化，支持 LabelEncoder 重建）
  - `models/<model_id>/state_dict.pt` — PyTorch 模型参数
  - `models/<model_id>/config.json` — 超参数、最终指标等元数据

**新增 `routes/projects.py`** — Blueprint `projects_bp`：

- `GET /api/projects` — 列出所有项目（按更新时间降序）
- `POST /api/projects` — 创建项目（支持 multipart 上传数据集）
- `GET /api/projects/<id>` — 获取单个项目信息
- `DELETE /api/projects/<id>` — 删除项目
- `POST /api/projects/<id>/activate` — 激活项目，将数据集/拆分/模型恢复到 SessionManager
- 激活时存储 `session["active_project_id"]`

**训练自动保存** — `routes/training.py` SSE 流和同步训练完成后：

- 自动 `pm.save_split(project_id, split_result)` 保存拆分数据
- 自动 `pm.save_model(project_id, model_id, state_dict, meta)` 保存模型

### 前端

- **项目列表首页** — `#projectList` 作为默认页面（取代直接展示 Step 1）
- **项目卡片网格** — `.project-grid` CSS Grid + `.project-card` 卡片，显示名称/文件名/日期/模型数
- **GSAP 入场动画** — 卡片 `stagger: 0.06` 从下向上淡入，`clearProps: "transform"` 避免与 CSS hover 冲突
- **New Project 模态框** — 支持填写项目名 + 上传数据集
- **← Projects 返回按钮** — `backToProjects()` 切回项目列表，重新加载项目数据
- **训练流程** — Activate 项目后进入训练流程，Step 2 自动加载数据预览
- 调用 `projectManager.save_model()` 时使用 `next_model_id()` 自动生成 ID（`model_001`, `model_002`, …）

### 测试

- `test_routes.py` — 新增 `TestProjectCRUD`（8 个测试）和 `TestProjectActivation`（4 个测试）
- 覆盖：空列表、创建（含/不含文件）、无效格式、存在性/不存在查询、删除、激活、激活后训练→模型持久化验证
- fixture 隔离：`client()` 使用 tempdir 隔离 project 存储路径
- 测试总数 140（新增 13），0 failed

### 涉及文件

- `utils/project_manager.py` — **新增** 190 行
- `routes/projects.py` — **新增** 119 行
- `routes/__init__.py` — 导出 `projects_bp`
- `routes/training.py` — SSE 流 + 同步训练增加持久化存储；引用 `session`
- `main.py` — 注册 `ProjectManager` + `projects_bp`；`/api/reset` 清理 `active_project_id`
- `templates/index.html` — 项目列表区 + trainingFlow 包装 + 模态框 + GSAP CDN
- `static/css/style.css` — `.project-grid` / `.project-card` / `.modal-overlay` / `.empty-state` 等样式
- `static/js/ui.js` — `populateProjectGrid()` / `showNewProjectModal()` / `backToProjects()` / `hideNewProjectModal()`
- `static/js/api.js` — `loadProjects()` / `createProject()` / `activateProject()` / `deleteProject()`
- `static/js/app.js` — 初始化时 `loadProjects()`；全局 `_activeProjectId`

---

## 2026-06-08: Phase 2 — 模型导出功能 (jiagou_youhua 分支)

在项目系统中增加模型文件导出功能，用户可下载训练好的模型（state_dict.pt），支持自定义导出文件名。

- **后端**：`GET /api/projects/<id>/models` 列出模型；`GET /api/projects/<id>/models/<mid>/export?name=` 下载
- **前端**：Step 6 新增 Models 标签，每行显示模型信息 + Export 按钮，弹出 prompt 自定义文件名
- **测试**：新增 `TestModelExport`（6 个测试），总数 146，0 failed

---

## 2026-06-08: Phase 3 — 多模型预测对比图 (jiagou_youhua 分支)

在 Models 标签页中增加多模型对比功能，用户可选中多个已训练的模型，生成对比折线图。

### 后端

- **`utils/plot_utils.py:plot_model_comparison()`** — 新建绘图函数，接收 `y_true` + `predictions_dict`，黑色虚线显示真实值，8 色循环显示各模型预测值
- **`POST /api/projects/<id>/models/compare`** — 接受 `{"model_ids": [...]}`，加载各模型 → 推理 → 反归一化 → 渲染对比图返回 base64

### 前端

- Models 标签每行左侧新增复选框，底部 **Compare Selected** 按钮
- 点击后在按钮下方展示对比图，支持任意数量模型（≥1）

### 测试

- 新增 `TestModelComparison`（5 个测试）— 双模型、单模型、空参数、无效 ID、不存在项目
- 测试总数 152（新增 6），0 failed

## 2026-06-08: 平均每轮耗时 + Step 6 模型选择器 (jiagou_youhua 分支)

### 新增功能

1. **平均每轮训练耗时** — Training Summary 新增 `Avg Time / Epoch` 指标
   - 后端 `train_model()` 记录每轮时间 → `avg_epoch_time` 加入 `final_metrics`
   - Training Summary 的 stats-grid 和 metrics-grid 均显示该指标
   - Models 标签页的模型卡片也显示 `Time/Epoch` 芯片

2. **Step 6 共享模型选择器** — Evaluation / Cross Validation / Predictions 可切换已训练模型
   - 新增 `POST /api/projects/<id>/load-model/<model_id>` — 从磁盘加载模型到 SessionManager
   - Step 6 顶部添加 Active Model 下拉框，选择即加载
   - 激活项目时自动加载最新模型到主槽位（`data_id`，非带后缀的 key）
   - 训练完成后自动刷新下拉框，选中刚训练的模型

### 后端

- **`utils/model_utils.py`** — 新增 `import time`；history 增加 `epoch_times` 列表；每轮记录耗时
- **`routes/training.py`** — 同步/SSE 路径均计算 `avg_epoch_time` 加入 `final_metrics` 和持久化 `meta`
- **`routes/projects.py`** — 新增 `load_model_into_session()` 端点；`activate_project()` 自动加载最新模型到 SessionManager 主槽位

### 前端

- **`templates/index.html`** — Step 6 标签栏上方插入模型选择器（select + loaded badge）
- **`static/js/api.js`** — `loadModelToSession()` / `showLoadedModelBadge()` / `refreshModelDropdown()`；`activateProject()` 激活后自动填充下拉框
- **`static/js/ui.js`** — `populateModelDropdown()` / `onModelSelect()`；avg_epoch_time 展示
- **`static/js/app.js`** — 进入 Step 6 时显示模型选择器

### 测试

- 58 现有测试全部通过，无需新增测试

---

## 2026-06-08: Bug 修复 — New Project 上传文件名溢出 (v2-project-system 分支)

**问题**: New Project 模态框中选择长文件名后，文件名文字超出上传框边界。

**修复**: `#projectUploadText` 追加 `overflow: hidden; text-overflow: ellipsis; white-space: nowrap`，超长文件名截断为 `…`。
`.modal-content .upload-zone` 追加 `max-width: 100%; box-sizing: border-box` 防止容器溢出。

--- + 强制 y 归一化 (jiagou_youhua 分支)

### Bug 1: MAE 累加缺少除以样本总数

**症状**：Train MAE / Val MAE 显示值极大（如 681.61 / 162.16），即使启用归一化后仍未改善。

**根因**：`train_model()` 中对每个 batch 累加 `F.l1_loss(pred, batch_y).item() * batch_x.size(0)` 得到各 batch 的 L1 之和，但在 epoch 结束时缺少 `/= len(train_loader.dataset)` 转换为均值。对比 `train_loss` 有正确的 `train_loss /= len(train_loader.dataset)` 行。

效果：存储在 `history["train_metric"]` 和 `history["val_metric"]` 中的 MAE 实际是 **所有样本的 L1 总和**，而非均值。用户有 N 个训练样本时，MAE ≈ N × 真实均值。

**修复**：在 `utils/model_utils.py` 第 83 行添加 `train_mae /= len(train_loader.dataset)`，第 105 行添加 `val_mae /= len(val_loader.dataset)`。

### Bug 2: 回归目标不随 X 归一化独立启用

**症状**：当用户选择归一化="none"（默认）时，y 保持原始尺度，MAE/MSE 在 y 值量级大时仍然很大。

**根因**：`_setup_training()` 将 y 归一化放在 `if norm_method in ("minmax", "mean"):` 块内，意味着 y 归一化与 X 归一化绑定。用户只想做特征归一化或不做 X 归一化时，y 得不到归一化。

**修复**：将回归 y 归一化逻辑移出 X 归一化条件块。当 X 归一化="none" 时，y 默认使用 "mean" (z-score) 归一化，确保 Loss/MAE 始终在无量纲尺度。

### 涉及文件

- `utils/model_utils.py` — 添加 `train_mae /= len(train_loader.dataset)` 和 `val_mae /= len(val_loader.dataset)`
- `routes/training.py` — `_setup_training()` 回归 y 归一化与 X 归一化解耦，始终独立执行
- `tests/test_model_utils.py` — 新增 `TestTrainMAEScale.test_mae_is_mean_not_sum`
- `tests/test_routes.py` — 新增 MAE < 10.0 断言到 `test_train_returns_history`

### 测试

- 测试总数 127（新增 1），0 failed

---

## 2026-06-08: Predictions 简化 + Training 重构 (jiagou_youhua 分支)

### 变更

1. **Residual 图比例调整** — `figsize=(6,4)` → `(5,4)`，适配 eval 卡片容器
2. **移除冗余 Training History 图** — `plot_training_history` 调用和 `trainingImages` 容器全删除。Live Training Progress 的 Chart.js 已实时展示 Loss/Metric 曲线，不再需要训练后静态 matplotlib 图
3. **新增 Loss/Metric 数据下载** — 新端点 `GET /api/train/history/download?format=csv|xlsx`，训练完成后 Training Summary 区追加两个下载按钮

### 涉及文件

- `utils/plot_utils.py` — residual plot figsize 调整
- `routes/training.py` — 去掉 `plot_training_history` 导入和调用；新增 `/api/train/history/download` 端点
- `templates/index.html` — 去掉 `trainingImages` 容器；Training Summary 增加下载按钮
- `static/js/api.js` — SSE complete handler 移除 trainingImages 渲染；`resetAll()` 移除 trainingImages 清理；新增 `downloadHistory()` 函数

### 测试

- `test_routes.py` — 新增 `test_history_download_csv`、`test_history_download_without_training_returns_400`
- 总计 126 tests, 0 failed

---

## 2026-06-08: 简化 Predictions 页面——去掉表格 + 下拉框 (jiagou_youhua 分支)

### 变更

- **移除 `predictSource` 下拉框** — 不再支持选择 train/test，固定预测 test 集
- **移除预测数据表格** — `predTable` 的 HTML 表格渲染代码全部删除
- **保留两张图表** — 散点图 + 折线对比图正常展示
- **保留下载按钮** — CSV/XLSX 下载固定为 test 集预测结果

### 涉及文件

- `templates/index.html` — 去掉 `predictSource` select 和 `predTable` div
- `static/js/api.js` — `runPredict()` 移除表格渲染逻辑；`downloadPredictions()` 固定 source=test

---

## 2026-06-08: 全局 DPI 提升至 300 (jiagou_youhua 分支)

### 变更

所有 matplotlib 绘图函数 DPI 从 100-120 统一提升到 300，消除图片模糊。

| 函数 | 旧 DPI | 新 DPI |
|---|---|---|
| `plot_training_history` | 120 | 300 |
| `plot_feature_importance` | 100 | 300 |
| `plot_data_distribution` | 100 | 300 |
| `plot_correlation_heatmap` | 120 | 300 |
| `fig_to_base64`（默认） | 100 | 300 |
| `plot_pred_vs_true` | 300 | 300（不变） |
| `plot_pred_vs_true_line` | 300 | 300（不变） |

### 涉及文件

- `utils/plot_utils.py` — 5 处 dpi 参数修改 + `fig_to_base64()` 默认值变更

---

## 2026-06-08: 配置 Matt Pocock's Skills 工程架构 (jiagou_youhua 分支)

### 新增文件

- **`CLAUDE.md`** — 新增 `## Agent skills` 区块，引用 `docs/agents/`
- **`docs/agents/issue-tracker.md`** — Local markdown issue tracker 约定（`.scratch/<feature>/`）
- **`docs/agents/triage-labels.md`** — 五个 triage 角色映射表
- **`docs/agents/domain.md`** — 单上下文领域文档消费规则

### 配置项

- Issue tracker: Local markdown（`.scratch/<feature>/issues/`）
- Triage 标签: 使用默认名称（needs-triage / needs-info / ready-for-agent / ready-for-human / wontfix）
- 领域文档: 单上下文（`CONTEXT.md` + `docs/adr/` 在根目录）

---

## 2026-06-08: 测试覆盖 (jiagou_youhua 分支)

### 新增测试文件

- **`tests/test_data_utils.py`** — 28 个测试覆盖 normalize, split, clean, fill 函数
- **`tests/test_session.py`** — 24 个测试覆盖 SessionManager CRUD + 磁盘重载
- **`tests/test_model_utils.py`** — 20 个测试覆盖模型创建、训练、推理、评估、交叉验证
- **`tests/test_plot_utils.py`** — 21 个测试覆盖所有绘图函数的 base64 PNG 输出有效性
- **`tests/test_routes.py`** — 31 个测试覆盖 Flask 路由端到端（上传→训练→评估→预测→下载→重置）

### 代码修复

- `utils/plot_utils.py:plot_confusion_matrix()` — 空矩阵时返回 `None` 避免 matplotlib crash
- `utils/model_utils.py` — `predict()` 和 `evaluate()` 回归分支使用 `np.atleast_1d` 防止单样本场景下 0-d 标量导致 sklearn 报错
- `utils/model_utils.py` — 训练循环中 `outputs.squeeze()` → `squeeze(-1)` 避免单样本 batch 的维度广播 warning

### 总计

124 tests, 0 failed, 0 skipped

## 2026-06-08: Predictions 图表调整——缩小尺寸 + 新增折线图 (jiagou_youhua 分支)

### 变更

1. **散点图缩小** — `plot_pred_vs_true()` 的 figsize 从 (6,5) 改为 (5,4)，dpi 从 100 提升到 300，点大小调小
2. **新增折线图** — `plot_pred_vs_true_line()`：以样本序号为 X 轴，蓝色 True Values / 橙色 Predictions 双折线对比
3. **API 返回两张图** — `api_predict()` 的 JSON 响应新增 `line_plot_image` 字段
4. **前端并排展示** — `predChart` 内 scatter + line 两个 `.image-card` 并排显示

### 涉及文件

- `utils/plot_utils.py` — 修改 `plot_pred_vs_true()` + 新增 `plot_pred_vs_true_line()`
- `routes/evaluation.py` — 引入并返回 `line_plot_image`
- `static/js/api.js` — 前端展示两张图

---

## 2026-06-08: Predictions 增加对比图 + CSV/XLSX 下载 (jiagou_youhua 分支)

### 新增功能

1. **预测值 vs 真实值散点图** — `api_predict()` 返回中增加 `plot_image`（回归任务使用 `plot_pred_vs_true()`），前端表格下方显示
2. **CSV/XLSX 下载** — 新增 `GET /api/predict/download?source=test&format=csv` 端点，前端两个下载按钮

### 重构

- 提取 `_compute_predictions()` 共享函数，消除 `api_predict()` 和下载端点之间的推理逻辑重复

### 验证

- Predict 端点返回 plot_image ✅
- CSV 下载 920 bytes，mimetype text/csv ✅
- XLSX 下载 6368 bytes，mimetype application/vnd.openxmlformats ✅
- 缺 openpyxl 时优雅降级返回错误提示 ✅

---

## 2026-06-08: 评估指标改为归一化尺度以避免数值误解 (jiagou_youhua 分支)

### 变更

用户反馈 `Run Evaluation` 后的 MSE=49.6912 看起来太大，怀疑归一化失效。经排查归一化链路正确——此数值是反归一化后的原始尺度 MSE。

为消除困惑，`evaluate()` 回归分支不再反归一化，评估指标（MSE/RMSE/MAE）直接使用归一化尺度，与训练过程中的 loss 一致。

### 涉及文件

- `utils/model_utils.py:evaluate()` — 移除 `y_scaler` 参数和 `denormalize_target()` 调用
- `routes/evaluation.py:api_evaluate()` — 不再传 `y_scaler` 给 evaluate()

### 对比验证

使用测试数据（y 范围 88~251）：

| 场景 | Train Loss | Eval MSE | RMSE | R² |
|---|---|---|---|---|
| 归一化 + 反归一化(旧) | 0.124 | 90.64 | 9.52 | 0.86 |
| **归一化 + 归一化(新)** | **0.124** | **0.116** | **0.34** | **0.85** |
| 无归一化 | 547.12 | 90.45 | 9.51 | 0.86 |

---

## 2026-06-08: 架构深度优化 (jiagou_youhua 分支)

### 候选 1: 交叉验证使用已训练的 PyTorch 模型

**问题**：`/api/validate` 创建了一个 sklearn MLP 模型（硬编码 (64,32) 架构），与用户实际训练的 PyTorch 模型完全无关。

**修复**：
- 新增 `cross_validate_model()` (`utils/model_utils.py`) — KFold 分割数据后，每折创建一个与训练相同架构的新模型，训练后评分
- `api_validate()` 从 `SessionManager` 读取 `model_config`（模型类型、参数），传入 `cross_validate_model()`
- `SessionManager` 新增 `set_model_config()` / `get_model_config()` 存储训练配置

### 候选 2: 反归一化提取为共享函数

**问题**：`y_scaler` 反归一化的 mean/std / minmax 分支在 `model_utils.py:evaluate()` 和 `routes/evaluation.py:api_predict()` 中重复。

**修复**：
- 新增 `denormalize_target(y, y_scaler)` (`utils/data_utils.py`) — 单行调用替代两处的 10 行重复逻辑

### 候选 3: 配置默认值集中管理

**问题**：学习率、epochs、batch_size 等超参默认值散落在 `routes/training.py` 的路由、模型注册表和 HTML 中。

**修复**：
- 新增 `utils/config.py` — TRAINING、MODEL、CV 三个配置字典 + DEVICE，统一管理所有训练默认值
- `_build_config()` 和 `_setup_training()` 改为从 `config` 模块读取默认值

### 候选 4: 移除 evaluate() 中的字体副作用

**问题**：`evaluate()` 在计算指标前调用 `setup_chinese_font()`，全局修改 matplotlib 字体。

**修复**：
- `plot_utils.py` 已在模块初始化时调用 `setup_chinese_font()`，`evaluate()` 中移除重复调用即可

---

## 2026-06-08: 训练进度实时曲线图 (jiagou_youhua 分支)

### 变更描述

在实时训练面板（Live Training Progress）的指标数字卡片下方，增加 2 个并排的趋势图：

- **Loss 曲线** — Train Loss（蓝）/ Val Loss（红），epoch 维度实时延伸
- **Metric 曲线** — Train Metric（绿）/ Val Metric（黄），epoch 维度实时延伸

### 技术方案

- 使用 Chart.js (CDN) 绘制折线图，`animation: false` 避免频繁更新时的闪烁
- 每收到一个 SSE progress 事件，向 `_chartData` 追加数据点，调用 `chart.update("none")` 无动画刷新
- 每次 `initTrainingProgress()` 时调用 `destroyCharts()` + `_initCharts()` 重置
- `destroyCharts()` 暴露为全局函数，供 `resetAll()` 清理

### 涉及文件

- `templates/index.html` — Chart.js CDN script + 2 个 canvas 容器
- `static/js/ui.js` — 图表初始化、数据追加、更新逻辑
- `static/css/style.css` — `.chart-box` 容器高度 + 移动端 1 列堆叠

---

## 2026-06-05: 新增归一化功能后上传数据不显示

### Bug描述

在 Model Config 步骤新增归一化（Min-Max / Mean）选项后，用户上传数据集时后端返回成功，但前端页面内容不加载（Step 2 不显示数据预览、统计等）。

### 根因分析

`static/js/app.js` 中 `startTraining()` 函数的 `params` 对象定义缺少闭合符号 `};`。

具体过程：在添加 `normalization` 字段到 `params` 对象时，Edit 替换操作意外删除了对象末尾的闭合 `    };`。导致 `startTraining()` 函数体内的对象字面量变为语法错误：

```javascript
const params = {
    ...
    normalization: document.getElementById("normalization").value,
                                                    // ← 缺少 }; 
try {                                               // ← 解析器在此处报错
```

由于整个 `<script>app.js` 在解析阶段就失败，**所有函数**（`handleUpload`、`populateStep2`、`goToStep` 等）均不会被定义。因此：

- `/api/upload` 后端接口正常（curl 测试通过）
- 但上传成功后前端无反应，因为 `handleUpload` 从未执行
- 任何交互函数（`runClean`、`runFill`、`startTraining`）也都不可用

### 修复

在 `app.js:308` 补上缺失的 `    };`，闭合 params 对象：

```javascript
        normalization: document.getElementById("normalization").value,
    };                         // ← 补上这一行

    try {
```

### 教训

编辑 JS 代码后，应在浏览器控制台或通过 Node.js 检查语法：

```bash
node -e "new Function(require('fs').readFileSync('static/js/app.js','utf8')); console.log('OK')"
```

Flask debug reloader 不检测前端文件变化，浏览器也可能缓存旧版 JS。修改后应 Ctrl+F5 强制刷新。

---

## 2026-06-05: 修复目标归一化未接入导致 Loss 过大 (jiagou_youhua 分支)

### Bug描述

选择归一化（Min-Max / Mean）后训练回归模型，Train Loss 依然高达 137.88，归一化似乎"没有用"。

### 根因分析

此前归一化只对特征矩阵 X 做了归一化（`normalize_data()`），但目标值 y 没有归一化。MSE = mean((y_pred - y_true)²)，对于 y 值在数百或数千量级的回归问题，即使 X 归一化得再好，loss 依然与 y 的平方成正比。

### 修复

分两步：

1. **新增 `normalize_target()`** (`utils/data_utils.py:259`) — 对 y 做与 X 相同的归一化（z-score 或 minmax），返回归一化后的 y 和反归一化参数（mean/std 或 min/max）

2. **接入训练管线**：
   - `routes/training.py:_setup_training()` — 回归任务且启用归一化时，对 y_train/y_test 同步归一化
   - `utils/model_utils.py:evaluate()` — 接受 `y_scaler` 参数，计算指标前反归一化预测值和真实值
   - `routes/evaluation.py:api_predict()` — 返回预测结果前反归一化，用户看到原始单位的预测值

效果：归一化后训练过程中的 loss 在合理范围（~1.0 量级），评估指标反归一化后依然反映原始尺度的误差。

---

## 2026-06-05: 代码架构优化 (jiagou_youhua 分支)

### 优化项 1: 提取 evaluate() 绘图代码

将 `utils/model_utils.py` 中 `evaluate()` 函数的内联 matplotlib 代码提取到 `utils/plot_utils.py` 的独立函数中：

- 新增 `fig_to_base64(fig)` — 通用 Figure→base64 工具函数
- 新增 `plot_confusion_matrix(cm, class_names)` — 混淆矩阵
- 新增 `plot_roc_curve(y_true, y_score)` — ROC 曲线 (返回 (base64, auc))
- 新增 `plot_pred_vs_true(y_true, y_pred)` — 预测值 vs 真实值散点图
- 新增 `plot_residuals(y_true, y_pred)` — 残差分布直方图

`evaluate()` 从 ~130 行缩减到 ~60 行，所有绘图逻辑集中在 `plot_utils.py`。

### 优化项 2: SessionManager 封装会话状态

用 `utils/session.py` 中的 `SessionManager` 类替代 `main.py` 中的 4 个全局 dict：

- `_data` — DataFrame 缓存，支持 cache miss 时自动从磁盘重载
- `_models` — 训练好的模型缓存
- `_splits` — 训练/测试拆分结果
- `_histories` — 训练历史记录

### 优化项 3: Blueprint 拆分

将 `main.py` 中的所有路由按功能拆分到 `routes/` 包：

- `routes/data.py` — 数据处理 (upload / info / clean / fill / sample)
- `routes/training.py` — 模型训练 (train)
- `routes/evaluation.py` — 评估/预测/交叉验证 (evaluate / predict / validate)

`main.py` 从 427 行精简到 40 行，仅保留 Flask 初始化、Blueprint 注册和静态页面路由。

### 优化项 4: 前端 JS 模块化

将单文件 `static/js/app.js` (532 行) 按职责拆分为三个文件：

- `ui.js` — DOM 渲染函数 (populateStep2 / populateStep3Columns / toggleModelParams 等)
- `api.js` — 所有 fetch/API 调用函数 (handleUpload / startTraining / runEvaluation 等)
- `app.js` — 仅保留事件监听器和初始化逻辑 (缩减 87%)

### 分支提交记录

```
a9732fa refactor: 前端 JS 模块化
9f23c16 refactor: Blueprint 拆分
b382cd1 refactor: 引入 SessionManager
7b090e8 refactor: 提取 evaluate() 绘图代码
```

---

## 2026-06-08: 候选 C — 消除路由样板代码 (v2-project-system 分支)

### 问题

Flask 路由函数中存在大量重复的 try/except 模板、`current_app.config["session_manager"]` 重复获取、以及 `if not data` 空值检查。每个路由平均约 18 行，其中 10 行是防御性样板代码。

### 方案

引入三个基础设施组件：

1. **`RouteError(Exception)`** — 携带 HTTP status_code 的自定义异常，允许路由直接 `raise RouteError("msg", 400)` 终止请求
2. **`handle_errors` 装饰器** — 捕获 `RouteError` → `jsonify` + status_code；捕获未预期异常 → traceback 日志 + 500
3. **`get_sm()` / `ensure_data()` 快捷函数** — `get_sm()` 替代 `current_app.config["session_manager"]`；`ensure_data()` 替代 `if not data: abort(400)`

### 效果

- **`routes/data.py`** — 5 个路由全部使用 `@handle_errors`，零 try/except，从 ~90 行缩减到 ~40 行纯业务逻辑
- **`routes/evaluation.py`** — 4 个路由同样改造，`if not sm.has_model()` 替换为 `raise RouteError("Model not trained yet")`
- **`routes/training.py`** — `train()` 和 `train_setup()` 使用装饰器，SSE 流保持独立
- 所有路由统一错误返回格式：`{"error": "message"}` + 合适的 HTTP 状态码

### 涉及文件

- `utils/session.py` — 新增 `RouteError`、`handle_errors` 装饰器、`get_sm()`、`ensure_data()`
- `routes/data.py` — 全面采用新机制重写
- `routes/evaluation.py` — 全面采用新机制重写
- `routes/training.py` — 部分采用新机制重写
- `tests/test_routes.py` — 维持 82 测试通过

---

## 2026-06-08: 候选 D — 前端 API/DOM 分离 (v2-project-system 分支)

### 问题

前端 JS 长期存在 API 调用与 DOM 操作混合的问题。`api.js` 中一个函数既负责 `fetch()` 又负责 `document.getElementById()` 渲染，导致无法独立测试、逻辑混乱。

### 方案

将前端 JS 重构为严格的三层架构：

| 层 | 文件 | 职责 | DOM 操作 | fetch 调用 |
|---|---|---|---|---|
| **API** | `api.js` | 纯数据获取 | 禁止 | 唯一来源 |
| **Orchestration** | `app.js` | 流程编排 | 读（form） | 调用 api.js |
| **UI** | `ui.js` | 视图渲染 | 读写（全部） | 禁止 |

### 具体变更

- **`api.js`** — 所有函数重命名为 `_` 前缀（`_uploadFile`、`_cleanData` 等），移除全部 `document.getElementById` 调用（计数归零）。每个函数仅做：fetch → 解析 → 错误检查 → 返回数据
- **`ui.js`** — 新增 `showUploadResult()`、`showCleanResult()`、`showTrainingComplete()`、`showEvalResult()`、`showCVResult()`、`showPredResult()`、`showLoadedModelBadge()`、`resetAllUI()`、`showProjectList()`、`hideProjectList()`、`showUploadLoading()`、`showUploadError()` 等函数。零 fetch 调用
- **`app.js`** — 保留旧函数名称（`handleUpload`、`startTraining` 等）兼容 HTML `onclick`；每个函数链式调用：读取 form → 调用 `_` API → 调用 UI 渲染。管理全局状态（`currentDataInfo`、`_activeProjectId`）

### 额外修复

- **Transformer num_layers 读取错误** — 原代码对 transformer 模型也读取 `rnnNumLayers` 输入框，修复为根据 `modelType` 值选择 `transNumLayers` 或 `rnnNumLayers`
- **`resetAll()` 未清除 `_activeProjectId`** — 修复为显式设置 `_activeProjectId = null`

### 涉及文件

- `static/js/api.js` — 全面重写（~200 行纯 API 层）
- `static/js/ui.js` — 新增 12 个渲染函数
- `static/js/app.js` — 状态管理 + 编排逻辑重写（~310 行）

---

## 2026-06-08: 候选 E — 消除超参数默认值重复 (v2-project-system 分支)

### 问题

训练超参数（learning_rate、batch_size、epochs、dropout 等）和模型参数（hidden_channels、kernel_size、d_model、nhead 等）的默认值散落在三个地方：
1. `utils/config.py` — 定义 TRAINING/MODEL/CV 字典
2. `templates/index.html` — `<input value="...">` 硬编码
3. `static/js/app.js` — JS `||` 回退值硬编码

修改默认值需要同步三处，极易遗漏。

### 方案

通过 Jinja2 模板注入将 config.py 作为唯一真相源，不再手动同步 HTML 和 JS 中的硬编码值。

### 具体变更

1. **config.py 增加 CV 配置** — 增加 `DEFAULT_FOLDS`、`DEFAULT_SEED` 到 CV 字典
2. **`main.py` 传递配置** — `render_template("index.html", cfg={"training": config.TRAINING, "model": config.MODEL, "cv": config.CV})`
3. **`index.html`** — 添加 `<script id="config-data" type="application/json">{{ cfg | tojson }}</script>` JSON 数据块；15 个 `<input value>` 改为 `{{ cfg.training.xxx }}` / `{{ cfg.model.xxx }}`
4. **`app.js`** — `const DEFAULTS = JSON.parse(document.getElementById("config-data").textContent)`；所有 `||` 回退改为 `|| DEFAULTS.training.xxx` / `|| DEFAULTS.model.xxx`

### 效果

- 15 个 HTML `value` 硬编码 → 0
- 15+ 个 JS `||` 回退字面量 → 0
- 修改默认值只需编辑 `utils/config.py`
- 新增超参数只需在 config.py 添加 + HTML 添加输入框，JS 自动获取默认值

### 涉及文件

- `utils/config.py` — 新增 CV 配置字典
- `main.py` — 传递 cfg 给模板
- `templates/index.html` — config-data JSON 块 + 15 处输入值替换
- `static/js/app.js` — DEFAULTS 全局对象 + 回退引用替换

---

## 2026-06-15: Bug 修复 — 项目激活时模型重建 shape 不匹配 (feat/informer-integration 分支)

### Bug: 点击项目报 "size mismatch for rnn.weight_ih_l0"

**症状**: 点击项目（激活）时弹出错误：
```
Error: Failed to reconstruct model: Error(s) in loading state_dict for RNNModel:
size mismatch for rnn.weight_ih_l0: copying a param with shape torch.Size([32,5]) from checkpoint, the shape in current model is torch.Size([32,1])
```

**根因**: 同一项目中先后训练了不同维度的模型（如先 TS RNN input_dim=5，后 General MLP input_dim=1）。每次训练时 `_run_and_persist` 都调用 `pm.save_split()` 覆盖项目的 split 数据。项目激活时，`activate_project()` 从 split 读取 `input_dim` 重建所有模型，导致后续模型使用错误的维度重建 → `load_state_dict` shape 不匹配。

**修复**: `_reconstruct_model()` 优先使用 `meta["input_dim"]` / `meta["output_dim"]`（每模型独立保存），降级到调用方传入的参数。split_result 中的维度只作为后备。

### 涉及文件

- `routes/projects.py` — `_reconstruct_model()` 优先使用 meta 维度

---

## 2026-06-15: 功能增强 — 模型下拉框按任务类型过滤 (feat/informer-integration 分支)

### 功能：Active Model 下拉框按任务类型过滤

**背景**: 时间序列项目里会看到通用模型（如 MLP），通用项目里会看到时间序列模型（如 LSTM），造成混淆。

**方案**: 新增 `filterModelsByTask(models, taskType)` 纯函数，`populateModelDropdown()` 根据当前任务类型过滤模型列表。

### 涉及变更

1. **`static/js/ui.js`**
   - 新增 `filterModelsByTask(models, taskType)` — 单一过滤逻辑
   - `populateModelDropdown()` 接受 taskType 参数，返回过滤后的模型数组
   - 过滤为空时不隐藏选择器，显示 `"No {taskLabel} models trained yet"` 占位消息

2. **`static/js/app.js`**
   - `activateProject()` / `refreshModelDropdown()` — 利用 `populateModelDropdown()` 返回值替代重复过滤
   - `startTraining()` — 训练前自动同步前端任务类型到服务端，避免任务类型不匹配导致模型 `is_time_series` 字段错误

3. **`routes/projects.py`**
   - `activate_project()` 响应中模型字段补充 `is_time_series` 和 `task_type`（之前遗漏导致前端过滤永远返回空）

### 验证

```
TS Project: rnn: is_time_series=True → time_series filter: 1 | general filter: 0
General Project: mlp: is_time_series=False → time_series filter: 0 | general filter: 1
```

### 修复的 Bug

| Bug | 根因 | 修复 |
|-----|------|------|
| 项目激活后下拉框始终为空 | `activate_project()` 响应缺 `is_time_series` 字段 | 补充 `is_time_series` + `task_type` |
| 切换任务类型后训练，下拉框消失 | 任务类型未同步到服务端，模型 `is_time_series` 错误 | `startTraining()` 训练前自动同步 task_config |
| 过滤为空时下拉框完全隐藏 | `display: none` 导致用户误以为功能丢失 | 显示占位消息 + 保留下拉框可见 |

### 涉及文件

- `static/js/ui.js` — filterModelsByTask + populateModelDropdown 重构
- `static/js/app.js` — activateProject/refreshModelDropdown/startTraining 三处修改
- `routes/projects.py` — activate_project 响应补充字段

---

## 2026-06-15: 新增模型 — Crossformer (Two-Stage Attention) (feat/informer-integration 分支)

### 集成概述

从 `time_series_models_labs` 移植 Crossformer 模型，遵循 Autoformer/Informer 的 `pipeline="large"` 模式。

### 架构

Crossformer 的核心创新是 **Two-Stage Attention (TSA)**：
1. **Cross-Time Stage**: 在每个特征维度内部分别做时间维度的 FullAttention
2. **Cross-Dimension Stage**: 通过一组可学习的 router vectors 在不同特征维度间交换信息

Encoder 使用 `scale_block` + `SegMerging` 实现多尺度层次化表示，Decoder 使用 TSA self-attention + cross-attention 逐层生成预测。

### 涉及文件

| 文件 | 变更 |
|------|------|
| `utils/models/crossformer.py` | **新建** — CrossformerWrapper (pipeline="large")，含 output_proj 投影 enc_in→1 |
| `utils/models/crossformer_layers/__init__.py` | **新建** — 包导出 |
| `utils/models/crossformer_layers/Embed.py` | **新建** — PatchEmbedding, PositionalEmbedding（自包含） |
| `utils/models/crossformer_layers/SelfAttention_Family.py` | **新建** — FullAttention, AttentionLayer, TwoStageAttentionLayer |
| `utils/models/crossformer_layers/Crossformer_EncDec.py` | **新建** — SegMerging, scale_block, Encoder, Decoder, DecoderLayer |
| `utils/models/__init__.py` | **修改** — 注册 CrossformerWrapper 到 MODEL_REGISTRY |
| `utils/config.py` | **修改** — 添加 crossformer 默认超参 |
| `templates/index.html` | **修改** — 添加模型选项和配置面板 |
| `static/js/app.js` | **修改** — 添加 crossformer 到 allOptions/tsModels/startTraining |
| `static/js/ui.js` | **修改** — toggleModelParams 添加 crossformerParams |
| `static/dependency_graph.html` | **修改** — 添加 Crossformer 节点和边 |

### 关键设计决策

1. **输出投影**: Crossformer 原生输出 (batch, pred_len, enc_in) 所有特征维度，通过 `nn.Linear(enc_in, 1)` 投影为 (batch, pred_len, 1)，与 LargePipelineStrategy 兼容。
2. **自包含包**: `crossformer_layers/` 独立复制所有依赖层（Follow Autoformer 模式），避免与 Informer/shared_layers 耦合。
3. **超参暴露**: 暴露 `seg_len` 和 `win_size` 给用户，默认值 12/2 与原文一致。`d_ff` 默认 32 与 Autoformer/Informer 对齐。

### 验证

```
模型实例化: ✓ (input_dim=5, output_dim=12, seq_len=48)
前向传播:   ✓ → (4, 12, 1)
训练流水线:  ✓ (epochs=3, batch=8, loss 正常下降)
预测:       ✓ → (10, 12)
保存/加载:  ✓ (state_dict 一致)
测试套件:   ✓ (160 passed)
```

### 后续修复

| 提交 | 问题 | 修复 |
|------|------|------|
| `4413796` | Crossformer 缺少 `activation` 参数，`TwoStageAttentionLayer` 和 `DecoderLayer` 始终使用 GELU 无法切换；seg_len/win_size 无输入校验 | 统一 activation 为超参（支持 relu/gelu），前后端注册表同步；`_setup_training()` 新增 seg_len 除数校验和 win_size ≤ e_layers 校验 |
| `520ea76` | large pipeline 未要求 Time Series 模式，非时序任务中 `split_data` 不生成 `x_mark_train/dec_inp/y_mark`，`LargePipelineStrategy` 报 `ValueError` | `_setup_training()` 在 `split_data` 前检查 `pipeline=="large"` 且非 time_series 时提前抛出 `RouteError`，引导用户切换任务类型 |

### 三层防御链（避免用户跨模型搭配）

| 层级 | 位置 | 机制 |
|------|------|------|
| ① 前端过滤 | `app.js: tsModels` + `updateModelOptions()` | 时序任务只显示 `pipeline="large"` 模型，非时序任务隐藏 |
| ② 前端同步 | `app.js: startTraining()` | 训练前自动同步任务类型到服务端 task_config |
| ③ 后端校验 | `training.py: _setup_training()` | `pipeline=="large"` + 非 time_series 时抛 `RouteError` |

---

## 2026-06-16: 新增模型 — DLinear (Decomposition Linear) (feat/informer-integration 分支)

### 集成概述

从 `time_series_models_labs` 移植 DLinear 模型。DLinear 是 Areal 团队提出的极简线性模型（与 Autoformer 同一团队），核心创新是使用 series decomposition 将时间序列分解为 seasonal + trend 分量，每个分量独立通过 Linear 层映射 seq_len → pred_len。

### 架构

```
x (batch, seq_len, enc_in)
    ↓
series_decomp ──→ seasonal (batch, seq_len, enc_in)
│                    ↓ permute(0,2,1)
│                    ↓ Linear(seq_len, pred_len)
│                    ↓ (batch, enc_in, pred_len)
│
└──→ trend (batch, seq_len, enc_in)
                     ↓ permute(0,2,1)
                     ↓ Linear(seq_len, pred_len)
                     ↓ (batch, enc_in, pred_len)
                         ↓ sum
                    (batch, enc_in, pred_len)
                         ↓ permute(0,2,1)
                    (batch, pred_len, enc_in)
                         ↓ output_proj enc_in→1
                    (batch, pred_len, 1)
                         ↓ squeeze
                    (batch, pred_len)
```

### Pipeline 决策

与 Autoformer/Informer/Crossformer（large pipeline）不同，DLinear 使用 **small pipeline**：

| 维度 | Large Pipeline 模型 | DLinear |
|------|-------------------|---------|
| forward | `(x_enc, x_mark, x_dec, x_mark)` | `(x)` |
| 时间特征 | 需要 | 不需要 |
| PipelineData | 必须 | 不需要 |
| 依赖层 | attention/embedding/encoder/decoder | 仅 series_decomp（30行内联） |
| 输出 | `(batch, pred_len, 1)` → format squeeze | 直接 `(batch, pred_len)` |

### 设计决策

| 决策 | 选择 | 理由 |
|------|------|------|
| series_decomp 来源 | 内联在 `dlinear.py` | 仅 30 行，不值得独立包或跨包引用 |
| 参数暴露 | `moving_avg` + `individual` | DLinear 原始参数就这两个 |
| Dropout | 不暴露 | 原始 DLinear 无 dropout，线性层加 dropout 意义不大 |
| Weight 初始化 | `(1/seq_len) × ones(pred_len, seq_len)` | 论文核心设计——从均值预测开始训练 |

### 涉及文件

| 文件 | 变更 |
|------|------|
| `utils/models/dlinear.py` | **新建** — DLinearWrapper (pipeline="small") + 内联 _MovingAvg / _SeriesDecomp + _RawDLinear |
| `utils/models/__init__.py` | **修改** — 注册 DLinearWrapper，参数 moving_avg + individual |
| `utils/config.py` | **修改** — MODEL 字典添加 dlinear 默认超参（moving_avg=25, individual=False） |
| `templates/index.html` | **修改** — 添加 DLinear 选项和参数面板（moving_avg + individual toggle） |
| `static/js/app.js` | **修改** — allOptions 添加 DLinear；tsModels 添加；startTraining 参数收集 |
| `static/js/ui.js` | **修改** — toggleModelParams 添加 dlinearParams |
| `static/dependency_graph.html` | **修改** — 添加 DLinear 节点和继承边 |

### 验证

| 功能 | 状态 |
|------|------|
| 模型实例化 (input_dim=5, output_dim=12, seq_len=48) | ✓ |
| 前向传播 shared | ✓ → (4, 12) |
| 前向传播 individual | ✓ → (4, 12) |
| 注册表查询 | ✓ pipeline="small", params=moving_avg+individual |
| 测试套件 | ✓ 160 passed |

### Bug: DLinear 训练时报 shape mismatch (mat1 and mat2 cannot be multiplied)

**症状**: 训练 DLinear 时崩溃：
```
RuntimeError: mat1 and mat2 shapes cannot be multiplied (32x10 and 96x5)
```

**根因**: `DLinearWrapper.__init__` 中 `seq_len` 默认值为 96（硬编码），但 small pipeline 的 TS 数据 `seq_len` 为 10（来自 `config.TIME_SERIES.seq_len`）。`SmallPipelineStrategy.extra_model_kwargs()` 返回空 dict，不传递 `seq_len`，导致模型 Linear 层用 `nn.Linear(96, pred_len)` 初始化，但实际输入是 `(batch, 10, enc_in)`。

**涉及路径**:
| 路径 | 问题 | 修复 |
|------|------|------|
| `_run_and_persist()` 训练创建模型 | `pd_train=None` 导致 `extra_model_kwargs` 返回 `{}` | 从 `split_result` 补充 `seq_len` |
| `_reconstruct_model()` 项目重建 | 虽构建 `PipelineData(seq_len=...)`，但 `SmallPipelineStrategy` 忽略 | `SmallPipelineStrategy.extra_model_kwargs()` 返回 `pd.seq_len` |

**修复**:
1. `routes/training.py:_run_and_persist()` — 模型创建前检查 `extra_kw` 是否有 `seq_len`，没有则从 `split_result` 补充
2. `utils/pipeline_strategy.py:SmallPipelineStrategy.extra_model_kwargs()` — 返回 `pd` 中的 `seq_len` 和 `label_len`（修复项目重建路径）

**验证**: 手动复现脚本测试通过。160 测试全部通过。

---

### Bug: 项目激活时 DLinear state_dict size mismatch

**症状**: 激活含 DLinear 模型的项目时报错：
```
Failed to reconstruct model: Error(s) in loading state_dict for DLinearWrapper:
size mismatch for _model.Linear_Seasonal.0.weight:
  copying a param with shape torch.Size([5,10]) from checkpoint,
  the shape in current model is torch.Size([5,96]).
```

**根因**: `_reconstruct_model()` (routes/projects.py:42) 中 PipelineData 创建和 `extra_model_kwargs()` 调用被包裹在 `if meta.get("pipeline") == "large":` 条件块内。DLinear 的 pipeline="small"，跳过该块，`model_kw` 永远不含 `seq_len`，模型用默认值 `seq_len=96` 初始化。checkpoint 保存的是实际训练时的 `seq_len=10`，导致 `load_state_dict` 时 Linear 层 weight shape 不匹配。

**修复**: 将条件从 `if meta.get("pipeline") == "large":` 改为 `if meta.get("is_time_series"):`，使所有时序模型（无论 pipeline 类型）都能正确获取 `seq_len`/`label_len`。`label_len` 降级：large pipeline 默认 `seq_len//2`，small pipeline 默认 `0`。

---

## 2026-06-16: 架构加深优化 — PipelineData.from_split + SplitResult 数据类 + 归一化去重 + 基础设施测试 (feat/informer-integration 分支)

### 概述

基于 `/improve-codebase-architecture` 分析结果，实施 4 个架构加深候选：

| 候选 | 目标 | 涉及文件 |
|------|------|---------|
| ① `PipelineData.from_split()` 工厂 | 消除 8 个调用点的手动构造重复 | `pipeline_strategy.py`, `routes/training.py`, `routes/evaluation.py` |
| ② `normalize_target()` 支持预计算参数 | 消除 `_setup_training()` 35 行内联归一化 | `data_utils.py`, `routes/training.py` |
| ③ `SplitResult` 类型化数据类 | 消除 split_result 隐式 dict 契约（30+ key） | `data_utils.py`, `project_manager.py` |
| ④ 基础设施单元测试 | PipelineStrategy + ProjectManager + SplitResult 覆盖 | 3 个新测试文件 |

### 候选 1: PipelineData.from_split() 工厂方法

**`utils/pipeline_strategy.py`** — 新增 `@classmethod from_split(cls, split_result, subset)`：

```python
@classmethod
def from_split(cls, split_result: dict, subset: str = "train") -> "PipelineData":
    suffix = f"_{subset}"
    key = f"x_mark{suffix}"
    if key not in split_result:
        return cls()
    return cls(
        X_mark=split_result[f"x_mark{suffix}"],
        dec_inp=split_result[f"dec_inp{suffix}"],
        y_mark=split_result[f"y_mark{suffix}"],
        n_time_features=split_result.get("n_time_features", 4),
        seq_len=split_result.get("seq_len", 96),
        label_len=split_result.get("label_len", 0),
    )
```

key 不存在时返回空的 `PipelineData()`（全字段 None），而非返回 None。调用者无需 `if/else`。

**替换的 6 处调用点**:

| 位置 | 旧代码 (行数) | 新代码 |
|------|---------------|--------|
| `routes/training.py:_run_and_persist` pd_train | 14 行 if-block | `PipelineData.from_split(split_result, "train")` |
| `routes/training.py:_run_and_persist` pd_val | 6 行 | `PipelineData.from_split(split_result, "test")` |
| `routes/evaluation.py:api_evaluate` | 4 行 ternary | `PipelineData.from_split(split_result, "test")` |
| `routes/evaluation.py:_compute_predictions` test | 4 行 ternary | `PipelineData.from_split(split_result, "test")` |
| `routes/evaluation.py:_compute_predictions` train | 4 行 ternary | `PipelineData.from_split(split_result, "train")` |
| `routes/evaluation.py:api_validate` | 8 行 if-block | **保留显式构造**（cross_validate_model 使用 None 作为 fold 数据切片哨兵值） |

### 候选 2: normalize_target 支持预计算参数

**`utils/data_utils.py`** — `normalize_target()` 新增可选 `norm_params` 和 `target_idx` 参数：

```python
def normalize_target(y_train, y_test, method="mean",
                     norm_params=None, target_idx=None):
    if norm_params is not None and target_idx is not None:
        # Extract target statistics from precomputed norm_params
        if method == "minmax":
            t_min = np.array(norm_params["min"])[target_idx]
            t_max = np.array(norm_params["max"])[target_idx]
            ...
```

**`routes/training.py`** — `_setup_training()` 中 35 行内联归一化块替换为：

```python
if split_result["task_type"] == "regression":
    if pipeline == "large" and norm_method in ("minmax", "mean"):
        y_norm_result = normalize_target(
            split_result["y_train"], split_result["y_test"],
            method=norm_method,
            norm_params=norm_params,
            target_idx=split_result.get("target_idx", -1),
        )
    else:
        ...
```

### 候选 3: SplitResult 类型化数据类

**`utils/data_utils.py`** — 新增 `SplitResult` 数据类（26 字段）：

```python
@dataclass
class SplitResult:
    X_train: np.ndarray; X_test: np.ndarray
    y_train: np.ndarray; y_test: np.ndarray
    feature_names: list; target_name: str; task_type: str; n_classes: int
    target_encoder: Optional[object] = None; input_dim: int = 0
    is_time_series: bool = False; seq_len: int = 0; pred_len: int = 0
    label_len: int = 0; time_col: str = ""; time_granularity: str = ""
    time_encoding_features: list = field(default_factory=list)
    n_time_features: int = 0
    x_mark_train: Optional[np.ndarray] = None; dec_inp_train: Optional[np.ndarray] = None
    y_mark_train: Optional[np.ndarray] = None; x_mark_test: Optional[np.ndarray] = None
    dec_inp_test: Optional[np.ndarray] = None; y_mark_test: Optional[np.ndarray] = None
    target_idx: int = -1; y_scaler: Optional[dict] = None
```

向后兼容设计：
- `__getitem__` / `__setitem__` — 属性访问伪装成 dict 访问
- `__contains__` — 可选字段（default=None）只在值不为 None 时返回 True，匹配旧 dict 语义
- `get()` / `items()` / `update()` — dict 兼容方法

**`utils/project_manager.py`** — `load_split()` 返回 `SplitResult(**data)`；`save_split()` 用 `dataclasses.asdict()` 序列化。

### 候选 4: 基础设施单元测试

| 测试文件 | 测试数 | 覆盖内容 |
|----------|--------|---------|
| `tests/test_pipeline_strategy.py` | 22 | PipelineData.from_split 4 场景 + SmallPipelineStrategy 9 方法 + LargePipelineStrategy 7 方法 |
| `tests/test_project_manager.py` | 7 | SplitResult 序列化往返（通用/时序/y_scaler/encoder/None/空） |
| `tests/test_split_result.py` | 10 | 数据类 dict 兼容（getitem/setitem/get/contains/items/update） |

新增 41 测试，原有 159 测试全部通过，总计 200 测试。

### 验证

```
200 passed in 131.19s (0:02:11)
```

---

## Prior Issues (前序会话已解决)

- NaN JSON 序列化：`clean_nan()` 递归转换
- CSV 中文编码：增加 gbk/gb2312/gb18030 编码回退链
- ReduceLROnPlateau 参数：移除 `verbose`
- 混淆矩阵 thresh：`max(cm)` → `max(max(row) for row in cm)`
- 空 DataFrame 清理保护：至少保留 1 列，IQR 跳过小数据集
- Toggle CSS 重叠：`display: inline-block` + `box-sizing: content-box`
- 中文字体自动检测：`utils/fonts.py`
- 会话/缓存失联：固定 `secret_key` + `get_data()` 磁盘回退
