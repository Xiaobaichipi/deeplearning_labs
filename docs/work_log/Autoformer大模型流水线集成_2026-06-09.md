# Autoformer 大模型流水线集成

**日期**: 2026-06-09
**分支**: feat/large-model-pipeline

---

## 工作任务

1. **双流水线架构设计** — 将模型分为 small/large 两种 pipeline：
   - **small**: 传统小模型（MLP/CNN/RNN/LSTM/GRU/Transformer），forward 签名 `model(X)` → output
   - **large**: 大模型（Autoformer/Informer/Transformer 长序列版），forward 签名 `model(x_enc, x_mark_enc, x_dec, x_mark_dec)` → output
   - pipeline 由模型注册表中的 `pipeline` 字段驱动，路由层不感知具体模型类型

2. **Autoformer 模型集成** — 作为首个 large pipeline 模型样板：
   - 从 `time_series_models_labs/models/Autoformer.py` 适配核心架构
   - 从 `time_series_models_labs/layers/` 引入 AutoCorrelation、Autoformer_EncDec、Embed 层
   - 修复趋势分量维度不匹配问题（`trend_proj` 线性投影）
   - 修复时间特征嵌入维度问题（自定义 `_DataEmbeddingWoPos` 替代硬编码 freq → dim 映射）

3. **数据处理流水线改造** — 新增 `_create_large_windows()`，支持带时间特征的滑动窗口数据构建

4. **训练/评估/预测/交叉验证全链路适配** — 所有路由根据 pipeline 类型分发数据

5. **模型持久化** — 保存 pipeline 字段到 meta，确保反序列化时能正确重建

---

## 修改文件清单

### 1. `utils/models/base.py` — 基础模型类

| 行号 | 修改 |
|------|------|
| L10 | 新增 `pipeline = "small"` 类属性，所有小模型默认 small |

### 2. `utils/models/__init__.py` — 模型注册表

| 行号 | 修改 |
|------|------|
| L5-9 | 每条注册记录新增 `pipeline` 字段（"small"/"large"） |
| L12 | 注册 AutoformerWrapper，pipeline="large" |
| L14-27 | Autoformer 超参数 schema：d_model, n_heads, e_layers, d_layers, d_ff, moving_avg, factor, dropout, activation |
| L33 | 新增 `get_model_pipeline(model_type)` 函数 |
| L37 | 新增 `get_large_model_types()` 函数 |

### 3. `utils/models/autoformer.py` — Autoformer 封装（新文件）

| 行号 | 内容 |
|------|------|
| L1-85 | `_DataEmbeddingWoPos`：自定义嵌入层，用 `nn.Linear(n_time_features, d_model)` 替代硬编码 freq → dim 映射 |
| L87-170 | `_RawAutoformer`：从参考代码适配，含 `trend_proj` 修复 |
| L172-200 | `AutoformerWrapper(BaseModel)`：pipeline="large"，兼容现有训练框架 |

**关键修复**：
- `trend_proj = nn.Linear(enc_in, c_out)` — trend_init 从 enc_in 维投影到 c_out=1 维
- `self.label_len` 缓存，用于 dec_inp 截取

### 4. `utils/models/autoformer_layers/` — Autoformer 依赖层（新目录）

| 文件 | 来源 |
|------|------|
| `__init__.py` | 包初始化 |
| `Embed.py` | 时间特征、值、位置嵌入 |
| `AutoCorrelation.py` | Autoformer 核心的 AutoCorrelation 机制 + 时延聚合 |
| `Autoformer_EncDec.py` | Encoder、Decoder、series_decomp 模块 |

### 5. `utils/config.py` — 默认配置

| 行号 | 修改 |
|------|------|
| L93-95 | 新增 `MODEL["autoformer"]` 条目：d_model=256, n_heads=8, e_layers=3, d_layers=3, d_ff=32, moving_avg=25, factor=3, dropout=0.1, activation="gelu" |

### 6. `utils/data_utils.py` — 数据处理

| 行号 | 修改 |
|------|------|
| L30+ | 新增 `_create_large_windows()` — 返回 5 元组 (batch_x, batch_x_mark, dec_inp, y_mark, batch_y) |
| L80+ | `split_data()` 时间序列路径新增 large pipeline 分支：分离时间特征与值特征，目标列保留在 X_values 中 |
| L30+ | 新增 `normalize_data_apply()` — 用预计算参数对 dec_inp 应用归一化 |

### 7. `utils/model_utils.py` — 模型工具

| 行号 | 修改 |
|------|------|
| L20+ | `train_model()` 检查 `model.pipeline == "large"`，创建 5 元组 TensorDataset，单独前向循环 |
| L60+ | `predict()` large 分支调用 `model(X_t, xm_t, di_t, ym_t)`，squeeze 输出 |
| L80+ | `evaluate()` large 分支处理 4 参数前向，直接返回回归指标 |
| L100+ | `cross_validate_model()` 增加 kwargs 透传 large 数据 |

### 8. `routes/training.py` — 训练路由

| 行号 | 修改 |
|------|------|
| L186-200 | `_run_and_persist()` 增加 `extra_model_kw`（n_time_features, seq_len, label_len）传给 create_model |
| L208-217 | 透传 `large_kw`（X_mark, dec_inp, y_mark）到 train_model |
| L283+ | meta 保存 `pipeline` 字段 |
| L292-347 | `_setup_training()` 导入 `get_model_pipeline()`，传 pipeline 给 split_data，归一化 dec_inp |

### 9. `routes/evaluation.py` — 评估/预测/验证路由

| 行号 | 修改 |
|------|------|
| L20+ | evaluate/predict/validate 三个路由均检查 `"x_mark_test" in split_result`，透传 large 数据 |

### 10. `routes/projects.py` — 项目管理路由

| 行号 | 修改 |
|------|------|
| L40+ | `_reconstruct_model()` 从 meta 读取 seq_len/label_len 传给大模型 |
| L60+ | `compare_models()` 检查 split_result，传 large 数据到 predict() |

### 11. `templates/index.html` — 前端模板

| 行号 | 修改 |
|------|------|
| L120 | modelType 下拉菜单新增 `<option value="autoformer">` |
| L200-230 | 新增 `autoformerParams` 区域：d_model, n_heads, e_layers, d_layers, d_ff, moving_avg, factor, activation 控件 |

### 12. `static/js/ui.js` — 前端交互

| 行号 | 修改 |
|------|------|
| L50+ | `toggleModelParams()` 新增 autoformerParams 显示/隐藏逻辑 |

### 13. `static/js/app.js` — 前端编排

| 行号 | 修改 |
|------|------|
| L241-262 | `startTraining()` 新增 Autoformer 参数读取：d_model, n_heads, e_layers, d_layers, d_ff, moving_avg, factor, activation |

---

## 关键问题与解决方案

### 问题 1：Autoformer 输出维度为 (batch, pred_len, 5) 而非 (batch, pred_len, 1)

**原因**：`series_decomp(x_enc)` 产生的 trend_init 具有 enc_in=5 维。DecoderLayer 通过 Conv1d(d_model→c_out=1) 产生 residual_trend。PyTorch 广播使 `trend = trend_init + residual_trend` 得到 enc_in=5 维。

**修复**：在 `_RawAutoformer.__init__` 中添加 `self.trend_proj = nn.Linear(configs.enc_in, configs.c_out)`，将 trend_init 投影到 c_out 维度。

### 问题 2：时间特征维度不匹配

**原因**：参考代码的 `TimeFeatureEmbedding` 硬编码 `freq_map = {'h': 4}`，使用 `nn.Linear(4, d_model)`。但我们的 `time_encoding()` 每小时粒度产生 5 个特征（year, month, day, weekday, hour）。

**修复**：创建自定义 `_DataEmbeddingWoPos` 类，用 `nn.Linear(n_time_features, d_model)` 替代硬编码映射。n_time_features 从 split_result 获取并通过 kwargs 传入。

### 问题 3：数据流不兼容

**原因**：现有代码假设 `model(X)` 单参数前向；Autoformer 需要 4 参数前向。

**修复**：引入 `model.pipeline` 属性区分。`train_model()`、`predict()`、`evaluate()` 检查此属性，选择对应的数据组装和前向逻辑。分裂路由不直接检查模型类型。

---

## 测试结果

- Python 集成测试通过：581 训练窗口，loss 递减 (0.9078→0.4402→0.2271)，预测 shape (101, 12)
- `assert out.shape == (2, 12, 1)` 通过
- curl 端到端服务器测试：需使用 `/api/data/task-config`（**非** `/api/task-config`）

---

## 第二次验证（2026-06-09 端到端全链路测试）

### 测试场景
- 数据集：800 行合成小时级数据（feature1, feature2, target），含 datetime 列
- 模型：Autoformer（d_model=256, n_heads=8, e_layers=3, d_layers=3, d_ff=32, pred_len=12, seq_len=48, label_len=24）
- 训练：5 epochs via SSE 流

### 通过的功能

| 功能 | 结果 | 说明 |
|------|------|------|
| 项目创建 + CSV 上传 | ✅ | 正确创建并解析 800 行数据集 |
| 任务配置（time_series） | ✅ | seq_len=48, pred_len=12, label_len=24, granularity=h |
| SSE 训练流 | ✅ | train_loss 16.43→2.82，avg_epoch_time 11.75s |
| 评估（evaluate） | ✅ | MSE/R2 等回归指标 + 可视化图表 |
| 预测（predict） | ✅ | 返回 101 条预测结果 + 折线图 |
| 交叉验证（validate） | ✅ | 2-fold CV 完成（原为 bug） |
| 项目激活 + 模型重建 | ✅ | 反序列化后评估结果与训练后一致 |

### 验证中修复的 Bug

| Bug | 文件 | 修复 |
|-----|------|------|
| CV 创建 Autoformer 时缺少 `n_time_features` → 维度错误 `(1536x5 and 4x256)` | `routes/evaluation.py:198-220`, `utils/model_utils.py:282` | 从 split_result 传入 `extra_model_kw`（含 n_time_features, seq_len, label_len）到 `cross_validate_model`，再传给 `create_model` |
| `_reconstruct_model` 缺少 `n_time_features` → 项目激活后模型重建失败 | `routes/projects.py:22-25` | 添加 `model_kw["n_time_features"] = meta.get("n_time_features", 4)` |
| 模型 meta 未持久化 `n_time_features` | `routes/training.py:283` | 在 meta 字典中添加 `"n_time_features"` 条目 |

---

## 待办事项

- [ ] Step 6 模型选择器下拉菜单（前端交互）
- [ ] 提交所有变更到 feat/large-model-pipeline 分支
- [ ] 更新 ISSUES.md + 重新生成依赖图
