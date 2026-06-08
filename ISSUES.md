# Issues Log

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

## Prior Issues (前序会话已解决)

- NaN JSON 序列化：`clean_nan()` 递归转换
- CSV 中文编码：增加 gbk/gb2312/gb18030 编码回退链
- ReduceLROnPlateau 参数：移除 `verbose`
- 混淆矩阵 thresh：`max(cm)` → `max(max(row) for row in cm)`
- 空 DataFrame 清理保护：至少保留 1 列，IQR 跳过小数据集
- Toggle CSS 重叠：`display: inline-block` + `box-sizing: content-box`
- 中文字体自动检测：`utils/fonts.py`
- 会话/缓存失联：固定 `secret_key` + `get_data()` 磁盘回退
