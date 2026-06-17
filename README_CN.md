# DeepLearning Labs

> 基于浏览器的表格数据与时序数据深度学习实验平台。上传、探索、清洗、训练、评估、预测 —— 全在浏览器中完成，后端由 PyTorch 驱动。

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c)](https://pytorch.org/)
[![Flask](https://img.shields.io/badge/Flask-3.0%2B-black)](https://flask.palletsprojects.com/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

---

## 功能特性

- **📤 上传与探索** — 拖拽导入 CSV/Excel，自动检测编码；数据预览、列信息、统计信息、分布直方图、相关性热力图
- **🧹 清洗与填充** — 去重、IQR 离群值处理、多种缺失值填充策略（均值/中位数/众数/前向填充/后向填充/常量）
- **🧠 多架构模型** — 14 种架构，覆盖通用表格分类/回归（MLP、CNN、RNN、LSTM、GRU、Tabular Transformer）与时序预测（Autoformer、Informer、Crossformer、DLinear、ETSformer、FEDformer、FiLM、Vanilla Transformer），全部支持超参配置
- **📊 实时训练** — SSE 流式传输每 epoch 进度，实时 Loss/Metric 曲线图（Chart.js），早停机制 + 学习率调度
- **📈 评估与可视化** — 回归指标（MSE/RMSE/MAE/R²）、分类指标（准确率/精确率/召回率/F1）、混淆矩阵、ROC 曲线、残差分布图
- **🔁 交叉验证** — K 折 CV 使用与训练相同的模型架构
- **🔮 预测与导出** — 散点图 + 折线图对比；结果导出为 CSV 或 Excel
- **🔁 多模型对比** — 选择多个已训练模型并排比较预测结果
- **⏱ 时序预测** — 可配置序列长度/预测长度/标记长度、时间列选择、时间粒度检测、时间特征编码
- **💾 项目系统** — 持久化项目存储，数据集/模型跨会话版本管理
- **⚡ 设备选择** — 支持 CPU / GPU / 多 GPU DataParallel 训练，UI 端直接选择
- **🧩 模型导出** — 下载训练好的模型权重供外部使用
- **🔌 可扩展** — 简单模型注册机制 —— 添加新架构只需一个文件 + 一条注册项
- **双管道系统** — small pipeline（2 参数 forward）用于简单模型，large pipeline（4 参数 forward）用于带时间标记编码的时序模型

---

## 快速开始

### 环境要求

- Python 3.10+
- pip

### 安装与运行

```bash
# 克隆或下载项目
cd deeplearning_labs

# 安装依赖（CPU 版 PyTorch 即可）
pip install -r requirements.txt

# 启动服务
python main.py
```

打开浏览器访问 **http://localhost:5000**。

> **GPU 加速**：如拥有 CUDA 显卡，可安装 CUDA 版 PyTorch 加速训练：
> ```bash
> pip install torch --index-url https://download.pytorch.org/whl/cu118
> ```

### 使用启动脚本

```bash
./start.sh              # 默认端口 5000 启动
./start.sh 8080         # 自定义端口启动
./start.sh stop         # 停止服务
./start.sh status       # 查看运行状态
```

---

## 使用向导

界面采用 7 步引导流程（含项目创建）：

| 步骤 | 操作 | 结果 |
|---|---|---|
| **0. 项目** | 创建或激活项目 | 持久化数据集/模型跨会话存储 |
| **1. 上传** | 拖拽或点击上传 CSV/XLSX | 自动检测编码，数据加载到项目 |
| **2. 探索** | 浏览各个标签页 | 数据预览、列信息、统计、分布图、相关性图 |
| **3. 清洗填充** | 勾选清洗选项、选择填充策略 | 去重、离群值修剪、缺失值填充 |
| **4. 模型配置** | 选择任务类型（通用/时序）、模型架构与超参数 | 14 种模型选型 + CPU/GPU 选择 + 归一化配置 + 时序参数 |
| **5. 训练** | 点击"Start Training" | 实时进度条 + Loss/Metric 曲线、早停、训练历史图 |
| **6. 评估预测** | 运行评估/交叉验证/预测 | 指标、可视化图表、多模型对比图、CSV/XLSX 下载 |

---

## 支持的模型

| 模型 | 类型 | 管道 | 描述 | 关键参数 |
|---|---|---|---|---|
| **MLP** | 通用 | small | 全连接前馈网络 | 隐藏层结构、dropout |
| **CNN 1D** | 通用 | small | 一维卷积网络 | 通道数、卷积核大小、dropout |
| **RNN** | 通用+时序 | small | 标准循环神经网络 | 隐藏层大小、层数、双向、dropout |
| **LSTM** | 通用+时序 | small | 长短期记忆网络 | 隐藏层大小、层数、双向、dropout |
| **GRU** | 通用+时序 | small | 门控循环单元网络 | 隐藏层大小、层数、双向、dropout |
| **Transformer (Tabular)** | 通用 | small | Transformer 编码器（分类/回归用） | d_model、注意力头数、层数、dropout |
| **Vanilla Transformer** | 时序 | large | 完整 Encoder-Decoder + DataEmbedding | d_model, n_heads, e_layers, d_layers, d_ff, dropout, activation |
| **Autoformer** | 时序 | large | 分解架构 + Auto-Correlation 机制 | d_model, n_heads, e_layers, d_layers, d_ff, moving_avg, factor, dropout, activation |
| **Informer** | 时序 | large | ProbSparse 自注意力长序列预测 | d_model, n_heads, e_layers, d_layers, d_ff, factor, distil, dropout, activation |
| **Crossformer** | 时序 | large | 两阶段注意力（DSW + DMS）+ 分段嵌入 | d_model, n_heads, e_layers, d_ff, factor, seg_len, win_size, dropout, activation |
| **ETSformer** | 时序 | large | 指数平滑 + 傅里叶频率注意力 | d_model, n_heads, e_layers, d_ff, top_k, dropout, activation |
| **FEDformer** | 时序 | large | 频域增强分解 Transformer（傅里叶/小波） | d_model, n_heads, e_layers, d_layers, d_ff, moving_avg, modes, version, mode_select, dropout, activation |
| **FiLM** | 时序 | large | 频域增强 Legendre 记忆 + HiPPO-LegT + SpectralConv1d | window_size, multiscale, dropout |
| **DLinear** | 时序 | small | 分解线性模型 + 序列分解 | moving_avg, individual |

添加新模型请参考[模型扩展指南](templates/models_guide.html)。

---

## 项目架构

```
deeplearning_labs/
├── main.py                     # Flask 入口
├── routes/
│   ├── data.py                 # 上传、清洗、填充端点
│   ├── training.py             # 训练（同步 + SSE 流）
│   ├── evaluation.py           # 评估、预测、交叉验证、下载
│   └── projects.py             # 项目 CRUD、模型管理
├── utils/
│   ├── model_utils.py          # 训练循环、推理、CV、评估
│   ├── data_utils.py           # 加载、拆分、归一化、清洗、填充
│   ├── plot_utils.py           # Matplotlib → base64 PNG
│   ├── session.py              # SessionManager（状态缓存）
│   ├── project_manager.py      # 项目持久化（磁盘 I/O）
│   ├── pipeline_strategy.py    # Small/Large 管道调度器
│   ├── config.py               # 集中化默认配置
│   ├── fonts.py                # 中文字体检测
│   └── models/                 # 模型注册（14 种架构）
│       ├── base.py             # 抽象 BaseModel
│       ├── mlp.py, cnn.py, rnn.py, lstm.py, gru.py, transformer.py
│       ├── vanilla_transformer.py, autoformer.py, informer.py
│       ├── crossformer.py, dlinear.py, etsformer.py
│       ├── fedformer.py, film.py
│       ├── shared_layers/      # 共享组件（Embed、EncDec、Attention）
│       ├── autoformer_layers/  # Autoformer 内部包
│       ├── informer_layers/    # Informer 内部包
│       ├── crossformer_layers/ # Crossformer 内部包
│       ├── etsformer_layers/   # ETSformer 内部包
│       ├── fedformer_layers/   # FEDformer 内部包
│       └── film_layers/        # FiLM 内部包
├── static/
│   ├── js/app.js               # 事件绑定与初始化
│   ├── js/api.js               # API 调用函数
│   ├── js/ui.js                # DOM 渲染与 Chart.js
│   ├── js/__tests__/           # Vitest 前端测试
│   ├── css/style.css           # 全局样式
│   └── dependency_graph.html   # 代码依赖关系图（Mermaid）
├── templates/
│   ├── index.html              # 主应用界面
│   └── models_guide.html       # 模型扩展文档
├── tests/                      # Python 测试套件（pytest）
│   ├── test_routes.py
│   ├── test_model_utils.py
│   ├── test_data_utils.py
│   ├── test_session.py
│   ├── test_plot_utils.py
│   ├── test_pipeline_strategy.py
│   ├── test_project_manager.py
│   └── test_split_result.py
├── uploads/                    # 上传数据（按会话隔离）
├── outputs/                    # 生成输出
├── projects/                   # 持久化项目存储
├── docs/
│   ├── PRD.md                  # 产品需求文档
│   └── adr/                    # 架构决策记录
├── start.sh                    # 启停脚本
├── requirements.txt            # Python 依赖
├── vitest.config.js            # Vitest 配置
└── vitest.setup.js             # Vitest jsdom 环境配置
```

### 关键设计决策

- **Flask Blueprint 路由组织** — data、training、evaluation、projects 分离为独立模块
- **SSE（Server-Sent Events）** 实现实时训练进度 — 比 WebSocket 更简单，Flask 原生支持
- **SessionManager 内存缓存 + 磁盘回退** — Flask debug 重载后自动恢复数据
- **模型注册机制** — 添加模型 = 一个文件 + 一条注册项，无需修改其他代码
- **双管道系统** — small（2 参数 forward）用于简单模型，large（4 参数 forward）用于带时间标记编码的时序模型
- **共享层** — `shared_layers/` 中存放通用 Transformer 组件，被多个 large-pipeline 模型复用
- **Chart.js 实时图表** — `animation: false` 避免 SSE 高频更新时闪烁
- **matplotlib Agg 后端** — 服务端渲染为 base64 PNG，内嵌到页面中
- **Vitest 前端测试** — 54 个 JS 测试覆盖 UI 切换、参数采集、模型过滤等核心交互逻辑
- **SplitResult 数据类** — 类型安全的拆分结果，替代隐式 dict 契约

---

## 默认配置

超参默认值集中管理在 `utils/config.py`：

```python
TRAINING = {
    "test_size": 0.2, "learning_rate": 0.001,
    "batch_size": 32, "epochs": 50,
    "patience": 10, "dropout": 0.2,
    "normalization": "none",
}

MODEL = { ... }  # 各架构默认值（14 种模型）
TIME_SERIES = {"seq_len": 10, "pred_len": 1, "label_len": 0}
CV = {"default_folds": 5, "max_epochs_per_fold": 20}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
```

---

## 开发

```bash
# 安装开发依赖
pip install pytest pytest-cov
npm install          # Vitest 前端测试

# 运行 Python 测试（202 个）
pytest tests/

# 运行 JS 测试（54 个）
npx vitest run

# 启动调试模式（Flask debug，代码修改后自动重载）
python main.py
```

---

## 项目状态

在 `feat/informer-integration` 分支上活跃开发。202 个 Python 测试 + 54 个 Vitest JS 测试全部通过，覆盖训练、评估、预测、交叉验证、数据处理、项目管理和前端交互逻辑。完整变更日志见 [ISSUES.md](ISSUES.md)。

---

## 许可证

MIT
