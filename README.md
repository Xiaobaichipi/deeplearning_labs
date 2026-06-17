# DeepLearning Labs

> A browser-based deep learning experiment platform for tabular and time-series data. Upload, explore, clean, train, evaluate, and predict — all from your browser, powered by PyTorch.

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c)](https://pytorch.org/)
[![Flask](https://img.shields.io/badge/Flask-3.0%2B-black)](https://flask.palletsprojects.com/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

---

## Features

- **📤 Upload & Explore** — Drag-and-drop CSV/Excel import with auto encoding detection, data preview, statistics, distribution histograms, and correlation heatmaps
- **🧹 Clean & Fill** — Remove duplicates, handle outliers (IQR), fill missing values with multiple strategies (mean, median, mode, ffill, bfill, constant)
- **🧠 Multi-Architecture Models** — 22 architectures covering general tabular (MLP, CNN, RNN, LSTM, GRU, Tabular Transformer, Random Forest, XGBoost, LightGBM, Decision Tree) and time-series forecasting (Autoformer, Informer, Crossformer, DLinear, ETSformer, FEDformer, FiLM, Vanilla Transformer) — all with configurable hyperparameters
- **📊 Real-Time Training** — SSE-streamed per-epoch progress with live Loss/Metric curves (Chart.js), early stopping, and LR scheduling
- **📈 Evaluation & Visualization** — Regression metrics (MSE/RMSE/MAE/R²), classification metrics (accuracy/precision/recall/F1), confusion matrix, ROC curve, residual plots
- **🔁 Cross-Validation** — K-fold CV using the same model architecture as training
- **🔮 Predict & Export** — Scatter + line comparison charts; download results as CSV or Excel
- **🔁 Multi-Model Comparison** — Select multiple trained models and compare predictions side-by-side
- **⏱ Time Series Support** — Configurable sequence/prediction/label lengths, time column selection, granularity detection, time feature encoding
- **💾 Project System** — Persistent project storage with dataset/model versioning across sessions
- **⚡ Device Selection** — CPU / GPU / multi-GPU DataParallel training, selectable from the UI
- **🧩 Model Export** — Download trained model state dicts for external use
- **🔌 Extensible** — Simple model registry — add new architectures by dropping in a file and one registry entry
- **Small/Large Pipeline** — Dual pipeline system: "small" for 2-argument models, "large" for 4-argument models with time-mark encoding

---

## Quick Start

### Prerequisites

- Python 3.10+
- pip

### Install & Run

```bash
# Clone or download the project
cd deeplearning_labs

# Install dependencies (CPU version of PyTorch is sufficient)
pip install -r requirements.txt

# Start the server
python main.py
```

The app will be available at **http://localhost:5000**.

> **GPU Support**: If you have a CUDA-capable GPU, install the CUDA version of PyTorch for faster training:
> ```bash
> # Choose the version matching your NVIDIA driver:
> # Driver 535+ → CUDA 12.1
> pip install torch --index-url https://download.pytorch.org/whl/cu121
> # Driver 525+ → CUDA 12.0
> pip install torch --index-url https://download.pytorch.org/whl/cu120
> ```

### Using the Startup Script

```bash
./start.sh          # Start on port 5000
./start.sh 8080     # Start on custom port
./start.sh stop     # Stop the server
./start.sh status   # Check server status
```

---

## Usage Walkthrough

The interface is organized as a 6-step wizard with an initial project setup:

| Step | What You Do | What You Get |
|---|---|---|
| **0. Projects** | Create or activate a project | Persistent dataset/model storage across sessions |
| **1. Upload** | Drop a CSV/XLSX file | Auto encoding detection, data loaded |
| **2. Explore** | Browse tabs | Data preview, column info, statistics, distribution plots, correlation heatmap |
| **3. Clean & Fill** | Toggle options | Duplicate removal, outlier clipping (IQR), missing value imputation |
| **4. Model Config** | Select task type (General / Time Series), model, device & params | Architecture choice from 22 models, CPU/GPU selection, hyperparameters, normalization |
| **5. Train** | Click "Start Training" | Real-time progress bar + live Loss/Metric charts, early stopping, LR scheduling |
| **6. Evaluate & Predict** | Run evaluation/cross-val/predict | Metrics, multi-model comparison, charts, CSV/XLSX download |

---

## Model Support

| Model | Type | Pipeline | Description | Key Parameters |
|---|---|---|---|---|
| **MLP** | General | small | Fully connected feedforward network | Hidden layers, dropout |
| **CNN 1D** | General | small | 1D convolutional network for tabular data | Channels, kernel size, dropout |
| **RNN** | Both | small | Vanilla recurrent neural network | Hidden size, layers, bidirection, dropout |
| **LSTM** | Both | small | Long short-term memory network | Hidden size, layers, bidirection, dropout |
| **GRU** | Both | small | Gated recurrent unit network | Hidden size, layers, bidirection, dropout |
| **Transformer (Tabular)** | General | small | Transformer encoder-only for tabular data | d_model, nhead, layers, dropout |
| **Vanilla Transformer** | Time Series | large | Full Encoder-Decoder Transformer with DataEmbedding | d_model, n_heads, e_layers, d_layers, d_ff, dropout, activation |
| **Autoformer** | Time Series | large | Decomposition architecture with Auto-Correlation | d_model, n_heads, e_layers, d_layers, d_ff, moving_avg, factor, dropout, activation |
| **Informer** | Time Series | large | ProbSparse self-attention for long sequence forecasting | d_model, n_heads, e_layers, d_layers, d_ff, factor, distil, dropout, activation |
| **Crossformer** | Time Series | large | Two-stage attention (DSW + DMS) with segment embedding | d_model, n_heads, e_layers, d_ff, factor, seg_len, win_size, dropout, activation |
| **ETSformer** | Time Series | large | Exponential smoothing with Fourier frequency attention | d_model, n_heads, e_layers, d_ff, top_k, dropout, activation |
| **FEDformer** | Time Series | large | Frequency enhanced decomposed Transformer (Fourier/Wavelets) | d_model, n_heads, e_layers, d_layers, d_ff, moving_avg, modes, version, mode_select, dropout, activation |
| **FiLM** | Time Series | large | Frequency-enhanced Legendre Memory with HiPPO-LegT + SpectralConv1d | window_size, multiscale, dropout |
| **DLinear** | Time Series | small | Decomposition linear model with series decomposition | moving_avg, individual |
| **Random Forest** | General | small | Ensemble of decision trees (regression + classification) | n_estimators, max_depth, min_samples_split, min_samples_leaf |
| **XGBoost** | General | small | Gradient boosted decision trees (regression + classification) | n_estimators, max_depth, min_samples_split, min_samples_leaf |
| **LightGBM** | General | small | Lightweight gradient boosting (regression + classification) | n_estimators, max_depth, min_samples_split, min_samples_leaf |
| **Decision Tree** | General | small | Single decision tree (regression + classification) | max_depth, min_samples_split, min_samples_leaf |

To add a new model, see the [Model Extension Guide](templates/models_guide.html).

---

## Architecture

```
deeplearning_labs/
├── main.py                     # Flask entry point
├── routes/
│   ├── data.py                 # Upload, clean, fill endpoints
│   ├── training.py             # Training (sync + SSE stream)
│   ├── evaluation.py           # Evaluation, prediction, CV, download
│   └── projects.py             # Project CRUD, model management
├── utils/
│   ├── model_utils.py          # Train loop, inference, CV, evaluation
│   ├── data_utils.py           # Load, split, normalize, clean, fill
│   ├── plot_utils.py           # Matplotlib → base64 PNG
│   ├── session.py              # SessionManager (state cache)
│   ├── project_manager.py      # Project persistence (disk I/O)
│   ├── pipeline_strategy.py    # Small/Large pipeline dispatcher
│   ├── config.py               # Centralized defaults
│   ├── fonts.py                # Chinese font detection
│   └── models/                 # Model registry (22 architectures)
│       ├── base.py             # Abstract BaseModel
│       ├── mlp.py, cnn.py, rnn.py, lstm.py, gru.py, transformer.py
│       ├── vanilla_transformer.py, autoformer.py, informer.py
│       ├── crossformer.py, dlinear.py, etsformer.py
│       ├── fedformer.py, film.py
│       ├── shared_layers/      # Shared components (Embed, EncDec, Attention)
│       ├── autoformer_layers/  # Autoformer internal package
│       ├── informer_layers/    # Informer internal package
│       ├── crossformer_layers/ # Crossformer internal package
│       ├── etsformer_layers/   # ETSformer internal package
│       ├── fedformer_layers/   # FEDformer internal package
│       └── film_layers/        # FiLM internal package
├── static/
│   ├── js/app.js               # Event bindings & init
│   ├── js/api.js               # API call functions
│   ├── js/ui.js                # DOM rendering & Chart.js
│   ├── js/__tests__/           # Vitest frontend tests
│   ├── css/style.css           # Global styles
│   └── dependency_graph.html   # Code dependency visualization (Mermaid)
├── templates/
│   ├── index.html              # Main application UI
│   └── models_guide.html       # Model extension documentation
├── tests/                      # Python test suite (pytest)
│   ├── test_routes.py
│   ├── test_model_utils.py
│   ├── test_data_utils.py
│   ├── test_session.py
│   ├── test_plot_utils.py
│   ├── test_pipeline_strategy.py
│   ├── test_project_manager.py
│   └── test_split_result.py
├── uploads/                    # Uploaded data (per-session)
├── outputs/                    # Generated outputs
├── projects/                   # Persistent project storage
├── docs/
│   ├── PRD.md                  # Product requirements document
│   └── adr/                    # Architecture Decision Records
├── start.sh                    # Startup/shutdown script
├── requirements.txt            # Python dependencies
├── vitest.config.js            # Vitest configuration
└── vitest.setup.js             # Vitest jsdom setup
```

### Docker Deployment

```bash
docker compose up -d
# App available at http://localhost:5000
```

For GPU support with nvidia-docker, uncomment the `deploy` section in `docker-compose.yml` and ensure [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) is installed.

---

### Key Design Decisions

- **Flask Blueprints** for route organization — data, training, evaluation, projects are separate modules
- **Server-Sent Events** (SSE) for real-time training progress — simpler than WebSocket, native Flask support
- **SessionManager** in-memory cache with disk fallback — survives Flask debug reloads
- **Model Registry pattern** — adding a model = one file + one registry entry, no other code changes
- **Dual Pipeline** — small (2-arg forward) for simple models, large (4-arg forward) for time-series models with time-mark encoding
- **Shared Layers** — common Transformer components in `shared_layers/`, reused across large-pipeline models
- **Chart.js** for live charts — `animation: false` prevents flicker during rapid SSE updates
- **matplotlib Agg backend** — server-side rendering to base64 PNG, embedded inline

---

## Configuration

Default hyperparameters are centralized in `utils/config.py`:

```python
TRAINING = {
    "test_size": 0.2, "learning_rate": 0.001,
    "batch_size": 32, "epochs": 50,
    "patience": 10, "dropout": 0.2,
    "normalization": "none",
}

MODEL = { ... }  # Per-architecture defaults (22 models)
TIME_SERIES = {"seq_len": 10, "pred_len": 1, "label_len": 0}
CV = {"default_folds": 5, "max_epochs_per_fold": 20}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
```

---

## Development

```bash
# Install dev dependencies (optional)
pip install pytest pytest-cov
npm install          # Vitest for JS tests

# Run Python tests (216 tests)
pytest tests/

# Run JS tests (66 tests)
npx vitest run

# Start server (Flask debug mode, reloads on code changes)
python main.py
```

---

## Project Status

Active development on `feat/informer-integration` branch. 216 Python tests + 66 Vitest JS tests pass with full coverage of training, evaluation, prediction, cross-validation, data processing, project management, and frontend logic. See [ISSUES.md](ISSUES.md) for the change log.

---

## License

MIT
