# DeepLearning Labs

> A browser-based deep learning experiment platform for tabular data. Upload, explore, clean, train, evaluate, and predict — all from your browser, powered by PyTorch.

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c)](https://pytorch.org/)
[![Flask](https://img.shields.io/badge/Flask-3.0%2B-black)](https://flask.palletsprojects.com/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

---

## Features

- **📤 Upload & Explore** — Drag-and-drop CSV/Excel import with auto encoding detection, data preview, statistics, distribution histograms, and correlation heatmaps
- **🧹 Clean & Fill** — Remove duplicates, handle outliers (IQR), fill missing values with multiple strategies (mean, median, mode, ffill, bfill, constant)
- **🧠 Multi-Architecture Models** — MLP, CNN (1D), RNN, LSTM, GRU, Transformer — all with configurable hyperparameters
- **📊 Real-Time Training** — SSE-streamed per-epoch progress with live Loss/Metric curves (Chart.js), early stopping, and LR scheduling
- **📈 Evaluation & Visualization** — Regression metrics (MSE/RMSE/MAE/R²), classification metrics (accuracy/precision/recall/F1), confusion matrix, ROC curve, residual plots
- **🔁 Cross-Validation** — K-fold CV using the same PyTorch model architecture as training
- **🔮 Predict & Export** — Scatter + line comparison charts; download results as CSV or Excel
- **🔁 Multi-Model Comparison** — Select multiple trained models and compare predictions side-by-side
- **💾 Project System** — Persistent project storage with dataset/model versioning across sessions
- **⚡ Device Selection** — CPU / GPU / multi-GPU DataParallel training, selectable from the UI
- **🧩 Model Export** — Download trained model state dicts for external use
- **🔌 Extensible** — Simple model registry — add new architectures by dropping in a file and one registry entry

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

The interface is organized as a 6-step wizard:

| Step | What You Do | What You Get |
|---|---|---|
| **0. Projects** | Create or activate a project | Persistent dataset/model storage across sessions |
| **1. Upload** | Drop a CSV/XLSX file | Auto encoding detection, data loaded into project |
| **2. Explore** | Browse tabs | Data preview, column info, statistics, distribution plots, correlation heatmap |
| **3. Clean & Fill** | Toggle options | Duplicate removal, outlier clipping (IQR), missing value imputation |
| **4. Model Config** | Select model, device & params | Architecture choice (MLP/CNN/RNN/LSTM/GRU/Transformer), CPU/GPU selection, hyperparameters, normalization |
| **5. Train** | Click "Start Training" | Real-time progress bar + live Loss/Metric charts, early stopping, LR scheduling |
| **6. Evaluate & Predict** | Run evaluation/cross-val/predict | Metrics, multi-model comparison, charts, CSV/XLSX download |

---

## Model Support

| Model | Description | Key Parameters |
|---|---|---|
| **MLP** | Fully connected feedforward network | Hidden layers, dropout |
| **CNN 1D** | 1D convolutional network for tabular data | Channels, kernel size, dropout |
| **RNN** | Vanilla recurrent neural network | Hidden size, layers, bidirection, dropout |
| **LSTM** | Long short-term memory network | Hidden size, layers, bidirection, dropout |
| **GRU** | Gated recurrent unit network | Hidden size, layers, bidirection, dropout |
| **Transformer** | Transformer encoder for tabular data | d_model, nhead, feedforward dim, layers, dropout |

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
│   ├── config.py               # Centralized defaults
│   ├── fonts.py                # Chinese font detection
│   └── models/                 # Model registry (MLP, CNN, RNN, etc.)
├── static/
│   ├── js/app.js               # Event bindings & init
│   ├── js/api.js               # API call functions
│   ├── js/ui.js                # DOM rendering & Chart.js
│   └── css/style.css           # Global styles
├── templates/
│   ├── index.html              # Main application UI
│   └── models_guide.html       # Model extension documentation
├── uploads/                    # Uploaded data (per-session)
├── outputs/                    # Generated outputs
├── docs/
│   └── PRD.md                  # Product requirements document
├── start.sh                    # Startup/shutdown script
└── requirements.txt            # Python dependencies
```

### Docker Deployment

```bash
docker compose up -d
# App available at http://localhost:5000
```

For GPU support with nvidia-docker, uncomment the `deploy` section in `docker-compose.yml` and ensure [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) is installed.

---

### Key Design Decisions

- **Flask Blueprints** for route organization — data, training, evaluation are separate modules
- **Server-Sent Events** (SSE) for real-time training progress — simpler than WebSocket, native Flask support
- **SessionManager** in-memory cache with disk fallback — survives Flask debug reloads
- **Model Registry pattern** — adding a model = one file + one registry entry, no other code changes
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

MODEL = { ... }  # Per-architecture defaults
CV = {"default_folds": 5, "max_epochs_per_fold": 20}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # + get_available_devices() for UI dropdown
```

---

## Development

```bash
# Install dev dependencies (optional)
pip install pytest pytest-cov

# Run tests (152 tests covering routes, models, data utils, session, plots)
pytest tests/

# Enable Flask debug mode (default: on)
python main.py  # reloads on code changes
```

---

## Project Status

Stable and ready for use. 152 tests pass with full coverage of training, evaluation, prediction, cross-validation, data processing, and project management. See [ISSUES.md](ISSUES.md) for the change log.

---

## License

MIT
