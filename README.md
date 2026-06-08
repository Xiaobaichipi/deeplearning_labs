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
- **🔮 Predict & Export** — Generate predictions on train/test sets with scatter + line comparison charts; download results as CSV or Excel
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
> pip install torch --index-url https://download.pytorch.org/whl/cu118
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
| **1. Upload** | Drop a CSV/XLSX file | Auto encoding detection, data loaded into session |
| **2. Explore** | Browse tabs | Data preview, column info, statistics, distribution plots, correlation heatmap |
| **3. Clean & Fill** | Toggle options | Duplicate removal, outlier clipping (IQR), missing value imputation |
| **4. Model Config** | Select model & params | Architecture choice (MLP/CNN/RNN/LSTM/GRU/Transformer), hyperparameters, normalization |
| **5. Train** | Click "Start Training" | Real-time progress bar + live Loss/Metric charts, early stopping, training history plots |
| **6. Evaluate & Predict** | Run evaluation/cross-val/predict | Metrics, visualizations, prediction table + charts, CSV/XLSX download |

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
│   └── evaluation.py           # Evaluation, prediction, CV, download
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
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
```

---

## Development

```bash
# Install dev dependencies (optional)
pip install pytest pytest-cov

# Run tests (once added)
pytest tests/

# Enable Flask debug mode (default: on)
python main.py  # reloads on code changes
```

---

## Project Status

Active development on the `jiagou_youhua` branch. See [ISSUES.md](ISSUES.md) for the full change log.

---

## License

MIT
