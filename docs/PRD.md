# DeepLearning Labs — Product Requirements Document

## Problem Statement

Data scientists and ML practitioners need a lightweight, browser-based environment to rapidly experiment with deep learning models on tabular data — without leaving the browser, writing training scripts from scratch, or managing cloud GPU infrastructure. Existing solutions (Jupyter notebooks, cloud notebooks, AutoML platforms) are either too heavy, require local setup, or abstract away model internals.

DeepLearning Labs solves this by providing an all-in-one web UI that guides users from raw CSV/Excel upload through data exploration, cleaning, model configuration, real-time training, evaluation, and prediction export — all running locally with PyTorch.

## Solution

A Flask-based web application with a 6-step guided workflow:

1. **Upload** — Drag-and-drop CSV/Excel import with automatic encoding detection
2. **Explore** — Data preview, column info, statistics, distribution/correlation visualizations
3. **Clean & Fill** — Duplicate removal, outlier handling (IQR), missing value imputation
4. **Model Config** — Choose architecture (MLP/CNN/RNN/LSTM/GRU/Transformer), configure hyperparameters, select normalization
5. **Train** — Real-time training with SSE-streamed progress, live Loss/Metric charts (Chart.js), auto early stopping
6. **Evaluate & Predict** — Metrics, confusion matrix/ROC/Pred-vs-True plots, cross-validation, prediction export (CSV/XLSX)

## User Stories

1. As a data scientist, I want to upload CSV/Excel files, so that I can start working with my data immediately.
2. As a data scientist, I want to preview my data in a table, so that I can inspect raw values before processing.
3. As a data scientist, I want to see column data types, null counts, and basic statistics, so that I can assess data quality.
4. As a data scientist, I want to view distribution histograms and correlation heatmaps, so that I can understand feature relationships.
5. As a data scientist, I want to remove duplicate rows and drop unnecessary columns, so that I can clean my dataset.
6. As a data scientist, I want to handle outliers via IQR with configurable factor, so that extreme values don't skew training.
7. As a data scientist, I want to fill missing values using mean/median/mode/constant/ffill/bfill strategies, so that my dataset is complete.
8. As a ML practitioner, I want to choose from multiple deep learning architectures (MLP, CNN, RNN, LSTM, GRU, Transformer), so that I can match model type to data modality.
9. As a ML practitioner, I want to configure hyperparameters (learning rate, batch size, epochs, dropout, patience), so that I can control the training process.
10. As a ML practitioner, I want to apply feature/target normalization (Min-Max or Z-score), so that training converges faster.
11. As a ML practitioner, I want to see real-time training progress with live loss/metric curves, so that I can monitor convergence without waiting for completion.
12. As a ML practitioner, I want early stopping with configurable patience, so that training stops automatically when validation loss plateaus.
13. As a ML practitioner, I want to see final training history plots (Loss + Metric curves), so that I can diagnose overfitting.
14. As a ML practitioner, I want to evaluate the trained model on the test set, so that I can measure generalization performance.
15. As a ML practitioner, I want to see regression metrics (MSE, RMSE, MAE, R²) with prediction-vs-true scatter and residual plots, so that I can assess prediction quality.
16. As a ML practitioner, I want to see classification metrics (accuracy, precision, recall, F1) with confusion matrix and ROC curve, so that I can assess classifier performance.
17. As a ML practitioner, I want to run K-fold cross-validation using the same model architecture as my trained model, so that I can estimate performance stability.
18. As a ML practitioner, I want to generate predictions on train or test sets, so that I can inspect individual prediction results.
19. As a ML practitioner, I want to download predictions as CSV or Excel files, so that I can share results or use them in downstream tools.
20. As a ML practitioner, I want to see a scatter plot of predictions vs true values and a line comparison chart, so that I can visually assess prediction accuracy.
21. As a user, I want to reset all data and models, so that I can start a new experiment without restarting the server.
22. As a developer, I want to add new model architectures via a simple registry, so that the platform is extensible without modifying core logic.
23. As a developer, I want centralized configuration defaults, so that hyperparameters are consistent across routes and easy to change.

## Implementation Decisions

### Architecture

- **Backend**: Flask (Python 3) with Blueprint-based route organization
- **Frontend**: Vanilla JavaScript (no framework), Chart.js for live charts, modular JS files (app.js/ui.js/api.js)
- **Deep Learning**: PyTorch with CPU/CUDA auto-detection
- **Real-time**: Server-Sent Events (SSE) with threading + queue.Queue for per-epoch progress streaming
- **Session**: Flask session-backed data_id + in-memory SessionManager with disk fallback for DataFrame

### Module Structure

| Module | Type | Responsibility | Key Interface |
|---|---|---|---|
| `routes/data.py` | Route Blueprint | Upload, clean, fill, sample | `POST /api/upload`, `/api/data/clean`, `/api/data/fill` |
| `routes/training.py` | Route Blueprint | Training setup, sync/SSE training | `POST /api/train/setup`, `GET /api/train/stream`, `POST /api/train` |
| `routes/evaluation.py` | Route Blueprint | Evaluation, prediction, CV, download | `POST /api/evaluate`, `/api/predict`, `/api/validate`, `GET /api/predict/download` |
| `utils/model_utils.py` | Shared logic | Train loop, inference, CV, evaluate | `train_model()`, `predict()`, `evaluate()`, `cross_validate_model()` |
| `utils/data_utils.py` | Shared logic | Load, split, normalize, clean, fill | `load_data()`, `split_data()`, `normalize_data()`, `normalize_target()` |
| `utils/plot_utils.py` | Shared logic | Matplotlib plots → base64 PNG | `plot_pred_vs_true()`, `plot_training_history()`, `plot_confusion_matrix()` |
| `utils/session.py` | Shared logic | SessionManager (data/model/split/history cache) | `SessionManager` class |
| `utils/config.py` | Configuration | Default hyperparameters | `TRAINING`, `MODEL`, `CV`, `DEVICE` dicts |
| `utils/models/` | Model registry | 6 PyTorch model classes | `BaseModel` subclass + `MODEL_REGISTRY` entry |
| `utils/fonts.py` | Utility | Chinese font auto-detection | `setup_chinese_font()`, `find_chinese_font()` |

### Key Design Decisions

- **Target normalization for regression**: When normalization is enabled, both X and y are normalized. Predictions are denormalized back to original scale before display. Evaluation metrics use normalized scale to avoid confusion.
- **Cross-validation uses PyTorch models**: Each KFold fold creates a fresh model with the same architecture as the trained model, unlike sklearn-based CV alternatives.
- **SSE over WebSocket**: SSE is simpler for unidirectional server-to-client streaming and works natively with Flask without extra dependencies.
- **Chart.js with animation disabled**: `animation: false` prevents flickering during rapid per-epoch SSE updates.
- **matplotlib Agg backend**: Non-interactive backend for server-side rendering to base64 PNG.
- **Chinese font auto-detection**: Scans system fonts with fallback chain, called once at module init to avoid per-request font setup.
- **SessionManager disk fallback**: If in-memory DataFrame is lost (e.g., Flask reload in debug mode), data is reloaded from the uploaded file.

### API Contracts

**POST /api/upload**
- Input: multipart/form-data with `file`
- Output: `{success, data: {columns, shape, dtypes, null_counts, sample, info, distribution_images, correlation_image}}`

**POST /api/train/setup** + **GET /api/train/stream**
- Setup stores params, SSE emits per-epoch progress events + complete event
- Progress event: `{epoch, total_epochs, train_loss, val_loss, train_metric, val_metric}`
- Complete event: `{history, final_metrics, images, task_type, ...}`

**POST /api/evaluate**
- Output: `{evaluation: {task_type, mse/rmse/mae/r2 OR accuracy/precision/recall/f1, images}}`

**POST /api/predict**
- Output: `{predictions: [{index, prediction, true_value, ...}], plot_image, line_plot_image}`

**GET /api/predict/download?source=test&format=csv**
- Returns CSV or XLSX file with Content-Disposition header

## Testing Decisions

### Testing Philosophy

- Tests should focus on external behavior (function outputs for given inputs), not internal implementation details.
- Pure utility functions (data transforms, metric computation, plot generation) are the highest-value test targets.
- Flask route tests should use the test client to verify status codes, response shapes, and error handling — not mock internal functions.
- Frontend code is not unit-tested (vanilla JS DOM manipulation); manual browser testing is the primary validation method.

### Modules Suitable for Testing

| Module | Priority | Test Approach |
|---|---|---|
| `utils/data_utils.py` | High | Test `normalize_data()`, `normalize_target()`, `denormalize_target()`, `split_data()` with known inputs/expected outputs |
| `utils/model_utils.py` | Medium | Test `predict()` with a trained model; test `cross_validate_model()` with synthetic data |
| `utils/plot_utils.py` | Low | Test that functions return valid base64 PNG strings (not visual correctness) |
| `utils/session.py` | Medium | Test `SessionManager` CRUD operations, disk reload behavior |
| `utils/config.py` | Low | Test that config dicts have expected keys and types |
| `routes/` | Medium | Test each blueprint endpoint via Flask test client — status codes, error responses, response shape |

### Prior Art

The codebase does not currently contain tests. When adding tests, follow standard pytest conventions:
- Tests go in `tests/` directory mirroring the source structure
- Use `pytest` fixtures for Flask app and SessionManager setup
- Use sklearn's `make_regression` / `make_classification` for synthetic training data
- Use `tmp_path` fixture for file I/O operations (upload, disk reload)

## Out of Scope

- GPU cluster / distributed training support
- User authentication and multi-user isolation
- Experiment history persistence across server restarts
- Hyperparameter search / AutoML (grid search, Bayesian optimization)
- Model export to production formats (ONNX, TorchScript)
- Time-series-specific models and evaluation
- Natural language / image data support (tabular data only)
- Docker containerization and deployment scripts
- REST API for external programmatic access

## Further Notes

- The project uses a 6-step wizard UX pattern with step navigation and section visibility toggling.
- All matplotlib plots are rendered server-side to base64 PNG and embedded inline in the HTML.
- The design system follows a minimal black-on-white aesthetic defined in `DESIGN.md` (Ollama-inspired).
- Models are added via a registry pattern in `utils/models/__init__.py` — adding a new model requires: (1) a new class inheriting `BaseModel`, (2) an entry in `MODEL_REGISTRY` with param schema, (3) frontend HTML in the model config step.
- The SSE training endpoint uses a threading pattern: training runs in a daemon thread, pushing progress events to a `queue.Queue`, which the Flask response generator reads and yields as SSE events.
- Cross-validation per-fold epochs are capped at 20 (configurable via `CV.max_epochs_per_fold`) to keep total CV time reasonable.
- Predictions are capped at 100 rows in the API response for display, but downloaded files contain all predictions.
- The `.server.pid` file is a runtime artifact (not tracked).
