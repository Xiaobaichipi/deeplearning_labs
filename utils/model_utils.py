"""Training, prediction, and evaluation utilities.

Model classes live in utils/models/ — add new models there.
This module provides shared training infrastructure.
"""

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

from .models import get_model_class
from .pipeline_strategy import PipelineData, PipelineStrategy


def create_model(model_type, input_dim, output_dim, **params):
    """Create a model instance from the registry by type string."""
    model_class = get_model_class(model_type)
    return model_class(input_dim, output_dim, **params)


def _train_sklearn_model(model, X_train, y_train, X_val, y_val, task_type):
    """Train a sklearn-backed model via .fit() and return metrics.

    Returns ``(model, history_dict)`` where history contains train/val scores.
    """
    from sklearn.metrics import accuracy_score, r2_score, mean_squared_error
    model.fit(X_train, y_train)
    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)

    if task_type == "classification":
        train_score = float(accuracy_score(y_train, train_pred))
        val_score = float(accuracy_score(y_val, val_pred))
    else:
        train_score = float(r2_score(y_train, train_pred))
        val_score = float(r2_score(y_val, val_pred))

    return model, {
        "train_loss": [train_score],
        "val_loss": [val_score],
        "train_metric": [train_score],
        "val_metric": [val_score],
        "epoch_times": [0.0],
        "sklearn_backend": True,
    }


def _evaluate_sklearn(model, X_test, y_test, task_type, target_encoder=None, y_scaler=None):
    """Evaluate a sklearn-backed model — no PyTorch/cuda involved."""
    from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                                 f1_score, confusion_matrix, mean_squared_error,
                                 mean_absolute_error, r2_score)
    from .plot_utils import plot_confusion_matrix, plot_roc_curve, \
        plot_pred_vs_true, plot_residuals
    import numpy as np

    preds = model.predict(X_test)
    y_true = np.asarray(y_test)

    if task_type == "classification":
        probs = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None

        acc = accuracy_score(y_true, preds)
        avg = "binary" if len(np.unique(y_true)) == 2 else "weighted"
        prec = precision_score(y_true, preds, average=avg, zero_division=0)
        rec = recall_score(y_true, preds, average=avg, zero_division=0)
        f1 = f1_score(y_true, preds, average=avg, zero_division=0)
        cm = confusion_matrix(y_true, preds).tolist()

        classes = (
            [str(c) for c in target_encoder.classes_]
            if target_encoder else [str(i) for i in range(len(cm))]
        )

        images = {"confusion_matrix": plot_confusion_matrix(cm, classes)}
        if len(np.unique(y_true)) == 2 and probs is not None:
            roc_img, auc = plot_roc_curve(y_true, probs[:, 1])
            images["roc_curve"] = roc_img

        return {
            "accuracy": float(acc),
            "precision": float(prec),
            "recall": float(rec),
            "f1_score": float(f1),
            "confusion_matrix": cm,
            "class_names": classes,
            "task_type": "classification",
            "images": images,
        }
    else:
        mse = mean_squared_error(y_true, preds)
        rmse = float(np.sqrt(mse))
        mae = mean_absolute_error(y_true, preds)
        r2 = r2_score(y_true, preds)

        if y_scaler is not None:
            from .data_utils import denormalize_target
            plot_preds = denormalize_target(preds, y_scaler)
            plot_true = denormalize_target(y_true, y_scaler)
        else:
            plot_preds, plot_true = preds, y_true

        images = {
            "pred_vs_true": plot_pred_vs_true(plot_true, plot_preds),
            "residuals": plot_residuals(plot_true, plot_preds),
        }

        return {
            "mse": float(mse),
            "rmse": rmse,
            "mae": float(mae),
            "r2": float(r2),
            "task_type": "regression",
            "images": images,
        }


def train_model(model, X_train, y_train, X_val, y_val, task_type,
                epochs=50, batch_size=32, lr=0.001, patience=10,
                device="cpu", progress_callback=None,
                # PipelineData parameters (preferred API)
                pipeline_data=None, pipeline_data_val=None,
                pipeline_strategy=None,
                # Deprecated: use pipeline_data / pipeline_data_val instead
                X_mark_train=None, dec_inp_train=None, y_mark_train=None,
                X_mark_val=None, dec_inp_val=None, y_mark_val=None):
    """Train a model. Dispatches to sklearn .fit() when model uses sklearn backend.
    """
    if getattr(model, "uses_sklearn_backend", False):
        return _train_sklearn_model(model, X_train, y_train, X_val, y_val, task_type)

    if isinstance(device, list):
        device_ids = [torch.device(d) for d in device]
        _device = device_ids[0]
        model = model.to(_device)
        model = nn.DataParallel(model, device_ids=device_ids)
    else:
        _device = torch.device(device)
        model = model.to(_device)

    # ── Resolve pipeline strategy ──────────────────────────────────────
    strategy = pipeline_strategy or PipelineStrategy.for_model(model)

    # Backward compat: construct PipelineData from deprecated kwargs
    if pipeline_data is None and X_mark_train is not None:
        pipeline_data = PipelineData(
            X_mark=X_mark_train, dec_inp=dec_inp_train, y_mark=y_mark_train)
    if pipeline_data_val is None and X_mark_val is not None:
        pipeline_data_val = PipelineData(
            X_mark=X_mark_val, dec_inp=dec_inp_val, y_mark=y_mark_val)
    if pipeline_data_val is None:
        pipeline_data_val = pipeline_data

    train_dataset = strategy.build_dataset(X_train, y_train, task_type, pipeline_data)
    val_dataset = strategy.build_dataset(X_val, y_val, task_type, pipeline_data_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    criterion = nn.MSELoss() if task_type == "regression" else nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=patience // 2
    )

    history = {"train_loss": [], "val_loss": [], "train_metric": [], "val_metric": [], "epoch_times": []}
    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0

    for epoch in range(epochs):
        epoch_start = time.time()
        model.train()
        train_loss = 0.0
        train_mae = 0.0
        train_correct = 0
        train_total = 0

        for batch in train_loader:
            inputs, target = strategy.unpack_batch(batch, _device)
            optimizer.zero_grad()
            outputs = model(*inputs)
            pred = strategy.format_output(outputs, task_type)
            loss = criterion(pred, target)
            loss.backward()
            optimizer.step()

            bs = batch[0].size(0)
            train_loss += loss.item() * bs
            if task_type == "regression":
                train_mae += F.l1_loss(pred, target).item() * bs
            else:
                _, predicted = torch.max(outputs, 1)
                train_total += bs
                train_correct += (predicted == target).sum().item()

        train_loss /= len(train_loader.dataset)
        train_mae /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        val_mae = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for batch in val_loader:
                inputs, target = strategy.unpack_batch(batch, _device)
                outputs = model(*inputs)
                pred = strategy.format_output(outputs, task_type)
                loss = criterion(pred, target)

                bs = batch[0].size(0)
                val_loss += loss.item() * bs
                if task_type == "regression":
                    val_mae += F.l1_loss(pred, target).item() * bs
                else:
                    _, predicted = torch.max(outputs, 1)
                    val_total += bs
                    val_correct += (predicted == target).sum().item()

        val_loss /= len(val_loader.dataset)
        val_mae /= len(val_loader.dataset)

        train_acc = train_correct / train_total if train_total > 0 else 0
        val_acc = val_correct / val_total if val_total > 0 else 0

        history["train_loss"].append(float(train_loss))
        history["val_loss"].append(float(val_loss))
        if task_type == "classification":
            history["train_metric"].append(float(train_acc))
            history["val_metric"].append(float(val_acc))
        else:
            history["train_metric"].append(float(train_mae))
            history["val_metric"].append(float(val_mae))

        epoch_time = time.time() - epoch_start
        history["epoch_times"].append(round(epoch_time, 4))

        scheduler.step(val_loss)

        if progress_callback:
            progress_callback({
                "epoch": epoch + 1,
                "total_epochs": epochs,
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
                "train_metric": history["train_metric"][-1],
                "val_metric": history["val_metric"][-1],
                "epoch_time": round(epoch_time, 4),
            })

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    if best_state:
        model.load_state_dict(best_state)

    # Unwrap DataParallel so callers always get a clean model for saving
    if isinstance(model, nn.DataParallel):
        model = model.module

    return model, history


def predict(model, X, task_type, device="cpu",
            X_mark=None, dec_inp=None, y_mark=None,
            pipeline_strategy=None,
            pipeline_data=None):
    """Run inference and return predictions.

    For large-pipeline models, pass *pipeline_data* with ``X_mark``,
    ``dec_inp``, ``y_mark`` fields.  For sklearn-backed models, calls
    ``model.predict(X)`` directly.
    """
    if getattr(model, "uses_sklearn_backend", False):
        preds = model.predict(X)
        if task_type == "classification":
            probs = model.predict_proba(X) if hasattr(model, "predict_proba") else None
            return np.atleast_1d(preds), probs
        return np.atleast_1d(preds), None

    if isinstance(device, list):
        device = torch.device(device[0])
    else:
        device = torch.device(device)
    model = model.to(device)
    model.eval()

    if pipeline_data is None and X_mark is not None:
        pipeline_data = PipelineData(X_mark=X_mark, dec_inp=dec_inp, y_mark=y_mark)

    strategy = pipeline_strategy or PipelineStrategy.for_model(model)
    inputs = strategy.prepare_inputs(X, device, pipeline_data)
    with torch.no_grad():
        outputs = model(*inputs)
        pred = strategy.format_output(outputs, task_type).cpu().numpy()
        if task_type == "classification":
            probabilities = torch.softmax(outputs, dim=1)
            predictions = torch.argmax(outputs, dim=1)
            return predictions.cpu().numpy(), probabilities.cpu().numpy()
        else:
            return np.atleast_1d(pred), None


def _cross_validate_sklearn(model_type, input_dim, output_dim, X, y, task_type,
                             model_params, n_splits):
    """K-fold CV for sklearn-backed models — no epoch loop needed."""
    from sklearn.model_selection import KFold
    from sklearn.metrics import accuracy_score, r2_score
    import numpy as np

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = []
    for train_idx, val_idx in kf.split(X):
        model = create_model(model_type, input_dim, output_dim, **(model_params or {}))
        model.fit(X[train_idx], y[train_idx])
        preds = model.predict(X[val_idx])
        if task_type == "classification":
            scores.append(float(accuracy_score(y[val_idx], preds)))
        else:
            scores.append(float(r2_score(y[val_idx], preds)))

    return {
        "cv_scores": [round(s, 4) for s in scores],
        "mean_score": round(float(np.mean(scores)), 4),
        "std_score": round(float(np.std(scores)), 4),
        "n_splits": n_splits,
    }


def cross_validate_model(model_type, input_dim, output_dim, X, y, task_type,
                         model_params=None, n_splits=5, epochs=20,
                         batch_size=32, lr=0.001, device="cpu",
                         pipeline_data=None,
                         extra_model_kw=None,
                         pipeline_strategy=None):
    """K-fold cross-validation using the same model architecture as training.

    For each fold: creates a fresh model, trains it, and scores on the
    held-out fold.  An 80/20 split of the training fold is used as a
    validation set for early stopping.

    Returns a dict with ``cv_scores``, ``mean_score``, ``std_score``,
    and ``n_splits``.
    """
    from sklearn.model_selection import KFold, train_test_split
    from sklearn.metrics import accuracy_score, r2_score
    from .models import uses_sklearn_backend

    if uses_sklearn_backend(model_type):
        return _cross_validate_sklearn(
            model_type, input_dim, output_dim, X, y, task_type,
            model_params, n_splits,
        )

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = []
    model_params = model_params or {}
    strategy = pipeline_strategy or PipelineStrategy.for_model_type(model_type)

    for train_idx, val_idx in kf.split(X):
        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]

        if pipeline_data is not None:
            xm_tr = pipeline_data.X_mark[train_idx]
            xm_val = pipeline_data.X_mark[val_idx]
            di_tr = pipeline_data.dec_inp[train_idx]
            di_val = pipeline_data.dec_inp[val_idx]
            ym_tr = pipeline_data.y_mark[train_idx]
            ym_val = pipeline_data.y_mark[val_idx]

        # Further split train data for validation (early stopping)
        tr_sub, v_sub = train_test_split(
            np.arange(len(X_train_fold)), test_size=0.2, random_state=42,
        )

        model_kw = {**model_params, **strategy.extra_model_kwargs(pipeline_data), **(extra_model_kw or {})}
        model = create_model(model_type, input_dim, output_dim, **model_kw)

        # Per-fold PipelineData for training
        fold_pd = PipelineData(
            X_mark=xm_tr[tr_sub], dec_inp=di_tr[tr_sub], y_mark=ym_tr[tr_sub],
        ) if pipeline_data is not None else None
        fold_pd_val = PipelineData(
            X_mark=xm_tr[v_sub], dec_inp=di_tr[v_sub], y_mark=ym_tr[v_sub],
        ) if pipeline_data is not None else None

        trained_model, _ = train_model(
            model, X_train_fold[tr_sub], y_train_fold[tr_sub],
            X_train_fold[v_sub], y_train_fold[v_sub], task_type,
            epochs=epochs, batch_size=batch_size, lr=lr,
            patience=max(epochs // 2, 1), device=device,
            pipeline_strategy=strategy,
            pipeline_data=fold_pd, pipeline_data_val=fold_pd_val,
        )

        fold_pd_test = PipelineData(
            X_mark=xm_val, dec_inp=di_val, y_mark=ym_val,
        ) if pipeline_data is not None else None

        preds, _ = predict(
            trained_model, X_val_fold, task_type, device=device,
            pipeline_strategy=strategy, pipeline_data=fold_pd_test,
        )
        if task_type == "classification":
            scores.append(float(accuracy_score(y_val_fold, preds)))
        else:
            scores.append(float(r2_score(y_val_fold, preds)))

    return {
        "cv_scores": [round(s, 4) for s in scores],
        "mean_score": round(float(np.mean(scores)), 4),
        "std_score": round(float(np.std(scores)), 4),
        "n_splits": n_splits,
    }


def evaluate(model, X_test, y_test, task_type, target_encoder=None,
             device="cpu",
             X_mark_test=None, dec_inp_test=None, y_mark_test=None,
             y_scaler=None,
             pipeline_strategy=None,
             pipeline_data=None):
    """Evaluate a trained model and return metrics + visualization images.

    For large-pipeline models, pass *pipeline_data* with ``X_mark``,
    ``dec_inp``, ``y_mark`` fields.  For sklearn-backed models, computes
    metrics via ``model.predict()``.
    """
    if getattr(model, "uses_sklearn_backend", False):
        return _evaluate_sklearn(model, X_test, y_test, task_type, target_encoder, y_scaler)

    from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                                 f1_score, confusion_matrix, mean_squared_error,
                                 mean_absolute_error, r2_score)
    from .plot_utils import plot_confusion_matrix, plot_roc_curve, \
        plot_pred_vs_true, plot_residuals

    if isinstance(device, list):
        device = torch.device(device[0])
    else:
        device = torch.device(device)
    model = model.to(device)
    model.eval()

    if pipeline_data is None and X_mark_test is not None:
        pipeline_data = PipelineData(X_mark=X_mark_test, dec_inp=dec_inp_test, y_mark=y_mark_test)

    strategy = pipeline_strategy or PipelineStrategy.for_model(model)

    y_t = torch.FloatTensor(y_test) if task_type == "regression" else torch.LongTensor(y_test)

    with torch.no_grad():
        inputs = strategy.prepare_inputs(X_test, device, pipeline_data)
        outputs = model(*inputs)

        if task_type == "classification":
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            y_true = y_t.cpu().numpy()

            acc = accuracy_score(y_true, preds)
            avg = "binary" if len(np.unique(y_true)) == 2 else "weighted"
            prec = precision_score(y_true, preds, average=avg, zero_division=0)
            rec = recall_score(y_true, preds, average=avg, zero_division=0)
            f1 = f1_score(y_true, preds, average=avg, zero_division=0)
            cm = confusion_matrix(y_true, preds).tolist()

            classes = (
                [str(c) for c in target_encoder.classes_]
                if target_encoder else [str(i) for i in range(len(cm))]
            )

            images = {"confusion_matrix": plot_confusion_matrix(cm, classes)}

            # ROC curve (binary only)
            if len(np.unique(y_true)) == 2:
                roc_img, auc = plot_roc_curve(y_true, probs[:, 1])
                images["roc_curve"] = roc_img

            return {
                "accuracy": float(acc),
                "precision": float(prec),
                "recall": float(rec),
                "f1_score": float(f1),
                "confusion_matrix": cm,
                "class_names": classes,
                "task_type": "classification",
                "images": images,
            }
        else:
            preds = strategy.format_output(outputs, task_type).cpu().numpy()
            y_true = y_t.cpu().numpy()

            # Metrics on normalized scale (consistent across datasets)
            mse = mean_squared_error(y_true, preds)
            rmse = float(np.sqrt(mse))
            mae = mean_absolute_error(y_true, preds)
            r2 = r2_score(y_true, preds)

            # Denormalize only for visualization
            if y_scaler is not None:
                from .data_utils import denormalize_target
                plot_preds = denormalize_target(preds, y_scaler)
                plot_true = denormalize_target(y_true, y_scaler)
            else:
                plot_preds, plot_true = preds, y_true

            images = {
                "pred_vs_true": plot_pred_vs_true(plot_true, plot_preds),
                "residuals": plot_residuals(plot_true, plot_preds),
            }

            return {
                "mse": float(mse),
                "rmse": rmse,
                "mae": float(mae),
                "r2": float(r2),
                "task_type": "regression",
                "images": images,
            }
