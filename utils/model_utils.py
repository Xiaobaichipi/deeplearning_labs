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


def create_model(model_type, input_dim, output_dim, **params):
    """Create a model instance from the registry by type string."""
    model_class = get_model_class(model_type)
    return model_class(input_dim, output_dim, **params)


def train_model(model, X_train, y_train, X_val, y_val, task_type,
                epochs=50, batch_size=32, lr=0.001, patience=10,
                device="cpu", progress_callback=None):
    """Train a PyTorch model with early stopping and learning rate scheduling.

    If *progress_callback* is provided, it is called after each epoch with a dict
    containing ``epoch``, ``train_loss``, ``val_loss``, ``train_metric``, ``val_metric``.

    *device* can be a string (``'cpu'``, ``'cuda:0'``) or a list of device
    strings for DataParallel multi-GPU (e.g. ``['cuda:0', 'cuda:1']``).
    """
    if isinstance(device, list):
        device_ids = [torch.device(d) for d in device]
        _device = device_ids[0]
        model = model.to(_device)
        model = nn.DataParallel(model, device_ids=device_ids)
    else:
        _device = torch.device(device)
        model = model.to(_device)

    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.FloatTensor(y_train) if task_type == "regression" else torch.LongTensor(y_train)
    X_val_t = torch.FloatTensor(X_val)
    y_val_t = torch.FloatTensor(y_val) if task_type == "regression" else torch.LongTensor(y_val)

    train_dataset = TensorDataset(X_train_t, y_train_t)
    val_dataset = TensorDataset(X_val_t, y_val_t)
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

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(_device), batch_y.to(_device)
            optimizer.zero_grad()
            outputs = model(batch_x)

            if task_type == "regression":
                pred = outputs.squeeze(-1)
                loss = criterion(pred, batch_y)
                train_mae += F.l1_loss(pred, batch_y).item() * batch_x.size(0)
            else:
                loss = criterion(outputs, batch_y)
                _, predicted = torch.max(outputs, 1)
                train_total += batch_y.size(0)
                train_correct += (predicted == batch_y).sum().item()

            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch_x.size(0)

        train_loss /= len(train_loader.dataset)
        train_mae /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        val_mae = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(_device), batch_y.to(_device)
                outputs = model(batch_x)
                if task_type == "regression":
                    pred = outputs.squeeze(-1)
                    loss = criterion(pred, batch_y)
                    val_mae += F.l1_loss(pred, batch_y).item() * batch_x.size(0)
                else:
                    loss = criterion(outputs, batch_y)
                    _, predicted = torch.max(outputs, 1)
                    val_total += batch_y.size(0)
                    val_correct += (predicted == batch_y).sum().item()
                val_loss += loss.item() * batch_x.size(0)

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


def predict(model, X, task_type, device="cpu"):
    """Run inference and return predictions."""
    if isinstance(device, list):
        device = torch.device(device[0])
    else:
        device = torch.device(device)
    model = model.to(device)
    model.eval()
    X_t = torch.FloatTensor(X).to(device)

    with torch.no_grad():
        outputs = model(X_t)
        if task_type == "classification":
            probabilities = torch.softmax(outputs, dim=1)
            predictions = torch.argmax(outputs, dim=1)
            return predictions.cpu().numpy(), probabilities.cpu().numpy()
        else:
            return np.atleast_1d(outputs.squeeze().cpu().numpy()), None


def cross_validate_model(model_type, input_dim, output_dim, X, y, task_type,
                         model_params=None, n_splits=5, epochs=20,
                         batch_size=32, lr=0.001, device="cpu"):
    """K-fold cross-validation using the same model architecture as training.

    For each fold: creates a fresh model, trains it, and scores on the
    held-out fold.  An 80/20 split of the training fold is used as a
    validation set for early stopping.

    Returns a dict with ``cv_scores``, ``mean_score``, ``std_score``,
    and ``n_splits``.
    """
    from sklearn.model_selection import KFold, train_test_split
    from sklearn.metrics import accuracy_score, r2_score

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = []
    model_params = model_params or {}

    for train_idx, val_idx in kf.split(X):
        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]

        # Further split train data for validation (early stopping)
        X_tr, X_v, y_tr, y_v = train_test_split(
            X_train_fold, y_train_fold, test_size=0.2, random_state=42,
        )

        model = create_model(model_type, input_dim, output_dim, **model_params)
        trained_model, _ = train_model(
            model, X_tr, y_tr, X_v, y_v, task_type,
            epochs=epochs, batch_size=batch_size, lr=lr,
            patience=max(epochs // 2, 1), device=device,
        )

        preds, _ = predict(trained_model, X_val_fold, task_type, device=device)
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
             device="cpu"):
    """Evaluate a trained model and return metrics + visualization images."""
    from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                                 f1_score, confusion_matrix, mean_squared_error,
                                 mean_absolute_error, r2_score)
    from .plot_utils import plot_confusion_matrix, plot_roc_curve, \
        plot_pred_vs_true, plot_residuals

    if isinstance(device, list):
        device = torch.device(device[0])
    else:
        device = torch.device(device)
    model.eval()
    X_t = torch.FloatTensor(X_test).to(device)
    y_t = torch.FloatTensor(y_test) if task_type == "regression" else torch.LongTensor(y_test)

    with torch.no_grad():
        outputs = model(X_t)
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
            preds = np.atleast_1d(outputs.squeeze().cpu().numpy())
            y_true = np.atleast_1d(y_t.cpu().numpy())

            mse = mean_squared_error(y_true, preds)
            rmse = float(np.sqrt(mse))
            mae = mean_absolute_error(y_true, preds)
            r2 = r2_score(y_true, preds)

            images = {
                "pred_vs_true": plot_pred_vs_true(y_true, preds),
                "residuals": plot_residuals(y_true, preds),
            }

            return {
                "mse": float(mse),
                "rmse": rmse,
                "mae": float(mae),
                "r2": float(r2),
                "task_type": "regression",
                "images": images,
            }
