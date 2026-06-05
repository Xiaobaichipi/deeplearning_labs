"""Training, prediction, and evaluation utilities.

Model classes live in utils/models/ — add new models there.
This module provides shared training infrastructure.
"""

import torch
import torch.nn as nn
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
                device="cpu"):
    """Train a PyTorch model with early stopping and learning rate scheduling."""
    device = torch.device(device)
    model = model.to(device)

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

    history = {"train_loss": [], "val_loss": [], "train_metric": [], "val_metric": []}
    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)

            if task_type == "regression":
                loss = criterion(outputs.squeeze(), batch_y)
            else:
                loss = criterion(outputs, batch_y)
                _, predicted = torch.max(outputs, 1)
                train_total += batch_y.size(0)
                train_correct += (predicted == batch_y).sum().item()

            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch_x.size(0)

        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                if task_type == "regression":
                    loss = criterion(outputs.squeeze(), batch_y)
                else:
                    loss = criterion(outputs, batch_y)
                    _, predicted = torch.max(outputs, 1)
                    val_total += batch_y.size(0)
                    val_correct += (predicted == batch_y).sum().item()
                val_loss += loss.item() * batch_x.size(0)

        val_loss /= len(val_loader.dataset)

        train_acc = train_correct / train_total if train_total > 0 else 0
        val_acc = val_correct / val_total if val_total > 0 else 0

        history["train_loss"].append(float(train_loss))
        history["val_loss"].append(float(val_loss))
        if task_type == "classification":
            history["train_metric"].append(float(train_acc))
            history["val_metric"].append(float(val_acc))
        else:
            history["train_metric"].append(float(train_loss))
            history["val_metric"].append(float(val_loss))

        scheduler.step(val_loss)

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

    return model, history


def predict(model, X, task_type, device="cpu"):
    """Run inference and return predictions."""
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
            return outputs.squeeze().cpu().numpy(), None


def evaluate(model, X_test, y_test, task_type, target_encoder=None, device="cpu"):
    """Evaluate a trained model and return metrics + visualization images."""
    from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                                 f1_score, confusion_matrix, mean_squared_error,
                                 mean_absolute_error, r2_score)
    from .plot_utils import plot_confusion_matrix, plot_roc_curve, \
        plot_pred_vs_true, plot_residuals
    from .fonts import setup_chinese_font
    setup_chinese_font()

    device = torch.device(device)
    model = model.to(device)
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
            preds = outputs.squeeze().cpu().numpy()
            y_true = y_t.cpu().numpy()

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
