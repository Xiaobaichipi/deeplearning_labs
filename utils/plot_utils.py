import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import base64
import io
import json

from .fonts import setup_chinese_font

# Auto-detect and set Chinese font
_cn_font = setup_chinese_font()

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#d4d4d4",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.color": "#e5e5e5",
    "font.family": "sans-serif",
})
plt.rcParams["axes.unicode_minus"] = False


def plot_training_history(history):
    images = {}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(history["train_loss"], label="Train Loss", color="#000000", linewidth=2)
    ax1.plot(history["val_loss"], label="Validation Loss", color="#525252", linewidth=2, linestyle="--")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Loss Curves")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(history["train_metric"], label="Train Metric", color="#000000", linewidth=2)
    ax2.plot(history["val_metric"], label="Validation Metric", color="#525252", linewidth=2, linestyle="--")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Metric")
    ax2.set_title("Metric Curves")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
    buf.seek(0)
    images["training_history"] = base64.b64encode(buf.read()).decode()
    plt.close(fig)

    return images


def plot_feature_importance(feature_names, importance_values, title="Feature Importance"):
    fig, ax = plt.subplots(figsize=(8, max(4, len(feature_names) * 0.3)))
    indices = np.argsort(importance_values)
    ax.barh(range(len(indices)), [importance_values[i] for i in indices],
            color="#000000", height=0.6)
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([feature_names[i] for i in indices])
    ax.set_xlabel("Importance")
    ax.set_title(title)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
    buf.seek(0)
    result = base64.b64encode(buf.read()).decode()
    plt.close(fig)
    return result


def plot_data_distribution(df, columns=None):
    images = {}
    if columns is None:
        num_cols = df.select_dtypes(include=[np.number]).columns[:6]
    else:
        num_cols = [c for c in columns if c in df.select_dtypes(include=[np.number]).columns][:6]

    if len(num_cols) == 0:
        return images

    n = len(num_cols)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for i, col in enumerate(num_cols):
        axes[i].hist(df[col].dropna(), bins=30, edgecolor="white", color="#000000", alpha=0.8)
        axes[i].set_title(col, fontweight=500)
        axes[i].set_xlabel("")
        axes[i].grid(True, alpha=0.3)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
    buf.seek(0)
    images["data_distribution"] = base64.b64encode(buf.read()).decode()
    plt.close(fig)
    return images


def plot_correlation_heatmap(df):
    num_df = df.select_dtypes(include=[np.number])
    if num_df.shape[1] < 2:
        return None
    corr = num_df.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(corr.columns, fontsize=9)
    fig.colorbar(im, ax=ax, shrink=0.8)
    for i in range(len(corr.columns)):
        for j in range(len(corr.columns)):
            ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center",
                    fontsize=7, color="black" if abs(corr.values[i, j]) < 0.5 else "white")
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
    buf.seek(0)
    result = base64.b64encode(buf.read()).decode()
    plt.close(fig)
    return result


def fig_to_base64(fig, dpi=300):
    """Convert a matplotlib figure to a base64-encoded PNG string."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    buf.seek(0)
    data = base64.b64encode(buf.read()).decode()
    plt.close(fig)
    return data


def plot_confusion_matrix(cm, class_names):
    """Plot confusion matrix, return base64 PNG."""
    if not cm or not class_names:
        return None
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    tick_marks = np.arange(len(class_names))
    ax.set(xticks=tick_marks, yticks=tick_marks,
           xticklabels=class_names, yticklabels=class_names,
           xlabel="Predicted", ylabel="True")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    thresh = max(max(row) for row in cm) if cm else 0
    for i in range(len(cm)):
        for j in range(len(cm[i])):
            ax.text(j, i, cm[i][j], ha="center", va="center",
                    color="white" if cm[i][j] > thresh / 2. else "black")
    fig.tight_layout()
    return fig_to_base64(fig)


def plot_roc_curve(y_true, y_score):
    """Plot ROC curve, return (base64 PNG, auc_score)."""
    from sklearn.metrics import roc_curve, roc_auc_score
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f"ROC (AUC = {auc:.4f})")
    ax.plot([0, 1], [0, 1], "k--")
    ax.set(xlabel="False Positive Rate", ylabel="True Positive Rate",
           title="ROC Curve", xlim=[0, 1], ylim=[0, 1.05])
    ax.legend()
    fig.tight_layout()
    return fig_to_base64(fig), auc


def plot_pred_vs_true(y_true, y_pred):
    """Plot predictions vs true values (scatter), return base64 PNG."""
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(y_true, y_pred, alpha=0.5, s=15)
    lo = min(y_true.min(), y_pred.min())
    hi = max(y_true.max(), y_pred.max())
    ax.plot([lo, hi], [lo, hi], "r--", linewidth=1)
    ax.set(xlabel="True Values", ylabel="Predictions", title="Predictions vs True Values")
    fig.tight_layout()
    return fig_to_base64(fig, dpi=300)


def plot_pred_vs_true_line(y_true, y_pred):
    """Plot true vs predicted as two lines by sample index, return base64 PNG."""
    fig, ax = plt.subplots(figsize=(5, 4))
    indices = np.arange(len(y_true))
    ax.plot(indices, y_true, label="True Values", color="#2563eb", linewidth=1.5)
    ax.plot(indices, y_pred, label="Predictions", color="#ea580c", linewidth=1.5, alpha=0.8)
    ax.set(xlabel="Sample Index", ylabel="Value", title="Predictions vs True Values (Line)")
    ax.legend()
    fig.tight_layout()
    return fig_to_base64(fig, dpi=300)


def plot_model_comparison(y_true, predictions_dict):
    """Line chart comparing true values against multiple model predictions.

    Args:
        y_true: 1-D array of true values.
        predictions_dict: dict mapping model_label -> 1-D prediction array.
    Returns:
        base64 PNG string.
    """
    fig, ax = plt.subplots(figsize=(7, 4))
    indices = np.arange(len(y_true))

    ax.plot(indices, y_true, label="True Values", color="#000000",
            linewidth=2, linestyle="--", alpha=0.7)

    colors = ["#3b82f6", "#ea580c", "#27c93f", "#8b5cf6", "#f59e0b",
              "#ec4899", "#06b6d4", "#84cc16"]
    for i, (label, y_pred) in enumerate(predictions_dict.items()):
        color = colors[i % len(colors)]
        ax.plot(indices, y_pred, label=label, color=color,
                linewidth=1.5, alpha=0.85)

    ax.set(xlabel="Sample Index", ylabel="Value",
           title="Model Predictions Comparison")
    ax.legend(fontsize=9)
    fig.tight_layout()
    return fig_to_base64(fig, dpi=300)


def plot_residuals(y_true, y_pred):
    """Plot residual histogram, return base64 PNG."""
    fig, ax = plt.subplots(figsize=(5, 4))
    residuals = y_true - y_pred
    ax.hist(residuals, bins=30, edgecolor="black", alpha=0.7)
    ax.set(xlabel="Residual", ylabel="Frequency", title="Residual Distribution")
    fig.tight_layout()
    return fig_to_base64(fig)
