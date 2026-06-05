import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import base64
import io
import json

from .fonts import get_chinese_font, setup_chinese_font

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
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
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
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
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
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
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
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    buf.seek(0)
    result = base64.b64encode(buf.read()).decode()
    plt.close(fig)
    return result
