"""Pipeline strategy abstraction — encapsulates pipeline-specific behavior.

A *pipeline* is the dataflow pattern a model uses: how many arguments its
forward pass takes, how the DataLoader is structured, and how outputs are
formatted for loss/metric computation.

Two strategies exist:
  - ``SmallPipelineStrategy`` — ``model(x)`` → output, 2-tuple DataLoader.
  - ``LargePipelineStrategy`` — ``model(x, x_mark, dec, y_mark)`` → output, 5-tuple.

New pipeline types implement ``PipelineStrategy`` and register in
``PipelineStrategy.for_model_type()``.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset


@dataclass
class PipelineData:
    """Extra data arrays needed by large-pipeline models beyond standard X/y.

    For small-pipeline models all fields remain ``None`` / 0.
    Created by callers (routes) from ``split_result``; consumed by
    ``PipelineStrategy`` methods.
    """

    X_mark: Optional[np.ndarray] = None
    dec_inp: Optional[np.ndarray] = None
    y_mark: Optional[np.ndarray] = None
    n_time_features: int = 0
    seq_len: int = 0
    label_len: int = 0

    @classmethod
    def from_split(cls, split_result: dict, subset: str = "train") -> "PipelineData":
        """Construct PipelineData from a split_result dict.

        ``subset`` must be ``"train"`` or ``"test"``.  Returns a populated
        ``PipelineData`` when the corresponding ``x_mark_*`` keys exist in
        *split_result*, or an empty ``PipelineData()`` (all ``None`` fields)
        when they don't (e.g. non-time-series data).
        """
        suffix = f"_{subset}"  # "_train" or "_test"
        key = f"x_mark{suffix}"
        if key not in split_result:
            return cls()
        return cls(
            X_mark=split_result[f"x_mark{suffix}"],
            dec_inp=split_result[f"dec_inp{suffix}"],
            y_mark=split_result[f"y_mark{suffix}"],
            n_time_features=split_result.get("n_time_features", 4),
            seq_len=split_result.get("seq_len", 96),
            label_len=split_result.get("label_len", 0),
        )


class PipelineStrategy(ABC):
    """Interface for pipeline-specific behaviour in training and inference.

    There are two concrete implementations — see class docs above.
    Use ``PipelineStrategy.for_model(model)`` to resolve the right one.
    """

    # ── Dataset construction (for DataLoader) ──────────────────────────────

    @abstractmethod
    def build_dataset(
        self,
        X: np.ndarray,
        y: np.ndarray,
        task_type: str,
        pd: Optional[PipelineData] = None,
    ) -> TensorDataset:
        """Build a ``TensorDataset`` from split arrays for the DataLoader."""

    @abstractmethod
    def unpack_batch(
        self, batch, device: torch.device
    ) -> Tuple[Tuple[torch.Tensor, ...], torch.Tensor]:
        """Unpack a DataLoader batch into ``(model_inputs_tuple, target_tensor)``.

        The model-inputs tuple is designed to be passed as ``model(*inputs)``,
        so each strategy controls exactly how many arguments the forward call
        receives.
        """

    # ── Inference helpers (single forward pass, no DataLoader) ─────────────

    @abstractmethod
    def prepare_inputs(
        self,
        X: np.ndarray,
        device: torch.device,
        pd: Optional[PipelineData] = None,
    ) -> Tuple[torch.Tensor, ...]:
        """Convert raw numpy arrays to the tensor tuple expected by ``model(*...)``."""

    @abstractmethod
    def format_output(self, outputs: torch.Tensor, task_type: str) -> torch.Tensor:
        """Convert raw model outputs to the shape expected by loss/metric computation."""

    # ── Model construction helpers ─────────────────────────────────────────

    @abstractmethod
    def extra_model_kwargs(self, pd: Optional[PipelineData] = None) -> dict:
        """Extra keyword arguments needed when calling ``create_model()``."""

    # ── Resolution ─────────────────────────────────────────────────────────

    @classmethod
    def for_model(cls, model: nn.Module) -> "PipelineStrategy":
        """Resolve the strategy for a given model *instance*."""
        pipeline = getattr(model, "pipeline", "small")
        if pipeline == "large":
            return LargePipelineStrategy()
        return SmallPipelineStrategy()

    @classmethod
    def for_model_type(cls, model_type: str) -> "PipelineStrategy":
        """Resolve the strategy for a given *type string* (e.g. ``'autoformer'``)."""
        # Late import avoids circular dependency (models/__init__ → pipeline_strategy)
        from .models import get_model_pipeline

        pipeline = get_model_pipeline(model_type)
        if pipeline == "large":
            return LargePipelineStrategy()
        return SmallPipelineStrategy()


# ── Concrete implementations ───────────────────────────────────────────────


class SmallPipelineStrategy(PipelineStrategy):
    """``model(x)`` → output — standard 2-tuple DataLoader."""

    def build_dataset(self, X, y, task_type, pd=None):
        X_t = torch.FloatTensor(X)
        y_t = torch.FloatTensor(y) if task_type == "regression" else torch.LongTensor(y)
        return TensorDataset(X_t, y_t)

    def unpack_batch(self, batch, device):
        x, y = batch
        return ((x.to(device),), y.to(device))

    def prepare_inputs(self, X, device, pd=None):
        return (torch.FloatTensor(X).to(device),)

    def format_output(self, outputs, task_type):
        if task_type == "classification":
            return outputs  # raw logits for CrossEntropyLoss
        return outputs if outputs.size(-1) > 1 else outputs.squeeze(-1)

    def extra_model_kwargs(self, pd=None):
        kwargs = {}
        if pd is not None and pd.seq_len:
            kwargs["seq_len"] = pd.seq_len
        if pd is not None and pd.label_len is not None:
            kwargs["label_len"] = pd.label_len
        return kwargs


class LargePipelineStrategy(PipelineStrategy):
    """``model(x_enc, x_mark_enc, x_dec, x_mark_dec)`` → output — 5-tuple DataLoader.

    Only supports regression tasks (time-series forecasting).
    """

    def build_dataset(self, X, y, task_type, pd=None):
        _check_pd(pd)
        return TensorDataset(
            torch.FloatTensor(X),
            torch.FloatTensor(pd.X_mark),
            torch.FloatTensor(pd.dec_inp),
            torch.FloatTensor(pd.y_mark),
            torch.FloatTensor(y),
        )

    def unpack_batch(self, batch, device):
        x, xm, di, ym, y = batch
        return (
            (x.to(device), xm.to(device), di.to(device), ym.to(device)),
            y.to(device),
        )

    def prepare_inputs(self, X, device, pd=None):
        t = [torch.FloatTensor(X).to(device)]
        if pd is not None:
            if pd.X_mark is not None:
                t.append(torch.FloatTensor(pd.X_mark).to(device))
            if pd.dec_inp is not None:
                t.append(torch.FloatTensor(pd.dec_inp).to(device))
            if pd.y_mark is not None:
                t.append(torch.FloatTensor(pd.y_mark).to(device))
        return tuple(t)

    def format_output(self, outputs, task_type):
        # Large pipeline is always regression; outputs (batch, pred_len, c_out)
        if outputs.ndim == 3:
            if outputs.size(-1) == 1:
                return outputs.squeeze(-1)          # (batch, pred_len, 1) → (batch, pred_len)
            return outputs.reshape(outputs.size(0), -1)  # (batch, pred_len, c>1) → (batch, -1)
        return outputs

    def extra_model_kwargs(self, pd=None):
        if pd is None:
            return {}
        return {
            "n_time_features": pd.n_time_features,
            "seq_len": pd.seq_len,
            "label_len": pd.label_len,
        }


def _check_pd(pd: Optional[PipelineData]) -> None:
    """Guard: large-pipeline methods MUST receive non-None PipelineData."""
    if pd is None:
        raise ValueError(
            "LargePipelineStrategy requires PipelineData (X_mark, dec_inp, y_mark). "
            "Pass pipeline_data=… to the training/evaluation function, or ensure "
            "split_result contains 'x_mark_train'."
        )
