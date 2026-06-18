"""Tests for utils/pipeline_strategy.py — PipelineData and PipelineStrategy."""
import numpy as np
import torch
import pytest

from utils.pipeline_strategy import (
    PipelineData,
    PipelineStrategy,
    SmallPipelineStrategy,
    LargePipelineStrategy,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def small_split_result():
    """Simulate a small-pipeline time-series split result."""
    n = 20
    return {
        "X_train": np.random.randn(n, 5).astype(np.float32),
        "X_test": np.random.randn(n, 5).astype(np.float32),
        "y_train": np.random.randn(n, 3).astype(np.float32),
        "y_test": np.random.randn(n, 3).astype(np.float32),
        "is_time_series": True,
        "seq_len": 10,
        "pred_len": 3,
        "label_len": 0,
        "n_time_features": 4,
    }


@pytest.fixture
def large_split_result():
    """Simulate a large-pipeline time-series split result with x_mark etc."""
    n = 20
    return {
        "X_train": np.random.randn(n, 5, 3).astype(np.float32),
        "X_test": np.random.randn(n, 5, 3).astype(np.float32),
        "y_train": np.random.randn(n, 3).astype(np.float32),
        "y_test": np.random.randn(n, 3).astype(np.float32),
        "x_mark_train": np.random.randn(n, 5, 4).astype(np.float32),
        "x_mark_test": np.random.randn(n, 5, 4).astype(np.float32),
        "dec_inp_train": np.random.randn(n, 5, 3).astype(np.float32),
        "dec_inp_test": np.random.randn(n, 5, 3).astype(np.float32),
        "y_mark_train": np.random.randn(n, 5, 4).astype(np.float32),
        "y_mark_test": np.random.randn(n, 5, 4).astype(np.float32),
        "is_time_series": True,
        "seq_len": 10,
        "pred_len": 3,
        "label_len": 5,
        "n_time_features": 4,
    }


# ── PipelineData ──────────────────────────────────────────────────────────────

class TestPipelineDataFromSplit:
    def test_from_split_train(self, large_split_result):
        pd = PipelineData.from_split(large_split_result, "train")
        assert pd.X_mark is not None
        assert pd.dec_inp is not None
        assert pd.y_mark is not None
        assert pd.n_time_features == 4
        assert pd.seq_len == 10
        assert pd.label_len == 5

    def test_from_split_test(self, large_split_result):
        pd = PipelineData.from_split(large_split_result, "test")
        assert pd.X_mark is not None
        assert pd.dec_inp is not None
        assert pd.y_mark is not None
        assert pd.seq_len == 10  # from split_result defaults

    def test_from_split_no_keys(self, small_split_result):
        """Non-time-series data: no x_mark keys → returns empty PipelineData."""
        pd = PipelineData.from_split(small_split_result, "train")
        assert pd.X_mark is None
        assert pd.dec_inp is None
        assert pd.y_mark is None
        assert pd.seq_len == 0  # default

    def test_from_split_defaults(self):
        """Empty split_result returns empty PipelineData."""
        pd = PipelineData.from_split({}, "train")
        assert pd.X_mark is None
        assert pd.seq_len == 0


# ── SmallPipelineStrategy ─────────────────────────────────────────────────────

class TestSmallPipelineStrategy:
    def setup_method(self):
        self.strategy = SmallPipelineStrategy()

    def test_build_dataset_regression(self):
        X = np.random.randn(10, 5).astype(np.float32)
        y = np.random.randn(10, 3).astype(np.float32)
        ds = self.strategy.build_dataset(X, y, "regression")
        assert len(ds) == 10
        x_t, y_t = ds[0]
        assert isinstance(x_t, torch.FloatTensor)
        assert isinstance(y_t, torch.FloatTensor)
        assert x_t.shape == (5,)
        assert y_t.shape == (3,)

    def test_build_dataset_classification(self):
        X = np.random.randn(10, 5).astype(np.float32)
        y = np.random.randint(0, 3, size=10)
        ds = self.strategy.build_dataset(X, y, "classification")
        x_t, y_t = ds[0]
        assert isinstance(y_t, torch.LongTensor)

    def test_unpack_batch(self):
        X = torch.randn(4, 5)
        y = torch.randn(4, 3)
        batch = (X, y)
        device = torch.device("cpu")
        inputs, target = self.strategy.unpack_batch(batch, device)
        assert len(inputs) == 1
        assert torch.equal(inputs[0], X)
        assert torch.equal(target, y)

    def test_prepare_inputs(self):
        X = np.random.randn(4, 5).astype(np.float32)
        device = torch.device("cpu")
        inputs = self.strategy.prepare_inputs(X, device)
        assert len(inputs) == 1
        assert isinstance(inputs[0], torch.Tensor)
        assert inputs[0].shape == (4, 5)

    def test_format_output_regression(self):
        outputs = torch.randn(4, 3)  # (batch, pred_len)
        result = self.strategy.format_output(outputs, "regression")
        assert result.shape == (4, 3)  # squeeze no-op for seq > 1

    def test_format_output_regression_single(self):
        outputs = torch.randn(4, 1)  # (batch, 1)
        result = self.strategy.format_output(outputs, "regression")
        assert result.shape == (4,)  # squeezed

    def test_format_output_classification(self):
        outputs = torch.randn(4, 3)  # raw logits
        result = self.strategy.format_output(outputs, "classification")
        assert result.shape == (4, 3)  # unchanged

    def test_extra_model_kwargs_with_pd(self):
        pd = PipelineData(seq_len=10, label_len=5, n_time_features=4)
        kw = self.strategy.extra_model_kwargs(pd)
        assert kw["seq_len"] == 10
        assert kw["label_len"] == 5

    def test_extra_model_kwargs_empty(self):
        kw = self.strategy.extra_model_kwargs(None)
        assert kw == {}

    def test_for_model_small(self):
        """Models without pipeline attribute default to small."""
        model = torch.nn.Linear(5, 1)
        strategy = PipelineStrategy.for_model(model)
        assert isinstance(strategy, SmallPipelineStrategy)

    def test_for_model_type_mlp(self):
        strategy = PipelineStrategy.for_model_type("mlp")
        assert isinstance(strategy, SmallPipelineStrategy)


# ── LargePipelineStrategy ─────────────────────────────────────────────────────

class TestLargePipelineStrategy:
    def setup_method(self):
        self.strategy = LargePipelineStrategy()
        self.pd = PipelineData(
            X_mark=np.random.randn(10, 4).astype(np.float32),
            dec_inp=np.random.randn(10, 3).astype(np.float32),
            y_mark=np.random.randn(10, 4).astype(np.float32),
            seq_len=10,
            label_len=5,
            n_time_features=4,
        )

    def test_build_dataset(self):
        X = np.random.randn(10, 5).astype(np.float32)
        y = np.random.randn(10, 3).astype(np.float32)
        ds = self.strategy.build_dataset(X, y, "regression", self.pd)
        assert len(ds) == 10
        x_t, xm_t, di_t, ym_t, y_t = ds[0]
        assert x_t.shape == (5,)
        assert xm_t.shape == (4,)
        assert di_t.shape == (3,)
        assert ym_t.shape == (4,)
        assert y_t.shape == (3,)

    def test_build_dataset_raises_without_pd(self):
        X = np.random.randn(10, 5).astype(np.float32)
        y = np.random.randn(10, 3).astype(np.float32)
        with pytest.raises(ValueError, match="LargePipelineStrategy requires PipelineData"):
            self.strategy.build_dataset(X, y, "regression", None)

    def test_unpack_batch(self):
        X = torch.randn(4, 5)
        xm = torch.randn(4, 4)
        di = torch.randn(4, 3)
        ym = torch.randn(4, 4)
        y = torch.randn(4, 3)
        batch = (X, xm, di, ym, y)
        device = torch.device("cpu")
        inputs, target = self.strategy.unpack_batch(batch, device)
        assert len(inputs) == 4
        assert torch.equal(inputs[0], X)
        assert torch.equal(inputs[1], xm)
        assert torch.equal(inputs[2], di)
        assert torch.equal(inputs[3], ym)
        assert torch.equal(target, y)

    def test_prepare_inputs_with_pd(self):
        X = np.random.randn(4, 5).astype(np.float32)
        device = torch.device("cpu")
        inputs = self.strategy.prepare_inputs(X, device, self.pd)
        assert len(inputs) == 4  # X + X_mark + dec_inp + y_mark

    def test_prepare_inputs_without_pd(self):
        X = np.random.randn(4, 5).astype(np.float32)
        device = torch.device("cpu")
        inputs = self.strategy.prepare_inputs(X, device, None)
        assert len(inputs) == 1  # just X

    def test_format_output(self):
        outputs = torch.randn(4, 3, 1)  # (batch, pred_len, 1)
        result = self.strategy.format_output(outputs, "regression")
        assert result.shape == (4, 3)  # squeeze(-1)

    def test_extra_model_kwargs_with_pd(self):
        kw = self.strategy.extra_model_kwargs(self.pd)
        assert kw["seq_len"] == 10
        assert kw["label_len"] == 5
        assert kw["n_time_features"] == 4

    def test_extra_model_kwargs_without_pd(self):
        kw = self.strategy.extra_model_kwargs(None)
        assert kw == {}


# ── Large-model output shape validation ───────────────────────────────────

class TestLargeModelOutputShapes:
    """All large-pipeline models must return (batch, pred_len, 1), not
    (batch, label_len+pred_len, 1).  This validates the forward output shape
    for each registered large model across several pred_len values."""

    BATCH = 4
    INPUT_DIM = 8
    N_TIME_FEATURES = 4

    @pytest.mark.parametrize("model_type,label_len,pred_len,seq_len", [
        ("vanilla_transformer", 0, 5, 10),
        ("vanilla_transformer", 2, 5, 10),
        ("vanilla_transformer", 5, 12, 24),
        ("autoformer",           0, 5, 10),
        ("autoformer",           2, 5, 10),
        ("informer",             0, 5, 10),
        ("informer",             2, 5, 10),
    ])
    def test_output_shape(self, model_type, label_len, pred_len, seq_len):
        from utils.models import get_model_class
        from utils.pipeline_strategy import LargePipelineStrategy

        model_cls = get_model_class(model_type)
        model = model_cls(
            input_dim=self.INPUT_DIM,
            output_dim=pred_len,
            seq_len=seq_len,
            label_len=label_len,
            pred_len=pred_len,
            n_time_features=self.N_TIME_FEATURES,
        )

        batch_x = torch.randn(self.BATCH, seq_len, self.INPUT_DIM)
        batch_x_mark = torch.randn(self.BATCH, seq_len, self.N_TIME_FEATURES)
        batch_dec_inp = torch.randn(self.BATCH, label_len + pred_len, self.INPUT_DIM)
        batch_y_mark = torch.randn(self.BATCH, label_len + pred_len, self.N_TIME_FEATURES)

        output = model(batch_x, batch_x_mark, batch_dec_inp, batch_y_mark)

        expected = (self.BATCH, pred_len, 1)
        assert output.shape == expected, (
            f"{model_type}(pred_len={pred_len}, label_len={label_len}): "
            f"expected {expected}, got {output.shape}"
        )
