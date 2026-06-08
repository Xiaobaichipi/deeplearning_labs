"""End-to-end tests for Flask routes using test client."""
import json
import os
import tempfile

import pytest
from main import app


@pytest.fixture
def client():
    """Flask test client with a dedicated upload dir."""
    with tempfile.TemporaryDirectory() as tmpdir:
        app.config["UPLOAD_DIR"] = tmpdir
        app.config["TESTING"] = True
        with app.test_client() as c:
            yield c


@pytest.fixture
def csv_data():
    """Write a small CSV to a temp file and return its path."""
    content = b"feat1,feat2,target\n1.0,10.0,100.0\n2.0,20.0,200.0\n3.0,30.0,300.0\n4.0,40.0,400.0\n5.0,50.0,500.0\n"
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
        f.write(content)
        path = f.name
    yield path
    os.unlink(path)


# =============================================================================
# Index & static pages
# =============================================================================

class TestIndex:
    def test_index_returns_200(self, client):
        resp = client.get("/")
        assert resp.status_code == 200

    def test_models_guide_returns_200(self, client):
        resp = client.get("/models-guide")
        assert resp.status_code == 200


# =============================================================================
# Upload
# =============================================================================

class TestUpload:
    def test_upload_csv_success(self, client, csv_data):
        with open(csv_data, "rb") as f:
            resp = client.post("/api/upload", data={"file": f}, content_type="multipart/form-data")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["success"] is True
        assert "data" in data
        assert data["data"]["shape"] == [5, 3]

    def test_upload_no_file_returns_400(self, client):
        resp = client.post("/api/upload", data={}, content_type="multipart/form-data")
        assert resp.status_code == 400
        assert "error" in resp.get_json()

    def test_upload_empty_filename_returns_400(self, client):
        resp = client.post("/api/upload", data={"file": (b"", "")}, content_type="multipart/form-data")
        assert resp.status_code == 400

    def test_upload_unsupported_format_returns_400(self, client):
        import io
        resp = client.post("/api/upload", data={"file": (io.BytesIO(b"test"), "data.txt")}, content_type="multipart/form-data")
        assert resp.status_code == 400
        assert "Format not supported" in resp.get_json()["error"]


# =============================================================================
# Data info (requires uploaded data)
# =============================================================================

class TestDataInfo:
    def test_info_returns_correct_keys(self, client, csv_data):
        with open(csv_data, "rb") as f:
            client.post("/api/upload", data={"file": f}, content_type="multipart/form-data")
        resp = client.get("/api/data/info")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["success"] is True
        assert "columns" in data["data"]
        assert "dtypes" in data["data"]
        assert "shape" in data["data"]

    def test_info_without_upload_returns_400(self, client):
        resp = client.get("/api/data/info")
        assert resp.status_code == 400


# =============================================================================
# Data sample
# =============================================================================

class TestDataSample:
    def test_sample_returns_rows(self, client, csv_data):
        with open(csv_data, "rb") as f:
            client.post("/api/upload", data={"file": f}, content_type="multipart/form-data")
        resp = client.get("/api/data/sample")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["success"] is True
        assert len(data["rows"]) == 5
        assert data["total_rows"] == 5

    def test_sample_without_upload_returns_400(self, client):
        resp = client.get("/api/data/sample")
        assert resp.status_code == 400


# =============================================================================
# Data clean
# =============================================================================

class TestDataClean:
    def test_clean_returns_report(self, client, csv_data):
        with open(csv_data, "rb") as f:
            client.post("/api/upload", data={"file": f}, content_type="multipart/form-data")
        resp = client.post("/api/data/clean", json={"drop_duplicates": True})
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["success"] is True
        assert "report" in data

    def test_clean_without_upload_returns_400(self, client):
        resp = client.post("/api/data/clean", json={})
        assert resp.status_code == 400


# =============================================================================
# Data fill
# =============================================================================

class TestDataFill:
    def test_fill_returns_report(self, client, csv_data):
        with open(csv_data, "rb") as f:
            client.post("/api/upload", data={"file": f}, content_type="multipart/form-data")
        resp = client.post("/api/data/fill", json={"strategy": "mean"})
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["success"] is True

    def test_fill_without_upload_returns_400(self, client):
        resp = client.post("/api/data/fill", json={})
        assert resp.status_code == 400


# =============================================================================
# Training
# =============================================================================

class TestTraining:
    def test_train_returns_history(self, client, csv_data):
        with open(csv_data, "rb") as f:
            client.post("/api/upload", data={"file": f}, content_type="multipart/form-data")
        resp = client.post("/api/train", json={
            "target_col": "target",
            "model_type": "mlp",
            "epochs": 3,
            "hidden_layers": "4,2",
            "batch_size": 4,
            "normalization": "none",
        })
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["success"] is True
        assert "history" in data
        assert "train_loss" in data["history"]
        assert "val_loss" in data["history"]
        # MAE should be per-sample mean, not sum across all samples.
        # With forced y-normalization, MAE on z-score/minmax scale < 10.
        assert data["history"]["train_metric"][-1] < 10.0
        assert data["history"]["val_metric"][-1] < 10.0

    def test_train_without_upload_returns_400(self, client):
        resp = client.post("/api/train", json={"target_col": "target"})
        assert resp.status_code == 400

    def test_train_without_target_col_returns_400(self, client, csv_data):
        with open(csv_data, "rb") as f:
            client.post("/api/upload", data={"file": f}, content_type="multipart/form-data")
        resp = client.post("/api/train", json={})
        assert resp.status_code == 500  # KeyError on "target_col"


# =============================================================================
# Train setup + SSE stream
# =============================================================================

class TestTrainSetupAndStream:
    def test_setup_stores_params(self, client, csv_data):
        with open(csv_data, "rb") as f:
            client.post("/api/upload", data={"file": f}, content_type="multipart/form-data")
        resp = client.post("/api/train/setup", json={
            "target_col": "target",
            "model_type": "mlp",
            "epochs": 2,
            "hidden_layers": "4",
            "batch_size": 4,
        })
        assert resp.status_code == 200
        assert resp.get_json()["success"] is True

    def test_setup_without_target_returns_400(self, client, csv_data):
        with open(csv_data, "rb") as f:
            client.post("/api/upload", data={"file": f}, content_type="multipart/form-data")
        resp = client.post("/api/train/setup", json={})
        assert resp.status_code == 400
        assert "select a target column" in resp.get_json()["error"]

    def test_setup_with_bad_target_returns_400(self, client, csv_data):
        with open(csv_data, "rb") as f:
            client.post("/api/upload", data={"file": f}, content_type="multipart/form-data")
        resp = client.post("/api/train/setup", json={"target_col": "nonexistent"})
        assert resp.status_code == 400
        assert "not found" in resp.get_json()["error"]

    def test_stream_returns_progress_events(self, client, csv_data):
        with open(csv_data, "rb") as f:
            client.post("/api/upload", data={"file": f}, content_type="multipart/form-data")
        client.post("/api/train/setup", json={
            "target_col": "target",
            "model_type": "mlp",
            "epochs": 2,
            "hidden_layers": "4",
            "batch_size": 4,
        })
        resp = client.get("/api/train/stream")
        assert resp.status_code == 200
        # Parse SSE output
        text = resp.get_data(as_text=True)
        assert "event: progress" in text or "event: complete" in text

    def test_history_download_csv(self, client, csv_data):
        with open(csv_data, "rb") as f:
            client.post("/api/upload", data={"file": f}, content_type="multipart/form-data")
        client.post("/api/train", json={
            "target_col": "target",
            "model_type": "mlp",
            "epochs": 2,
            "hidden_layers": "4",
            "batch_size": 4,
        })
        resp = client.get("/api/train/history/download?format=csv")
        assert resp.status_code == 200
        assert resp.mimetype == "text/csv"
        assert resp.headers["Content-Disposition"].startswith("attachment")

    def test_history_download_without_training_returns_400(self, client):
        resp = client.get("/api/train/history/download")
        assert resp.status_code == 400


# =============================================================================
# Evaluate
# =============================================================================

class TestEvaluate:
    def test_evaluate_returns_metrics(self, client, csv_data):
        with open(csv_data, "rb") as f:
            client.post("/api/upload", data={"file": f}, content_type="multipart/form-data")
        client.post("/api/train", json={
            "target_col": "target",
            "model_type": "mlp",
            "epochs": 3,
            "hidden_layers": "4,2",
            "batch_size": 4,
        })
        resp = client.post("/api/evaluate", json={})
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["success"] is True
        ev = data["evaluation"]
        assert "mse" in ev
        assert "rmse" in ev
        assert "mae" in ev
        assert "r2" in ev
        assert "images" in ev

    def test_evaluate_without_model_returns_400(self, client):
        resp = client.post("/api/evaluate", json={})
        assert resp.status_code == 400


# =============================================================================
# Predict
# =============================================================================

class TestPredict:
    def test_predict_returns_predictions(self, client, csv_data):
        with open(csv_data, "rb") as f:
            client.post("/api/upload", data={"file": f}, content_type="multipart/form-data")
        client.post("/api/train", json={
            "target_col": "target",
            "model_type": "mlp",
            "epochs": 3,
            "hidden_layers": "4,2",
            "batch_size": 4,
        })
        resp = client.post("/api/predict", json={"use_test": True})
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["success"] is True
        assert "predictions" in data
        assert data["count"] > 0
        assert "plot_image" in data
        assert "line_plot_image" in data

    def test_predict_without_model_returns_400(self, client):
        resp = client.post("/api/predict", json={})
        assert resp.status_code == 400


# =============================================================================
# Predict download
# =============================================================================

class TestPredictDownload:
    def test_download_csv(self, client, csv_data):
        with open(csv_data, "rb") as f:
            client.post("/api/upload", data={"file": f}, content_type="multipart/form-data")
        client.post("/api/train", json={
            "target_col": "target",
            "model_type": "mlp",
            "epochs": 3,
            "hidden_layers": "4,2",
            "batch_size": 4,
        })
        resp = client.get("/api/predict/download?source=test&format=csv")
        assert resp.status_code == 200
        assert resp.mimetype == "text/csv"
        assert resp.headers["Content-Disposition"].startswith("attachment")

    def test_download_without_model_returns_400(self, client):
        resp = client.get("/api/predict/download")
        assert resp.status_code == 400


# =============================================================================
# Cross-validation
# =============================================================================

class TestCrossValidation:
    def test_validate_returns_cv_scores(self, client, csv_data):
        with open(csv_data, "rb") as f:
            client.post("/api/upload", data={"file": f}, content_type="multipart/form-data")
        client.post("/api/train", json={
            "target_col": "target",
            "model_type": "mlp",
            "epochs": 3,
            "hidden_layers": "4,2",
            "batch_size": 4,
        })
        resp = client.post("/api/validate", json={"n_splits": 2})
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["success"] is True
        assert "cv_scores" in data
        assert "mean_score" in data

    def test_validate_without_model_returns_400(self, client):
        resp = client.post("/api/validate", json={})
        assert resp.status_code == 400


# =============================================================================
# Reset
# =============================================================================

class TestReset:
    def test_reset_returns_success(self, client, csv_data):
        with open(csv_data, "rb") as f:
            client.post("/api/upload", data={"file": f}, content_type="multipart/form-data")
        resp = client.post("/api/reset")
        assert resp.status_code == 200
        assert resp.get_json()["success"] is True
        # After reset, info should return 400
        resp2 = client.get("/api/data/info")
        assert resp2.status_code == 400

    def test_reset_without_session_still_succeeds(self, client):
        resp = client.post("/api/reset")
        assert resp.status_code == 200
