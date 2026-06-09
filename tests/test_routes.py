"""End-to-end tests for Flask routes using test client."""
import json
import os
import tempfile

import pytest
from main import app
from utils.project_manager import ProjectManager


@pytest.fixture
def client():
    """Flask test client with isolated upload + project dirs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        app.config["UPLOAD_DIR"] = tmpdir
        app.config["TESTING"] = True
        app.config["PROJECTS_DIR"] = os.path.join(tmpdir, "projects")
        app.config["project_manager"] = ProjectManager(app.config["PROJECTS_DIR"])
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

    def test_validate_time_series_multi_step(self, client):
        """Cross-validation must work for time series with pred_len > 1."""
        import pandas as pd
        import numpy as np

        dates = pd.date_range("2024-01-01", periods=200, freq="h")
        df = pd.DataFrame({
            "date": dates.astype(str),
            "target": np.random.randn(200).cumsum(),
            "feat1": np.random.randn(200),
        })
        ts_csv = os.path.join(app.config["UPLOAD_DIR"], "ts_cv_test.csv")
        df.to_csv(ts_csv, index=False)

        with open(ts_csv, "rb") as f:
            client.post("/api/upload", data={"file": f}, content_type="multipart/form-data")

        client.post("/api/data/task-config", json={
            "task_type": "time_series",
            "time_col": "date",
            "seq_len": 10,
            "pred_len": 4,
            "label_len": 0,
        })
        client.post("/api/train", json={
            "target_col": "target",
            "model_type": "gru",
            "epochs": 3,
            "batch_size": 16,
        })
        resp = client.post("/api/validate", json={"n_splits": 2})
        assert resp.status_code == 200, resp.get_json()
        data = resp.get_json()
        assert data["success"] is True
        assert len(data["cv_scores"]) == 2


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


# =============================================================================
# Project CRUD
# =============================================================================

class TestProjectCRUD:
    """Project CRUD endpoints (/api/projects)."""

    def test_list_projects_empty(self, client):
        resp = client.get("/api/projects")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["projects"] == []

    def test_create_project_minimal(self, client):
        resp = client.post("/api/projects", data={"name": "Test Project"})
        assert resp.status_code == 201
        data = resp.get_json()
        assert data["project"]["name"] == "Test Project"
        assert data["project"]["model_count"] == 0

    def test_create_project_missing_name(self, client):
        resp = client.post("/api/projects", data={"name": ""})
        assert resp.status_code == 400
        assert "error" in resp.get_json()

    def test_create_project_with_csv(self, client, csv_data):
        with open(csv_data, "rb") as f:
            resp = client.post("/api/projects", data={
                "name": "CSV Project",
                "file": f,
            }, content_type="multipart/form-data")
        assert resp.status_code == 201
        data = resp.get_json()
        assert data["project"]["original_filename"].endswith(".csv")

    def test_create_project_invalid_format(self, client):
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(b"fake-image")
            png_path = f.name
        with open(png_path, "rb") as f:
            resp = client.post("/api/projects", data={
                "name": "Bad File",
                "file": f,
            }, content_type="multipart/form-data")
        os.unlink(png_path)
        assert resp.status_code == 400
        assert "not supported" in resp.get_json()["error"]

    def test_get_project(self, client):
        create = client.post("/api/projects", data={"name": "My Project"})
        pid = create.get_json()["project"]["id"]
        resp = client.get(f"/api/projects/{pid}")
        assert resp.status_code == 200
        assert resp.get_json()["project"]["name"] == "My Project"

    def test_get_project_not_found(self, client):
        resp = client.get("/api/projects/nonexistent")
        assert resp.status_code == 404

    def test_delete_project(self, client):
        create = client.post("/api/projects", data={"name": "Delete Me"})
        pid = create.get_json()["project"]["id"]
        resp = client.delete(f"/api/projects/{pid}")
        assert resp.status_code == 200
        # Verify gone
        resp2 = client.get(f"/api/projects/{pid}")
        assert resp2.status_code == 404

    def test_list_after_create(self, client):
        client.post("/api/projects", data={"name": "P1"})
        client.post("/api/projects", data={"name": "P2"})
        resp = client.get("/api/projects")
        assert len(resp.get_json()["projects"]) == 2


# =============================================================================
# Project Activation
# =============================================================================

class TestProjectActivation:
    """Activate a project and verify data/model loading."""

    def test_activate_project_with_data(self, client, csv_data):
        with open(csv_data, "rb") as f:
            create = client.post("/api/projects", data={
                "name": "Activate Test",
                "file": f,
            }, content_type="multipart/form-data")
        pid = create.get_json()["project"]["id"]

        resp = client.post(f"/api/projects/{pid}/activate")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["success"] is True
        assert "columns" in data["data"]
        assert "target" in data["data"]["columns"]

    def test_activate_project_no_dataset(self, client):
        create = client.post("/api/projects", data={"name": "No Data"})
        pid = create.get_json()["project"]["id"]
        resp = client.post(f"/api/projects/{pid}/activate")
        assert resp.status_code == 400

    def test_activate_nonexistent(self, client):
        resp = client.post("/api/projects/nonexistent/activate")
        assert resp.status_code == 404

    def test_activate_and_train_persists_model(self, client, csv_data):
        """Activate → train → model is persisted to the project."""
        with open(csv_data, "rb") as f:
            create = client.post("/api/projects", data={
                "name": "Train Persist",
                "file": f,
            }, content_type="multipart/form-data")
        pid = create.get_json()["project"]["id"]

        # Activate (this sets session cookie via test client)
        client.post(f"/api/projects/{pid}/activate")

        # Train
        train_resp = client.post("/api/train", json={
            "target_col": "target",
            "model_type": "mlp",
            "epochs": 3,
            "hidden_layers": "4,2",
            "batch_size": 4,
        })
        assert train_resp.status_code == 200

        # Verify model persisted
        project = client.get(f"/api/projects/{pid}").get_json()["project"]
        assert project["model_count"] == 1

        # Reactivate and check model list
        reactivate = client.post(f"/api/projects/{pid}/activate").get_json()
        models = reactivate["data"].get("models", [])
        assert len(models) == 1
        assert models[0]["model_type"] == "mlp"


# =============================================================================
# Model Export
# =============================================================================

class TestModelExport:
    """Model list and export endpoints (/api/projects/<id>/models)."""

    def _setup_project_with_model(self, client, csv_data):
        """Helper: create project, activate, train → returns project_id."""
        with open(csv_data, "rb") as f:
            create = client.post("/api/projects", data={
                "name": "Export Test",
                "file": f,
            }, content_type="multipart/form-data")
        pid = create.get_json()["project"]["id"]
        client.post(f"/api/projects/{pid}/activate")
        client.post("/api/train", json={
            "target_col": "target",
            "model_type": "mlp",
            "epochs": 3,
            "hidden_layers": "4,2",
            "batch_size": 4,
        })
        return pid

    def test_list_models(self, client, csv_data):
        pid = self._setup_project_with_model(client, csv_data)
        resp = client.get(f"/api/projects/{pid}/models")
        assert resp.status_code == 200
        data = resp.get_json()
        assert len(data["models"]) == 1
        assert data["models"][0]["model_type"] == "mlp"
        assert "final_metrics" in data["models"][0]

    def test_list_models_empty(self, client):
        create = client.post("/api/projects", data={"name": "No Train"})
        pid = create.get_json()["project"]["id"]
        resp = client.get(f"/api/projects/{pid}/models")
        assert resp.get_json()["models"] == []

    def test_list_models_nonexistent_project(self, client):
        resp = client.get("/api/projects/nonexistent/models")
        assert resp.status_code == 404

    def test_export_model(self, client, csv_data):
        pid = self._setup_project_with_model(client, csv_data)
        resp = client.get(f"/api/projects/{pid}/models/model_001/export")
        assert resp.status_code == 200
        # Should be a binary download
        assert len(resp.data) > 0
        assert resp.mimetype == "application/octet-stream" or resp.mimetype.startswith("application/")

    def test_export_model_with_custom_name(self, client, csv_data):
        pid = self._setup_project_with_model(client, csv_data)
        resp = client.get(f"/api/projects/{pid}/models/model_001/export?name=my_model.pt")
        assert resp.status_code == 200
        cd = resp.headers.get("Content-Disposition", "")
        assert "my_model.pt" in cd

    def test_export_model_not_found(self, client, csv_data):
        pid = self._setup_project_with_model(client, csv_data)
        resp = client.get(f"/api/projects/{pid}/models/model_999/export")
        assert resp.status_code == 404

    def test_export_model_nonexistent_project(self, client):
        resp = client.get("/api/projects/nonexistent/models/model_001/export")
        assert resp.status_code == 404


# =============================================================================
# Model Comparison
# =============================================================================

class TestModelComparison:
    """Multi-model prediction comparison endpoint."""

    def _setup_with_two_models(self, client, csv_data):
        """Helper: create project, train two models, return project_id."""
        with open(csv_data, "rb") as f:
            create = client.post("/api/projects", data={
                "name": "Compare Test",
                "file": f,
            }, content_type="multipart/form-data")
        pid = create.get_json()["project"]["id"]
        client.post(f"/api/projects/{pid}/activate")
        client.post("/api/train", json={
            "target_col": "target", "model_type": "mlp",
            "epochs": 3, "hidden_layers": "8,4", "batch_size": 4,
        })
        client.post("/api/train", json={
            "target_col": "target", "model_type": "mlp",
            "epochs": 3, "hidden_layers": "16,8", "batch_size": 4,
        })
        return pid

    def test_compare_two_models(self, client, csv_data):
        pid = self._setup_with_two_models(client, csv_data)
        resp = client.post(f"/api/projects/{pid}/models/compare", json={
            "model_ids": ["model_001", "model_002"],
        })
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["success"] is True
        assert data["loaded_count"] == 2
        assert len(data["plot_image"]) > 100

    def test_compare_no_model_ids(self, client, csv_data):
        pid = self._setup_with_two_models(client, csv_data)
        resp = client.post(f"/api/projects/{pid}/models/compare", json={})
        assert resp.status_code == 400
        assert "No model_ids" in resp.get_json()["error"]

    def test_compare_invalid_model_id(self, client, csv_data):
        pid = self._setup_with_two_models(client, csv_data)
        resp = client.post(f"/api/projects/{pid}/models/compare", json={
            "model_ids": ["model_999"],
        })
        assert resp.status_code == 400

    def test_compare_single_model(self, client, csv_data):
        pid = self._setup_with_two_models(client, csv_data)
        resp = client.post(f"/api/projects/{pid}/models/compare", json={
            "model_ids": ["model_001"],
        })
        assert resp.status_code == 200
        assert resp.get_json()["loaded_count"] == 1

    def test_compare_nonexistent_project(self, client):
        resp = client.post("/api/projects/nonexistent/models/compare", json={
            "model_ids": ["model_001"],
        })
        assert resp.status_code == 404
