import os
import uuid
import math
import numpy as np
from flask import Flask, request, jsonify, render_template, session
from werkzeug.utils import secure_filename

from utils.data_utils import load_data, get_data_info, clean_data, fill_missing, split_data, normalize_data
from utils.model_utils import create_model, train_model, predict, evaluate
from utils.models import get_model_params
from utils.plot_utils import plot_training_history, plot_data_distribution, plot_correlation_heatmap
from utils.session import SessionManager


def clean_nan(obj):
    """Recursively replace NaN/Infinity with None for safe JSON serialization."""
    if isinstance(obj, dict):
        return {k: clean_nan(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [clean_nan(v) for v in obj]
    elif isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else obj
    elif isinstance(obj, np.floating):
        val = float(obj)
        return None if (math.isnan(val) or math.isinf(val)) else val
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return clean_nan(obj.tolist())
    return obj


app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "deeplearning-labs-dev-key-2024")

UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "uploads")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
app.config["UPLOAD_DIR"] = UPLOAD_DIR
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024
ALLOWED_EXTENSIONS = {".csv", ".xls", ".xlsx"}

session_manager = SessionManager(UPLOAD_DIR)


def allowed_file(filename):
    _, ext = os.path.splitext(filename)
    return ext.lower() in ALLOWED_EXTENSIONS


def get_data_id():
    if "data_id" not in session:
        session["data_id"] = uuid.uuid4().hex
    return session["data_id"]


def json_ok(data):
    return jsonify(clean_nan(data))


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/models-guide")
def models_guide():
    return render_template("models_guide.html")


@app.route("/api/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400
    f = request.files["file"]
    if f.filename == "":
        return jsonify({"error": "Empty filename"}), 400
    if not allowed_file(f.filename):
        return jsonify({"error": f"Format not supported. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"}), 400

    filename = secure_filename(f.filename)
    # secure_filename strips non-ASCII chars; fallback to a safe name
    if not filename or filename == ".":
        ext = os.path.splitext(f.filename)[1] or ".csv"
        filename = f"upload_{uuid.uuid4().hex[:8]}{ext}"
    data_id = get_data_id()
    file_dir = os.path.join(UPLOAD_DIR, data_id)
    os.makedirs(file_dir, exist_ok=True)
    filepath = os.path.join(file_dir, filename)
    f.save(filepath)

    try:
        df = load_data(filepath)
        session_manager.set_data(data_id, df)
        info = get_data_info(df)
        info["filename"] = filename
        dist_images = plot_data_distribution(df)
        corr_img = plot_correlation_heatmap(df)
        info["distribution_images"] = dist_images
        info["correlation_image"] = corr_img
        return json_ok({"success": True, "data": info})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/data/info", methods=["GET"])
def data_info():
    df = session_manager.get_data(get_data_id())
    if df is None:
        return jsonify({"error": "No data uploaded"}), 400
    try:
        info = get_data_info(df)
        dist_images = plot_data_distribution(df)
        corr_img = plot_correlation_heatmap(df)
        info["distribution_images"] = dist_images
        info["correlation_image"] = corr_img
        return json_ok({"success": True, "data": info})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/data/clean", methods=["POST"])
def data_clean():
    data_id = get_data_id()
    df = session_manager.get_data(data_id)
    if df is None:
        return jsonify({"error": "No data uploaded"}), 400
    try:
        params = request.get_json() or {}
        cleaned, report = clean_data(
            df,
            drop_duplicates=params.get("drop_duplicates", True),
            drop_columns=params.get("drop_columns"),
            handle_outliers=params.get("handle_outliers", False),
            outlier_method=params.get("outlier_method", "iqr"),
            outlier_factor=float(params.get("outlier_factor", 1.5)),
        )
        session_manager.set_data(data_id, cleaned)
        info = get_data_info(cleaned)
        info["report"] = report
        dist_images = plot_data_distribution(cleaned)
        corr_img = plot_correlation_heatmap(cleaned)
        info["distribution_images"] = dist_images
        info["correlation_image"] = corr_img
        return json_ok({"success": True, "data": info, "report": report})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/data/fill", methods=["POST"])
def data_fill():
    data_id = get_data_id()
    df = session_manager.get_data(data_id)
    if df is None:
        return jsonify({"error": "No data uploaded"}), 400
    try:
        params = request.get_json() or {}
        filled, report = fill_missing(
            df,
            strategy=params.get("strategy", "auto"),
            columns=params.get("columns"),
            fill_value=params.get("fill_value"),
        )
        session_manager.set_data(data_id, filled)
        info = get_data_info(filled)
        info["report"] = report
        dist_images = plot_data_distribution(filled)
        corr_img = plot_correlation_heatmap(filled)
        info["distribution_images"] = dist_images
        info["correlation_image"] = corr_img
        return json_ok({"success": True, "data": info, "report": report})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/data/sample", methods=["GET"])
def data_sample():
    data_id = get_data_id()
    df = session_manager.get_data(data_id)
    if df is None:
        return jsonify({"error": "No data uploaded"}), 400
    try:
        return json_ok({
            "success": True,
            "columns": list(df.columns),
            "rows": df.head(100).to_dict(orient="records"),
            "total_rows": len(df),
            "total_cols": len(df.columns),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/train", methods=["POST"])
def train():
    data_id = get_data_id()
    df = session_manager.get_data(data_id)
    if df is None:
        return jsonify({"error": "No data uploaded"}), 400
    try:
        params = request.get_json() or {}
        target_col = params.get("target_col")
        if not target_col:
            return jsonify({"error": "Please select a target column"}), 400
        if target_col not in df.columns:
            return jsonify({"error": f"Target column '{target_col}' not found"}), 400

        test_size = float(params.get("test_size", 0.2))
        split_result = split_data(df, target_col, test_size=test_size)

        # Apply normalization if requested
        norm_method = params.get("normalization", "none")
        if norm_method in ("minmax", "mean"):
            X_train, X_test, norm_params = normalize_data(
                split_result["X_train"], split_result["X_test"], method=norm_method
            )
            split_result["X_train"] = X_train
            split_result["X_test"] = X_test
            split_result["norm_params"] = norm_params

        session_manager.set_split(data_id, split_result)

        model_type = params.get("model_type", "mlp")
        learning_rate = float(params.get("learning_rate", 0.001))
        batch_size = int(params.get("batch_size", 32))
        epochs = int(params.get("epochs", 50))
        patience = int(params.get("patience", 10))

        # Build model params from the registry schema
        model_params = {"dropout": float(params.get("dropout", 0.2))}
        if model_type == "mlp":
            raw = params.get("hidden_layers", "128,64,32")
            model_params["hidden_layers"] = [int(x) for x in raw.split(",") if x.strip()]
        else:
            schema = get_model_params(model_type)
            for key, info in schema.items():
                raw = params.get(key)
                if raw is None:
                    continue
                if info["type"] == "int":
                    model_params[key] = int(raw)
                elif info["type"] == "float":
                    model_params[key] = float(raw)
                elif info["type"] == "bool":
                    model_params[key] = raw in (True, "true", "True", 1, "1")
                else:
                    model_params[key] = raw

        input_dim = split_result["input_dim"]
        output_dim = split_result["n_classes"] if split_result["task_type"] == "classification" else 1

        model = create_model(model_type, input_dim, output_dim, **model_params)

        device = "cpu"
        try:
            import torch
            if torch.cuda.is_available():
                device = "cuda"
        except:
            pass

        trained_model, history = train_model(
            model,
            split_result["X_train"], split_result["y_train"],
            split_result["X_test"], split_result["y_test"],
            split_result["task_type"],
            epochs=epochs, batch_size=batch_size, lr=learning_rate,
            patience=patience, device=device,
        )

        session_manager.set_model(data_id, trained_model)
        session_manager.set_history(data_id, history)

        plot_images = plot_training_history(history)

        return json_ok({
            "success": True,
            "history": {
                "train_loss": [round(x, 6) for x in history["train_loss"]],
                "val_loss": [round(x, 6) for x in history["val_loss"]],
                "train_metric": [round(x, 6) for x in history["train_metric"]],
                "val_metric": [round(x, 6) for x in history["val_metric"]],
            },
            "final_metrics": {
                "train_loss": round(history["train_loss"][-1], 6),
                "val_loss": round(history["val_loss"][-1], 6),
                "epochs": len(history["train_loss"]),
            },
            "images": plot_images,
            "task_type": split_result["task_type"],
            "input_dim": input_dim,
            "output_dim": output_dim,
            "feature_names": split_result["feature_names"],
            "target_name": split_result["target_name"],
            "train_size": len(split_result["X_train"]),
            "test_size": len(split_result["X_test"]),
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/evaluate", methods=["POST"])
def api_evaluate():
    data_id = get_data_id()
    if not session_manager.has_model(data_id):
        return jsonify({"error": "Model not trained yet"}), 400
    try:
        model = session_manager.get_model(data_id)
        split_result = session_manager.get_split(data_id)
        result = evaluate(
            model,
            split_result["X_test"], split_result["y_test"],
            split_result["task_type"],
            target_encoder=split_result.get("target_encoder"),
            device="cpu",
        )
        return json_ok({"success": True, "evaluation": result, "task_type": split_result["task_type"]})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/predict", methods=["POST"])
def api_predict():
    data_id = get_data_id()
    if not session_manager.has_model(data_id):
        return jsonify({"error": "Model not trained yet"}), 400
    try:
        split_result = session_manager.get_split(data_id)
        model = session_manager.get_model(data_id)
        params = request.get_json() or {}
        use_test = params.get("use_test", True)

        if use_test:
            X = split_result["X_test"]
            y_true = split_result["y_test"]
        else:
            X = split_result["X_train"]
            y_true = split_result["y_train"]

        preds, probs = predict(model, X, split_result["task_type"], device="cpu")
        target_encoder = split_result.get("target_encoder")

        results = []
        for i in range(min(len(preds), 100)):
            row = {"index": int(i), "prediction": int(preds[i]) if split_result["task_type"] == "classification" else float(preds[i])}
            if target_encoder:
                row["prediction_label"] = str(target_encoder.inverse_transform([int(preds[i])])[0])
                row["true_label"] = str(target_encoder.inverse_transform([int(y_true[i])])[0])
            row["true_value"] = float(y_true[i]) if split_result["task_type"] == "regression" else int(y_true[i])
            if probs is not None:
                row["probabilities"] = [float(p) for p in probs[i]]
            results.append(row)

        return json_ok({
            "success": True,
            "predictions": results,
            "task_type": split_result["task_type"],
            "count": len(preds),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/validate", methods=["POST"])
def api_validate():
    data_id = get_data_id()
    if not session_manager.has_model(data_id):
        return jsonify({"error": "Model not trained yet"}), 400
    try:
        import numpy as np
        from sklearn.model_selection import cross_val_score, KFold
        from sklearn.neural_network import MLPClassifier, MLPRegressor

        split_result = session_manager.get_split(data_id)
        X_train = split_result["X_train"]
        y_train = split_result["y_train"]

        n_splits = int(request.get_json().get("n_splits", 5)) if request.get_json() else 5
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

        if split_result["task_type"] == "classification":
            scores = cross_val_score(MLPClassifier(
                hidden_layer_sizes=(64, 32), max_iter=200,
                random_state=42, early_stopping=True
            ), X_train, y_train, cv=kf, scoring="accuracy")
            return json_ok({
                "success": True,
                "cv_scores": [round(s, 4) for s in scores.tolist()],
                "mean_score": round(float(scores.mean()), 4),
                "std_score": round(float(scores.std()), 4),
                "n_splits": n_splits,
            })
        else:
            scores = cross_val_score(MLPRegressor(
                hidden_layer_sizes=(64, 32), max_iter=200,
                random_state=42, early_stopping=True
            ), X_train, y_train, cv=kf, scoring="r2")
            return json_ok({
                "success": True,
                "cv_scores": [round(s, 4) for s in scores.tolist()],
                "mean_score": round(float(scores.mean()), 4),
                "std_score": round(float(scores.std()), 4),
                "n_splits": n_splits,
            })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/reset", methods=["POST"])
def api_reset():
    data_id = get_data_id()
    session_manager.reset(data_id)
    session.pop("data_id", None)
    return jsonify({"success": True})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
