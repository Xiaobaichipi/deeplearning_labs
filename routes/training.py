import io
import json
import queue
import threading

import torch
from flask import Blueprint, Response, current_app, jsonify, request, session
from utils import config
from utils.data_utils import normalize_data, normalize_target, split_data
from utils.model_utils import create_model, train_model
from utils.models import get_model_params

from utils.session import get_data_id, json_ok

training_bp = Blueprint("training", __name__)

# ── Synchronous training (original) ──────────────────────────────────────────


@training_bp.route("/api/train", methods=["POST"])
def train():
    sm = current_app.config["session_manager"]
    data_id = get_data_id()
    df = sm.get_data(data_id)
    if df is None:
        return jsonify({"error": "No data uploaded"}), 400
    try:
        params = request.get_json() or {}
        _setup_training(sm, data_id, df, params)
        built = _build_config(params, df)
        split_result = sm.get_split(data_id)
        output_dim = (
            split_result["n_classes"]
            if split_result["task_type"] == "classification"
            else 1
        )

        model = create_model(
            built["model_type"], split_result["input_dim"], output_dim,
            **built["model_params"],
        )
        trained_model, history = train_model(
            model,
            split_result["X_train"], split_result["y_train"],
            split_result["X_test"], split_result["y_test"],
            split_result["task_type"],
            epochs=built["epochs"], batch_size=built["batch_size"],
            lr=built["learning_rate"], patience=built["patience"],
            device=built["device"],
        )

        sm.set_model(data_id, trained_model)
        sm.set_model_config(data_id, built)
        sm.set_history(data_id, history)

        avg_epoch_time = round(sum(history["epoch_times"]) / len(history["epoch_times"]), 2)

        # Persist to active project
        pm = current_app.config["project_manager"]
        active_project_id = session.get("active_project_id")
        if active_project_id:
            try:
                pm.save_split(active_project_id, split_result)
                model_id = pm.next_model_id(active_project_id)
                state_dict = trained_model.state_dict()
                meta = {
                    "model_type": built["model_type"],
                    "model_params": built["model_params"],
                    "final_metrics": {
                        "train_loss": round(history["train_loss"][-1], 6),
                        "val_loss": round(history["val_loss"][-1], 6),
                        "epochs": len(history["train_loss"]),
                        "avg_epoch_time": avg_epoch_time,
                    },
                    "task_type": split_result["task_type"],
                    "feature_names": split_result["feature_names"],
                    "target_name": split_result["target_name"],
                    "train_size": len(split_result["X_train"]),
                    "test_size": len(split_result["X_test"]),
                }
                pm.save_model(active_project_id, model_id, state_dict, meta)
            except Exception:
                pass

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
                "avg_epoch_time": avg_epoch_time,
            },
            "task_type": split_result["task_type"],
            "input_dim": split_result["input_dim"],
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


# ── Real-time training via SSE ───────────────────────────────────────────────


@training_bp.route("/api/train/setup", methods=["POST"])
def train_setup():
    """Store training params; the SSE stream will pick them up."""
    sm = current_app.config["session_manager"]
    data_id = get_data_id()
    df = sm.get_data(data_id)
    if df is None:
        return jsonify({"error": "No data uploaded"}), 400

    params = request.get_json() or {}
    target_col = params.get("target_col")
    if not target_col:
        return jsonify({"error": "Please select a target column"}), 400
    if target_col not in df.columns:
        return jsonify({"error": f"Target column '{target_col}' not found"}), 400

    try:
        _setup_training(sm, data_id, df, params)
        sm.set_pending_params(data_id, params)
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@training_bp.route("/api/train/stream")
def train_stream():
    """SSE endpoint: yields per-epoch progress events, then a 'complete' event."""
    sm = current_app.config["session_manager"]
    data_id = get_data_id()
    params = sm.get_pending_params(data_id)
    if params is None:
        return jsonify({"error": "No pending training params. Call /api/train/setup first."}), 400

    df = sm.get_data(data_id)
    if df is None:
        return jsonify({"error": "No data uploaded"}), 400

    try:
        built = _build_config(params, df)
        split_result = sm.get_split(data_id)
        output_dim = (
            split_result["n_classes"]
            if split_result["task_type"] == "classification"
            else 1
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    pm = current_app.config["project_manager"]
    active_project_id = session.get("active_project_id")

    def generate():
        q = queue.Queue()
        train_result = {}

        def callback(metrics):
            q.put(("progress", metrics))

        def do_train():
            try:
                model = create_model(
                    built["model_type"], split_result["input_dim"], output_dim,
                    **built["model_params"],
                )
                trained_model, history = train_model(
                    model,
                    split_result["X_train"], split_result["y_train"],
                    split_result["X_test"], split_result["y_test"],
                    split_result["task_type"],
                    epochs=built["epochs"], batch_size=built["batch_size"],
                    lr=built["learning_rate"], patience=built["patience"],
                    device=built["device"],
                    progress_callback=callback,
                )
                train_result["model"] = trained_model
                train_result["history"] = history
                q.put(("done", None))
            except Exception as e:
                import traceback
                q.put(("error", traceback.format_exc()))

        t = threading.Thread(target=do_train, daemon=True)
        t.start()

        while True:
            event_type, data = q.get()
            if event_type == "progress":
                yield f"event: progress\ndata: {json.dumps(data)}\n\n"
            elif event_type == "done":
                break
            elif event_type == "error":
                yield f"event: error\ndata: {json.dumps({'error': str(data)})}\n\n"
                return

        t.join()

        # Store results
        sm.set_model(data_id, train_result["model"])
        sm.set_model_config(data_id, built)
        sm.set_history(data_id, train_result["history"])

        history = train_result["history"]
        avg_epoch_time = round(sum(history["epoch_times"]) / len(history["epoch_times"]), 2)

        final = {
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
                "avg_epoch_time": avg_epoch_time,
            },
            "task_type": split_result["task_type"],
            "input_dim": split_result["input_dim"],
            "output_dim": output_dim,
            "feature_names": split_result["feature_names"],
            "target_name": split_result["target_name"],
            "train_size": len(split_result["X_train"]),
            "test_size": len(split_result["X_test"]),
        }

        # Persist split + model to active project
        if active_project_id:
            try:
                pm.save_split(active_project_id, split_result)
                model_id = pm.next_model_id(active_project_id)
                state_dict = train_result["model"].state_dict()
                meta = {
                    "model_type": built["model_type"],
                    "model_params": built["model_params"],
                    "final_metrics": final["final_metrics"],
                    "task_type": split_result["task_type"],
                    "feature_names": split_result["feature_names"],
                    "target_name": split_result["target_name"],
                    "train_size": final["train_size"],
                    "test_size": final["test_size"],
                }
                pm.save_model(active_project_id, model_id, state_dict, meta)
            except Exception:
                pass

        yield f"event: complete\ndata: {json.dumps(final)}\n\n"

    return Response(generate(), mimetype="text/event-stream")


@training_bp.route("/api/train/history/download")
def history_download():
    """Download training history (loss/metric per epoch) as CSV/XLSX."""
    sm = current_app.config["session_manager"]
    data_id = get_data_id()
    history = sm.get_history(data_id)
    if not history:
        return jsonify({"error": "No training history found"}), 400

    fmt = request.args.get("format", "csv")
    import pandas as pd
    df = pd.DataFrame({
        "epoch": list(range(1, len(history["train_loss"]) + 1)),
        "train_loss": history["train_loss"],
        "val_loss": history["val_loss"],
        "train_mae": history["train_metric"],
        "val_mae": history["val_metric"],
    })
    output = io.BytesIO()

    if fmt == "csv":
        df.to_csv(output, index=False)
        output.seek(0)
        return Response(
            output.getvalue(),
            mimetype="text/csv",
            headers={"Content-Disposition": "attachment;filename=training_history.csv"},
        )
    else:
        try:
            df.to_excel(output, index=False, engine="openpyxl")
        except ImportError:
            return jsonify({"error": "openpyxl not installed; use format=csv instead"}), 400
        output.seek(0)
        return Response(
            output.getvalue(),
            mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": "attachment;filename=training_history.xlsx"},
        )


# ── Shared helpers ───────────────────────────────────────────────────────────


def _setup_training(sm, data_id, df, params):
    """Validate params, split data, apply normalization, and store split."""
    target_col = params["target_col"]
    test_size = float(params.get("test_size", config.TRAINING["test_size"]))
    split_result = split_data(df, target_col, test_size=test_size)

    norm_method = params.get("normalization", "none")
    if norm_method in ("minmax", "mean"):
        X_train, X_test, norm_params = normalize_data(
            split_result["X_train"], split_result["X_test"], method=norm_method
        )
        split_result["X_train"] = X_train
        split_result["X_test"] = X_test
        split_result["norm_params"] = norm_params

    # Always normalize target for regression so loss/metric are scale-independent
    if split_result["task_type"] == "regression":
        y_norm = norm_method if norm_method in ("minmax", "mean") else "mean"
        y_train, y_test, y_scaler = normalize_target(
            split_result["y_train"], split_result["y_test"], method=y_norm
        )
        split_result["y_train"] = y_train
        split_result["y_test"] = y_test
        split_result["y_scaler"] = y_scaler

    sm.set_split(data_id, split_result)


def _build_config(params, df):
    """Build model config dict from request params."""
    default_cfg = config.TRAINING
    model_type = params.get("model_type", "mlp")
    learning_rate = float(params.get("learning_rate", default_cfg["learning_rate"]))
    batch_size = int(params.get("batch_size", default_cfg["batch_size"]))
    epochs = int(params.get("epochs", default_cfg["epochs"]))
    patience = int(params.get("patience", default_cfg["patience"]))
    device = config.DEVICE

    model_params = {"dropout": float(params.get("dropout", default_cfg["dropout"]))}
    model_defaults = config.MODEL.get(model_type, {})
    if model_type == "mlp":
        raw = params.get("hidden_layers", model_defaults.get("hidden_layers", "128,64,32"))
        model_params["hidden_layers"] = [int(x) for x in raw.split(",") if x.strip()]
    else:
        schema = get_model_params(model_type)
        for key, info in schema.items():
            raw = params.get(key)
            if raw is None:
                fallback = model_defaults.get(key)
                if fallback is not None:
                    model_params[key] = fallback
                continue
            if info["type"] == "int":
                model_params[key] = int(raw)
            elif info["type"] == "float":
                model_params[key] = float(raw)
            elif info["type"] == "bool":
                model_params[key] = raw in (True, "true", "True", 1, "1")
            else:
                model_params[key] = raw

    return {
        "model_type": model_type,
        "model_params": model_params,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "epochs": epochs,
        "patience": patience,
        "device": device,
    }
