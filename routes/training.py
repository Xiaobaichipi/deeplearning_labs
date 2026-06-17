import io
import json
import queue
import threading

import numpy as np
import torch
from flask import Blueprint, Response, current_app, jsonify, request, session
from utils import config
from utils.data_utils import normalize_data, normalize_data_apply, normalize_target, split_data
from utils.model_utils import create_model, train_model
from utils.models import get_model_class, get_model_params, get_model_pipeline
from utils.pipeline_strategy import PipelineData, PipelineStrategy

from utils.session import (
    RouteError, ensure_data, get_data_id, get_sm, handle_errors, json_ok,
)

training_bp = Blueprint("training", __name__)

# ── Synchronous training (original) ──────────────────────────────────────────


@training_bp.route("/api/train", methods=["POST"])
@handle_errors
def train():
    sm = get_sm()
    data_id = get_data_id()
    df = ensure_data(sm, data_id)
    params = request.get_json() or {}
    _setup_training(sm, data_id, df, params)
    built = _build_config(params, df)
    split_result = sm.get_split(data_id)
    if split_result.get("is_time_series"):
        output_dim = split_result["pred_len"]
    else:
        output_dim = (
            split_result["n_classes"]
            if split_result["task_type"] == "classification"
            else 1
        )

    history, final = _run_and_persist(
        sm, data_id, split_result, built, output_dim,
        pm=current_app.config["project_manager"],
        active_project_id=session.get("active_project_id"),
    )
    return json_ok({"success": True, **final})


# ── Real-time training via SSE ───────────────────────────────────────────────


@training_bp.route("/api/train/setup", methods=["POST"])
@handle_errors
def train_setup():
    """Store training params; the SSE stream will pick them up."""
    sm = get_sm()
    data_id = get_data_id()
    df = ensure_data(sm, data_id)

    params = request.get_json() or {}
    target_col = params.get("target_col")
    if not target_col:
        raise RouteError("Please select a target column")
    if target_col not in df.columns:
        raise RouteError(f"Target column '{target_col}' not found")

    _setup_training(sm, data_id, df, params)
    sm.set_pending_params(data_id, params)
    return jsonify({"success": True})


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
        if split_result.get("is_time_series"):
            output_dim = split_result["pred_len"]
        else:
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
                _, final = _run_and_persist(
                    sm, data_id, split_result, built, output_dim,
                    pm=pm, active_project_id=active_project_id,
                    progress_callback=callback,
                )
                train_result["final"] = final
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
        final = train_result.get("final", {})
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


def _run_and_persist(sm, data_id, split_result, built, output_dim,
                     pm=None, active_project_id=None, progress_callback=None):
    """Create model, train, store in SessionManager, persist to active project.

    Returns (history, final_dict).  *pm* and *active_project_id* are both
    required for project persistence; pass *progress_callback* for SSE.
    """
    strategy = PipelineStrategy.for_model_type(built["model_type"])

    pd_train = PipelineData.from_split(split_result, "train")
    pd_val = PipelineData.from_split(split_result, "test")

    extra_kw = strategy.extra_model_kwargs(pd_train)
    if split_result.get("is_time_series") and "seq_len" not in extra_kw:
        extra_kw["seq_len"] = split_result["seq_len"]
    model = create_model(
        built["model_type"], split_result["input_dim"], output_dim,
        **built["model_params"],
        **extra_kw,
    )

    trained_model, history = train_model(
        model,
        split_result["X_train"], split_result["y_train"],
        split_result["X_test"], split_result["y_test"],
        split_result["task_type"],
        epochs=built["epochs"], batch_size=built["batch_size"],
        lr=built["learning_rate"], patience=built["patience"],
        device=built["device"],
        progress_callback=progress_callback,
        pipeline_strategy=strategy,
        pipeline_data=pd_train, pipeline_data_val=pd_val,
    )

    sm.set_model(data_id, trained_model)
    sm.set_model_config(data_id, built)
    sm.set_history(data_id, history)

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
        "is_time_series": split_result.get("is_time_series", False),
        "seq_len": split_result.get("seq_len"),
        "pred_len": split_result.get("pred_len"),
    }

    if pm and active_project_id:
        try:
            pm.save_split(active_project_id, split_result)
            model_id = pm.next_model_id(active_project_id)
            state_dict = trained_model.state_dict()
            meta = {
                "model_type": built["model_type"],
                "model_params": built["model_params"],
                "final_metrics": final["final_metrics"],
                "task_type": split_result["task_type"],
                "feature_names": split_result["feature_names"],
                "target_name": split_result["target_name"],
                "train_size": final["train_size"],
                "test_size": final["test_size"],
                "input_dim": split_result["input_dim"],
                "output_dim": output_dim,
                "is_time_series": split_result.get("is_time_series", False),
                "seq_len": split_result.get("seq_len"),
                "pred_len": split_result.get("pred_len"),
                "time_col": split_result.get("time_col"),
                "time_granularity": split_result.get("time_granularity"),
                "pipeline": get_model_pipeline(built["model_type"]),
                "n_time_features": split_result.get("n_time_features"),
                "label_len": split_result.get("label_len"),
            }
            pm.save_model(active_project_id, model_id, state_dict, meta)
        except Exception:
            pass

    return history, final


def _setup_training(sm, data_id, df, params):
    """Validate params, split data, apply normalization, and store split."""
    target_col = params["target_col"]
    test_size = float(params.get("test_size", config.TRAINING["test_size"]))
    model_type = params.get("model_type", "mlp")
    pipeline = get_model_pipeline(model_type)

    # Check for time series config
    task_config = sm.get_task_config(data_id) or {}
    ts_params = {}

    # Large-pipeline models (Autoformer, Informer, Crossformer) require
    # time_series mode because they need multi-dimensional data
    # (x_mark, dec_inp, y_mark) produced by the large-pipeline split path.
    if pipeline == "large" and task_config.get("task_type") != "time_series":
        raise RouteError(
            f"{model_type} requires Time Series mode. "
            f"Set task type to 'Time Series' in Step 2 and click Apply & Refresh."
        )

    if task_config.get("task_type") == "time_series":
        seq_len = int(task_config.get("seq_len", config.TIME_SERIES["seq_len"]))
        pred_len = int(task_config.get("pred_len", config.TIME_SERIES["pred_len"]))
        label_len = int(task_config.get("label_len", config.TIME_SERIES["label_len"]))

        # Validate time-series parameters
        if seq_len < 2:
            raise RouteError(f"seq_len must be at least 2, got {seq_len}")
        if pred_len < 1:
            raise RouteError(f"pred_len must be at least 1, got {pred_len}")
        if label_len < 0:
            raise RouteError(f"label_len must be non-negative, got {label_len}")
        if seq_len <= pred_len:
            raise RouteError(f"seq_len ({seq_len}) must be greater than pred_len ({pred_len})")

        # Crossformer-specific validation
        if model_type == "crossformer":
            seg_len = int(params.get("seg_len", 12))
            win_size = int(params.get("win_size", 2))
            e_layers = int(params.get("e_layers", 3))
            if seq_len % seg_len != 0:
                raise RouteError(
                    f"seq_len ({seq_len}) must be divisible by seg_len ({seg_len}) "
                    f"for Crossformer. Try adjusting seg_len or seq_len."
                )
            if win_size > e_layers:
                raise RouteError(
                    f"win_size ({win_size}) must be ≤ e_layers ({e_layers}) for Crossformer."
                )

        # ETSformer-specific validation
        if model_type == "etsformer":
            top_k = int(params.get("top_k", 5))
            # freq_len after rfft + low_freq=1 slicing:
            #   even: seq_len//2 - 1   (e.g. 10→4)
            #   odd:  (seq_len+1)//2 - 1  (e.g. 11→5)
            max_freq = seq_len // 2  # rfft → 去除直流分量后的频率分量数
            if top_k > max_freq:
                raise RouteError(
                    f"top_k ({top_k}) exceeds available frequency components "
                    f"({max_freq}) for seq_len={seq_len}. "
                    f"Reduce top_k or increase seq_len."
                )

        # FiLM-specific validation
        if model_type == "film":
            ws_str = params.get("window_size", "256")
            ws_list = [int(x.strip()) for x in ws_str.split(",") if x.strip()]
            for ws in ws_list:
                if ws > seq_len:
                    raise RouteError(
                        f"window_size ({ws}) exceeds seq_len ({seq_len}) "
                        f"for FiLM. Reduce window_size or increase seq_len."
                    )
            ms_str = params.get("multiscale", "1,2,4")
            ms_list = [int(x.strip()) for x in ms_str.split(",") if x.strip()]
            max_scale = max(ms_list)
            if max_scale * pred_len > seq_len:
                raise RouteError(
                    f"multiscale max ({max_scale}) * pred_len ({pred_len}) = "
                    f"{max_scale * pred_len} exceeds seq_len ({seq_len}) "
                    f"for FiLM. Reduce multiscale, reduce pred_len, or increase seq_len."
                )

        ts_params = {
            "time_series": True,
            "time_col": task_config.get("time_col"),
            "seq_len": seq_len,
            "pred_len": pred_len,
            "label_len": label_len,
            "time_granularity": task_config.get("time_granularity", "auto"),
        }

    split_result = split_data(df, target_col, test_size=test_size, pipeline=pipeline, **ts_params)

    # Models with internal normalization (e.g. FiLM) handle normalization
    # themselves — skip external normalization entirely.
    model_cls = get_model_class(model_type)
    if getattr(model_cls, 'uses_internal_normalization', False):
        split_result['y_scaler'] = None
        sm.set_split(data_id, split_result)
        return

    norm_method = params.get("normalization", "none")
    if norm_method in ("minmax", "mean"):
        X_train, X_test, norm_params = normalize_data(
            split_result["X_train"], split_result["X_test"], method=norm_method
        )
        split_result["X_train"] = X_train
        split_result["X_test"] = X_test
        split_result["norm_params"] = norm_params

        # Large pipeline: apply same normalization to dec_inp
        if pipeline == "large" and "dec_inp_train" in split_result:
            d_tr, d_te = split_result["dec_inp_train"], split_result["dec_inp_test"]
            split_result["dec_inp_train"] = normalize_data_apply(d_tr, norm_params, norm_method)
            split_result["dec_inp_test"] = normalize_data_apply(d_te, norm_params, norm_method)

    # Always normalize target for regression so loss/metric are scale-independent
    if split_result["task_type"] == "regression":
        if pipeline == "large" and norm_method in ("minmax", "mean"):
            # Large-pipeline: extract the target column's statistics from the
            # precomputed norm_params (the target is the last feature column
            # in the decoder input) to keep both on the same scale.
            y_norm_result = normalize_target(
                split_result["y_train"], split_result["y_test"],
                method=norm_method,
                norm_params=norm_params,
                target_idx=split_result.get("target_idx", -1),
            )
        elif pipeline == "large":
            y_norm_result = split_result["y_train"], split_result["y_test"], None
        else:
            y_norm = norm_method if norm_method in ("minmax", "mean") else "mean"
            y_norm_result = normalize_target(
                split_result["y_train"], split_result["y_test"], method=y_norm)
        split_result["y_train"], split_result["y_test"], split_result["y_scaler"] = y_norm_result

    sm.set_split(data_id, split_result)


def _build_config(params, df):
    """Build model config dict from request params."""
    default_cfg = config.TRAINING
    model_type = params.get("model_type", "mlp")
    learning_rate = float(params.get("learning_rate", default_cfg["learning_rate"]))
    batch_size = int(params.get("batch_size", default_cfg["batch_size"]))
    epochs = int(params.get("epochs", default_cfg["epochs"]))
    patience = int(params.get("patience", default_cfg["patience"]))
    import torch
    device = config.parse_device(params.get("device", config.DEVICE))
    if not isinstance(device, list) and device.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError(
            f"CUDA is not available: the NVIDIA driver is too old (driver 535 supports CUDA 12.x, "
            f"but PyTorch was compiled with CUDA {torch.version.cuda}). "
            f"Please update your NVIDIA driver or install a PyTorch version matching your driver."
        )
    elif isinstance(device, list) and not torch.cuda.is_available():
        raise ValueError(
            f"Multi-GPU training requires CUDA, but the NVIDIA driver is too old "
            f"(driver 535 supports CUDA 12.x, PyTorch needs CUDA {torch.version.cuda})."
        )

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
