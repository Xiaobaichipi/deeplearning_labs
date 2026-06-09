import os
import uuid

from flask import Blueprint, current_app, jsonify, request
from werkzeug.utils import secure_filename

from utils import config
from utils.data_utils import clean_data, fill_missing, get_data_info, load_data
from utils.plot_utils import plot_correlation_heatmap, plot_data_distribution
from utils.session import (
    RouteError, allowed_file, ensure_data, get_data_id, get_sm, handle_errors, json_ok,
)

data_bp = Blueprint("data", __name__)


@data_bp.route("/api/upload", methods=["POST"])
@handle_errors
def upload_file():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400
    f = request.files["file"]
    if f.filename == "":
        return jsonify({"error": "Empty filename"}), 400
    if not allowed_file(f.filename):
        return jsonify({"error": "Format not supported. Allowed: .csv, .xls, .xlsx"}), 400

    filename = secure_filename(f.filename)
    if not filename or filename == ".":
        ext = os.path.splitext(f.filename)[1] or ".csv"
        filename = f"upload_{uuid.uuid4().hex[:8]}{ext}"
    data_id = get_data_id()
    upload_dir = current_app.config["UPLOAD_DIR"]
    file_dir = os.path.join(upload_dir, data_id)
    os.makedirs(file_dir, exist_ok=True)
    filepath = os.path.join(file_dir, filename)
    f.save(filepath)

    sm = get_sm()
    df = load_data(filepath)
    sm.set_data(data_id, df)
    info = get_data_info(df)
    info["filename"] = filename
    dist_images = plot_data_distribution(df)
    corr_img = plot_correlation_heatmap(df)
    info["distribution_images"] = dist_images
    info["correlation_image"] = corr_img
    return json_ok({"success": True, "data": info})


@data_bp.route("/api/data/info", methods=["GET"])
@handle_errors
def data_info():
    df = ensure_data(get_sm(), get_data_id())
    info = get_data_info(df)
    dist_images = plot_data_distribution(df)
    corr_img = plot_correlation_heatmap(df)
    info["distribution_images"] = dist_images
    info["correlation_image"] = corr_img
    return json_ok({"success": True, "data": info})


@data_bp.route("/api/data/clean", methods=["POST"])
@handle_errors
def data_clean():
    sm = get_sm()
    data_id = get_data_id()
    df = ensure_data(sm, data_id)
    params = request.get_json() or {}
    cleaned, report = clean_data(
        df,
        drop_duplicates=params.get("drop_duplicates", True),
        drop_columns=params.get("drop_columns"),
        handle_outliers=params.get("handle_outliers", False),
        outlier_method=params.get("outlier_method", "iqr"),
        outlier_factor=float(params.get("outlier_factor", 1.5)),
    )
    sm.set_data(data_id, cleaned)
    info = get_data_info(cleaned)
    info["report"] = report
    dist_images = plot_data_distribution(cleaned)
    corr_img = plot_correlation_heatmap(cleaned)
    info["distribution_images"] = dist_images
    info["correlation_image"] = corr_img
    return json_ok({"success": True, "data": info, "report": report})


@data_bp.route("/api/data/fill", methods=["POST"])
@handle_errors
def data_fill():
    sm = get_sm()
    data_id = get_data_id()
    df = ensure_data(sm, data_id)
    params = request.get_json() or {}
    filled, report = fill_missing(
        df,
        strategy=params.get("strategy", "auto"),
        columns=params.get("columns"),
        fill_value=params.get("fill_value"),
    )
    sm.set_data(data_id, filled)
    info = get_data_info(filled)
    info["report"] = report
    dist_images = plot_data_distribution(filled)
    corr_img = plot_correlation_heatmap(filled)
    info["distribution_images"] = dist_images
    info["correlation_image"] = corr_img
    return json_ok({"success": True, "data": info, "report": report})


@data_bp.route("/api/data/task-config", methods=["GET"])
@handle_errors
def get_task_config():
    sm = get_sm()
    data_id = get_data_id()
    ensure_data(sm, data_id)
    config = sm.get_task_config(data_id) or {}
    return json_ok({"success": True, "task_config": config})


@data_bp.route("/api/data/task-config", methods=["POST"])
@handle_errors
def set_task_config():
    sm = get_sm()
    data_id = get_data_id()
    df = ensure_data(sm, data_id)
    params = request.get_json() or {}

    task_type = params.get("task_type", "general")
    if task_type not in ("general", "time_series"):
        return jsonify({"error": "task_type must be 'general' or 'time_series'"}), 400

    new_config = {
        "task_type": task_type,
        "time_col": params.get("time_col", ""),
        "seq_len": int(params.get("seq_len", config.TIME_SERIES["seq_len"])),
        "pred_len": int(params.get("pred_len", config.TIME_SERIES["pred_len"])),
        "label_len": int(params.get("label_len", config.TIME_SERIES["label_len"])),
        "time_granularity": params.get("time_granularity", "auto"),
    }

    # Clear downstream state if config changed
    old_config = sm.get_task_config(data_id) or {}
    changed = (
        old_config.get("task_type") != task_type
        or old_config.get("time_col") != new_config.get("time_col")
        or old_config.get("seq_len") != new_config.get("seq_len")
        or old_config.get("pred_len") != new_config.get("pred_len")
        or old_config.get("label_len") != new_config.get("label_len")
        or old_config.get("time_granularity") != new_config.get("time_granularity")
    )
    if changed:
        # Clear downstream state — remove keys rather than setting None
        # so has_model() returns False for the caller
        sm._splits.pop(data_id, None)
        sm._models.pop(data_id, None)
        sm._histories.pop(data_id, None)
        sm._model_configs.pop(data_id, None)

    sm.set_task_config(data_id, new_config)
    return json_ok({"success": True, "task_config": new_config})


@data_bp.route("/api/data/sample", methods=["GET"])
@handle_errors
def data_sample():
    df = ensure_data(get_sm(), get_data_id())
    return json_ok({
        "success": True,
        "columns": list(df.columns),
        "rows": df.head(100).to_dict(orient="records"),
        "total_rows": len(df),
        "total_cols": len(df.columns),
    })
