import os
import uuid

from flask import Blueprint, current_app, jsonify, request
from werkzeug.utils import secure_filename

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
