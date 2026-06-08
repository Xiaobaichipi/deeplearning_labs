from flask import Blueprint, Response, jsonify, request

from utils import config
from utils.data_utils import denormalize_target
from utils.model_utils import cross_validate_model, evaluate, predict
from utils.plot_utils import plot_pred_vs_true, plot_pred_vs_true_line
from utils.session import (
    RouteError, get_data_id, get_sm, handle_errors, json_ok,
)

evaluation_bp = Blueprint("evaluation", __name__)


@evaluation_bp.route("/api/evaluate", methods=["POST"])
@handle_errors
def api_evaluate():
    sm = get_sm()
    data_id = get_data_id()
    if not sm.has_model(data_id):
        raise RouteError("Model not trained yet")
    model = sm.get_model(data_id)
    split_result = sm.get_split(data_id)
    result = evaluate(
        model,
        split_result["X_test"], split_result["y_test"],
        split_result["task_type"],
        target_encoder=split_result.get("target_encoder"),
        device="cpu",
    )
    return json_ok({"success": True, "evaluation": result, "task_type": split_result["task_type"]})


@evaluation_bp.route("/api/predict", methods=["POST"])
@handle_errors
def api_predict():
    sm = get_sm()
    data_id = get_data_id()
    if not sm.has_model(data_id):
        raise RouteError("Model not trained yet")
    params = request.get_json() or {}
    use_test = params.get("use_test", True)
    preds, y_true, probs, target_encoder, task_type = _compute_predictions(
        sm, data_id, use_test,
    )

    # Plot: pred vs true (regression only)
    plot_img = None
    line_plot_img = None
    if task_type == "regression":
        plot_img = plot_pred_vs_true(y_true[:len(preds)], preds)
        line_plot_img = plot_pred_vs_true_line(y_true[:len(preds)], preds)

    results = []
    for i in range(min(len(preds), 100)):
        row = {
            "index": int(i),
            "prediction": int(preds[i]) if task_type == "classification" else float(preds[i]),
        }
        if target_encoder:
            row["prediction_label"] = str(target_encoder.inverse_transform([int(preds[i])])[0])
            row["true_label"] = str(target_encoder.inverse_transform([int(y_true[i])])[0])
        row["true_value"] = float(y_true[i]) if task_type == "regression" else int(y_true[i])
        if probs is not None:
            row["probabilities"] = [float(p) for p in probs[i]]
        results.append(row)

    return json_ok({
        "success": True,
        "predictions": results,
        "task_type": task_type,
        "count": len(preds),
        "plot_image": plot_img,
        "line_plot_image": line_plot_img,
    })


@evaluation_bp.route("/api/predict/download")
@handle_errors
def predict_download():
    sm = get_sm()
    data_id = get_data_id()
    if not sm.has_model(data_id):
        raise RouteError("Model not trained yet")
    use_test = request.args.get("source", "test") == "test"
    fmt = request.args.get("format", "csv")
    preds, y_true, probs, target_encoder, task_type = _compute_predictions(
        sm, data_id, use_test,
    )

    import io
    import pandas as pd

    data = {"true_value": y_true, "prediction": preds}
    if probs is not None:
        data["confidence"] = probs.max(axis=1) if probs.ndim > 1 else probs

    df = pd.DataFrame(data)
    suffix = "test" if use_test else "train"
    output = io.BytesIO()

    if fmt == "csv":
        df.to_csv(output, index_label="index")
        output.seek(0)
        return Response(
            output.getvalue(),
            mimetype="text/csv",
            headers={"Content-Disposition": f"attachment;filename=predictions_{suffix}.csv"},
        )
    else:
        try:
            df.to_excel(output, index_label="index", engine="openpyxl")
        except ImportError:
            return jsonify({"error": "openpyxl not installed; use format=csv instead"}), 400
        output.seek(0)
        return Response(
            output.getvalue(),
            mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": f"attachment;filename=predictions_{suffix}.xlsx"},
        )


def _compute_predictions(sm, data_id, use_test):
    """Shared: run inference, denormalize, return (preds, y_true, probs, target_encoder, task_type)."""
    from utils.model_utils import predict
    split_result = sm.get_split(data_id)
    model = sm.get_model(data_id)

    if use_test:
        X = split_result["X_test"]
        y_true = split_result["y_test"]
    else:
        X = split_result["X_train"]
        y_true = split_result["y_train"]

    preds, probs = predict(model, X, split_result["task_type"], device="cpu")
    target_encoder = split_result.get("target_encoder")
    y_scaler = split_result.get("y_scaler")

    if split_result["task_type"] == "regression":
        preds = denormalize_target(preds, y_scaler)
        y_true = denormalize_target(y_true, y_scaler)

    return preds, y_true, probs, target_encoder, split_result["task_type"]


@evaluation_bp.route("/api/validate", methods=["POST"])
@handle_errors
def api_validate():
    sm = get_sm()
    data_id = get_data_id()
    if not sm.has_model(data_id):
        raise RouteError("Model not trained yet")
    split_result = sm.get_split(data_id)
    model_config = sm.get_model_config(data_id)
    if not model_config:
        raise RouteError("Model configuration not found; re-train the model")

    params = request.get_json() or {}
    n_splits = int(params.get("n_splits", config.CV["default_folds"]))
    output_dim = (
        split_result["n_classes"]
        if split_result["task_type"] == "classification"
        else 1
    )

    result = cross_validate_model(
        model_type=model_config["model_type"],
        input_dim=split_result["input_dim"],
        output_dim=output_dim,
        X=split_result["X_train"],
        y=split_result["y_train"],
        task_type=split_result["task_type"],
        model_params=model_config["model_params"],
        n_splits=n_splits,
        epochs=config.CV["max_epochs_per_fold"],
        batch_size=model_config.get("batch_size", config.TRAINING["batch_size"]),
        lr=model_config.get("learning_rate", config.TRAINING["learning_rate"]),
        device=model_config.get("device", config.DEVICE),
    )

    return json_ok({"success": True, **result})
