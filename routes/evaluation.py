from flask import Blueprint, current_app, jsonify, request

from utils import config
from utils.data_utils import denormalize_target
from utils.model_utils import cross_validate_model, evaluate, predict
from utils.session import get_data_id, json_ok

evaluation_bp = Blueprint("evaluation", __name__)


@evaluation_bp.route("/api/evaluate", methods=["POST"])
def api_evaluate():
    sm = current_app.config["session_manager"]
    data_id = get_data_id()
    if not sm.has_model(data_id):
        return jsonify({"error": "Model not trained yet"}), 400
    try:
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
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@evaluation_bp.route("/api/predict", methods=["POST"])
def api_predict():
    sm = current_app.config["session_manager"]
    data_id = get_data_id()
    if not sm.has_model(data_id):
        return jsonify({"error": "Model not trained yet"}), 400
    try:
        split_result = sm.get_split(data_id)
        model = sm.get_model(data_id)
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
        y_scaler = split_result.get("y_scaler")

        # Denormalize regression predictions back to original scale
        if split_result["task_type"] == "regression":
            preds = denormalize_target(preds, y_scaler)
            y_true = denormalize_target(y_true, y_scaler)

        results = []
        for i in range(min(len(preds), 100)):
            row = {
                "index": int(i),
                "prediction": int(preds[i]) if split_result["task_type"] == "classification" else float(preds[i]),
            }
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


@evaluation_bp.route("/api/validate", methods=["POST"])
def api_validate():
    sm = current_app.config["session_manager"]
    data_id = get_data_id()
    if not sm.has_model(data_id):
        return jsonify({"error": "Model not trained yet"}), 400
    try:
        split_result = sm.get_split(data_id)
        model_config = sm.get_model_config(data_id)
        if not model_config:
            return jsonify({"error": "Model configuration not found; re-train the model"}), 400

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
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
