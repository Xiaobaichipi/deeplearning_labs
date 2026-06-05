from flask import Blueprint, current_app, jsonify, request
from sklearn.model_selection import KFold, cross_val_score
from sklearn.neural_network import MLPClassifier, MLPRegressor

from utils.model_utils import evaluate, predict
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
        X_train = split_result["X_train"]
        y_train = split_result["y_train"]

        params = request.get_json() or {}
        n_splits = int(params.get("n_splits", 5))
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

        if split_result["task_type"] == "classification":
            scores = cross_val_score(
                MLPClassifier(
                    hidden_layer_sizes=(64, 32), max_iter=200,
                    random_state=42, early_stopping=True,
                ),
                X_train, y_train, cv=kf, scoring="accuracy",
            )
        else:
            scores = cross_val_score(
                MLPRegressor(
                    hidden_layer_sizes=(64, 32), max_iter=200,
                    random_state=42, early_stopping=True,
                ),
                X_train, y_train, cv=kf, scoring="r2",
            )

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
