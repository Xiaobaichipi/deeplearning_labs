from flask import Blueprint, current_app, jsonify, request
from utils.data_utils import normalize_data, split_data
from utils.model_utils import create_model, train_model
from utils.models import get_model_params
from utils.plot_utils import plot_training_history
from utils.session import get_data_id, json_ok

training_bp = Blueprint("training", __name__)


@training_bp.route("/api/train", methods=["POST"])
def train():
    sm = current_app.config["session_manager"]
    data_id = get_data_id()
    df = sm.get_data(data_id)
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

        norm_method = params.get("normalization", "none")
        if norm_method in ("minmax", "mean"):
            X_train, X_test, norm_params = normalize_data(
                split_result["X_train"], split_result["X_test"], method=norm_method
            )
            split_result["X_train"] = X_train
            split_result["X_test"] = X_test
            split_result["norm_params"] = norm_params

        sm.set_split(data_id, split_result)

        model_type = params.get("model_type", "mlp")
        learning_rate = float(params.get("learning_rate", 0.001))
        batch_size = int(params.get("batch_size", 32))
        epochs = int(params.get("epochs", 50))
        patience = int(params.get("patience", 10))

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

        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"

        input_dim = split_result["input_dim"]
        output_dim = (
            split_result["n_classes"]
            if split_result["task_type"] == "classification"
            else 1
        )

        model = create_model(model_type, input_dim, output_dim, **model_params)
        trained_model, history = train_model(
            model,
            split_result["X_train"], split_result["y_train"],
            split_result["X_test"], split_result["y_test"],
            split_result["task_type"],
            epochs=epochs, batch_size=batch_size, lr=learning_rate,
            patience=patience, device=device,
        )

        sm.set_model(data_id, trained_model)
        sm.set_history(data_id, history)

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
