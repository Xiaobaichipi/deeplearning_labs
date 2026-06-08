"""Project CRUD and activation routes."""

import os
import tempfile
from flask import Blueprint, current_app, jsonify, request, session, send_file
from utils.data_utils import load_data, get_data_info
from utils.session import allowed_file, get_data_id, json_ok

projects_bp = Blueprint("projects", __name__)


def _pm():
    return current_app.config["project_manager"]


@projects_bp.route("/api/projects", methods=["GET"])
def list_projects():
    return json_ok({"projects": _pm().list_projects()})


@projects_bp.route("/api/projects", methods=["POST"])
def create_project():
    name = request.form.get("name", "").strip()
    if not name:
        return jsonify({"error": "Project name is required"}), 400

    pm = _pm()
    project_id = pm.create_project(name)

    file = request.files.get("file")
    if file and file.filename:
        ext = os.path.splitext(file.filename)[1].lower()
        if ext not in (".csv", ".xls", ".xlsx"):
            pm.delete_project(project_id)
            return jsonify({"error": "Format not supported"}), 400
        try:
            with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
                file.save(tmp.name)
                tmp_path = tmp.name
            try:
                df = load_data(tmp_path)
            finally:
                os.unlink(tmp_path)
            pm.save_dataset(project_id, df, file.filename)
        except Exception as e:
            pm.delete_project(project_id)
            return jsonify({"error": str(e)}), 400

    return json_ok({"project": pm.get_project(project_id)}), 201


@projects_bp.route("/api/projects/<project_id>", methods=["GET"])
def get_project(project_id):
    project = _pm().get_project(project_id)
    if not project:
        return jsonify({"error": "Project not found"}), 404
    return json_ok({"project": project})


@projects_bp.route("/api/projects/<project_id>", methods=["DELETE"])
def delete_project(project_id):
    _pm().delete_project(project_id)
    return json_ok({"success": True})


@projects_bp.route("/api/projects/<project_id>/models", methods=["GET"])
def list_project_models(project_id):
    """List trained models with full metadata for a project."""
    pm = _pm()
    project = pm.get_project(project_id)
    if not project:
        return jsonify({"error": "Project not found"}), 404
    models = pm.list_models(project_id)
    return json_ok({"models": models})


@projects_bp.route("/api/projects/<project_id>/models/<model_id>/export", methods=["GET"])
def export_model(project_id, model_id):
    """Download a model's state_dict.pt with custom filename."""
    pm = _pm()
    state_dict, meta = pm.load_model(project_id, model_id)
    if state_dict is None:
        return jsonify({"error": "Model not found"}), 404

    name = request.args.get("name", "").strip()
    if not name:
        name = f"{meta.get('model_type', 'model')}_{model_id}"
    if not name.endswith(".pt"):
        name += ".pt"

    state_path = os.path.join(
        current_app.config["PROJECTS_DIR"],
        project_id, "models", model_id, "state_dict.pt")
    if not os.path.isfile(state_path):
        return jsonify({"error": "Model file not found"}), 404
    return send_file(state_path, as_attachment=True, download_name=name)


@projects_bp.route("/api/projects/<project_id>/models/compare", methods=["POST"])
def compare_models(project_id):
    """Run predictions on test data with multiple models and render comparison chart."""
    pm = _pm()
    project = pm.get_project(project_id)
    if not project:
        return jsonify({"error": "Project not found"}), 404

    model_ids = (request.get_json() or {}).get("model_ids", [])
    if not model_ids:
        return jsonify({"error": "No model_ids provided"}), 400

    split_result = pm.load_split(project_id)
    if split_result is None:
        return jsonify({"error": "No split data found; train a model first"}), 400

    task_type = split_result.get("task_type")
    if task_type != "regression":
        return jsonify({"error": "Model comparison is only supported for regression"}), 400

    X_test = split_result["X_test"]
    y_true = split_result["y_test"]
    y_scaler = split_result.get("y_scaler")
    input_dim = split_result.get("input_dim", 1)
    output_dim = 1

    from utils.data_utils import denormalize_target
    from utils.model_utils import create_model, predict
    from utils.plot_utils import plot_model_comparison

    predictions_dict = {}
    loaded_count = 0
    for mid in model_ids:
        state_dict, meta = pm.load_model(project_id, mid)
        if state_dict is None:
            continue
        try:
            model = create_model(
                meta["model_type"], input_dim, output_dim,
                **meta.get("model_params", {}),
            )
            model.load_state_dict(state_dict)
            preds, _ = predict(model, X_test, task_type, device="cpu")
            preds = denormalize_target(preds, y_scaler)
            label = f"{meta.get('model_type', 'model')} ({mid})"
            predictions_dict[label] = preds
            loaded_count += 1
        except Exception:
            pass

    if loaded_count == 0:
        return jsonify({"error": "Could not load any valid models"}), 400

    y_true_denorm = denormalize_target(y_true.copy(), y_scaler)
    plot_image = plot_model_comparison(y_true_denorm, predictions_dict)

    return json_ok({
        "success": True,
        "plot_image": plot_image,
        "model_ids": model_ids,
        "loaded_count": loaded_count,
    })


@projects_bp.route("/api/projects/<project_id>/load-model/<model_id>", methods=["POST"])
def load_model_into_session(project_id, model_id):
    """Load a saved model into SessionManager primary slot for evaluation/prediction."""
    sm = current_app.config["session_manager"]
    pm = _pm()
    data_id = get_data_id()

    project = pm.get_project(project_id)
    if not project:
        return jsonify({"error": "Project not found"}), 404

    state_dict, meta = pm.load_model(project_id, model_id)
    if state_dict is None:
        return jsonify({"error": "Model not found"}), 404

    split_result = pm.load_split(project_id)
    if split_result:
        sm.set_split(data_id, split_result)
        input_dim = split_result.get("input_dim", len(meta.get("feature_names", [])))
        output_dim = (
            split_result["n_classes"]
            if split_result.get("task_type") == "classification"
            else 1
        )
    else:
        input_dim = len(meta.get("feature_names", []))
        output_dim = 1

    from utils.model_utils import create_model
    try:
        model = create_model(
            meta["model_type"], input_dim, output_dim,
            **meta.get("model_params", {}),
        )
        model.load_state_dict(state_dict)
        model.eval()
    except Exception as e:
        return jsonify({"error": f"Failed to reconstruct model: {str(e)}"}), 500

    sm.set_model(data_id, model)
    sm.set_model_config(data_id, meta.get("model_params", {}))

    final_metrics = meta.get("final_metrics", {})
    if final_metrics:
        sm.set_history(data_id, final_metrics)

    return json_ok({
        "success": True,
        "model": {
            "id": model_id,
            "model_type": meta.get("model_type"),
            "model_params": meta.get("model_params", {}),
            "final_metrics": final_metrics,
            "task_type": meta.get("task_type"),
            "feature_names": meta.get("feature_names", []),
            "target_name": meta.get("target_name"),
            "train_size": meta.get("train_size"),
            "test_size": meta.get("test_size"),
        },
    })


@projects_bp.route("/api/projects/<project_id>/activate", methods=["POST"])
def activate_project(project_id):
    """Load project data + split into SessionManager so the UI can resume."""
    sm = current_app.config["session_manager"]
    pm = _pm()
    data_id = get_data_id()

    project = pm.get_project(project_id)
    if not project:
        return jsonify({"error": "Project not found"}), 404

    df = pm.load_dataset(project_id)
    if df is None:
        return jsonify({"error": "No dataset found for this project"}), 400
    sm.set_data(data_id, df)

    session["active_project_id"] = project_id

    split_result = pm.load_split(project_id)
    if split_result:
        sm.set_split(data_id, split_result)

    # Reconstruct trained models
    models = pm.list_models(project_id)
    if split_result:
        input_dim = split_result.get("input_dim", 1)
        output_dim = (
            split_result["n_classes"]
            if split_result.get("task_type") == "classification"
            else 1
        )
        for m in models:
            mid = m["id"]
            state_dict, meta = pm.load_model(project_id, mid)
            if state_dict is None:
                continue
            try:
                from utils.model_utils import create_model
                model = create_model(
                    meta["model_type"], input_dim, output_dim,
                    **meta.get("model_params", {}),
                )
                model.load_state_dict(state_dict)
                sm.set_model(f"{data_id}_{mid}", model)
            except Exception:
                pass

    # Load latest model into primary session slot
    if split_result and models:
        latest = models[0]
        state_dict, meta = pm.load_model(project_id, latest["id"])
        if state_dict is not None:
            try:
                model = create_model(
                    meta["model_type"], input_dim, output_dim,
                    **meta.get("model_params", {}),
                )
                model.load_state_dict(state_dict)
                model.eval()
                sm.set_model(data_id, model)
            except Exception:
                pass

    info = get_data_info(df)
    info["project"] = project
    info["models"] = [
        {"id": m["id"], "model_type": m.get("model_type"),
         "created_at": m.get("created_at")}
        for m in models
    ]
    return json_ok({"success": True, "data": info})
