import os
from flask import Flask, jsonify, render_template, session

from routes import data_bp, evaluation_bp, projects_bp, training_bp
from utils.project_manager import ProjectManager
from utils.session import SessionManager


app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "deeplearning-labs-dev-key-2024")

UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "uploads")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
app.config["UPLOAD_DIR"] = UPLOAD_DIR
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024
app.config["session_manager"] = SessionManager(UPLOAD_DIR)

PROJECTS_DIR = os.path.join(os.path.dirname(__file__), "projects")
app.config["PROJECTS_DIR"] = PROJECTS_DIR
app.config["project_manager"] = ProjectManager(PROJECTS_DIR)

app.register_blueprint(data_bp)
app.register_blueprint(training_bp)
app.register_blueprint(evaluation_bp)
app.register_blueprint(projects_bp)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/models-guide")
def models_guide():
    return render_template("models_guide.html")


@app.route("/api/reset", methods=["POST"])
def api_reset():
    sm = app.config["session_manager"]
    if "data_id" in session:
        sm.reset(session["data_id"])
        session.pop("data_id", None)
    session.pop("active_project_id", None)
    return jsonify({"success": True})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True, threaded=True)
