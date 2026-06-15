"""Persistent project management — datasets, splits, and models on disk."""

import json
import os
import shutil
import uuid
from datetime import datetime

import numpy as np
import pandas as pd
import torch


class ProjectManager:
    """Manages project directories under *projects_dir*.

    Each project lives at ``projects/<project_id>/``::

        config.json          — name, timestamps, model_count
        dataset/data.csv     — uploaded file (always CSV internally)
        splits/latest.json   — most recent train/test split (numpy→JSON)
        models/<model_id>/
            state_dict.pt    — torch.save(model.state_dict())
            config.json      — hyperparams, history, eval_metrics
    """

    def __init__(self, projects_dir: str):
        self._projects_dir = projects_dir
        os.makedirs(projects_dir, exist_ok=True)

    # ── paths ─────────────────────────────────────────────────────────────────

    def _pp(self, project_id: str, *parts: str) -> str:
        return os.path.join(self._projects_dir, project_id, *parts)

    # ── project CRUD ──────────────────────────────────────────────────────────

    def list_projects(self) -> list[dict]:
        projects = []
        if not os.path.isdir(self._projects_dir):
            return projects
        for name in os.listdir(self._projects_dir):
            config_path = self._pp(name, "config.json")
            if os.path.isfile(config_path):
                with open(config_path, encoding="utf-8") as f:
                    info = json.load(f)
                info["id"] = name
                projects.append(info)
        projects.sort(key=lambda p: p.get("updated_at", ""), reverse=True)
        return projects

    def create_project(self, name: str, original_filename: str = "") -> str:
        project_id = uuid.uuid4().hex[:12]
        path = self._pp(project_id)
        os.makedirs(path, exist_ok=True)
        now = datetime.now().isoformat()
        config = {
            "name": name,
            "original_filename": original_filename,
            "created_at": now,
            "updated_at": now,
            "model_count": 0,
        }
        with open(self._pp(project_id, "config.json"), "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        return project_id

    def get_project(self, project_id: str) -> dict | None:
        config_path = self._pp(project_id, "config.json")
        if not os.path.isfile(config_path):
            return None
        with open(config_path, encoding="utf-8") as f:
            info = json.load(f)
        info["id"] = project_id
        return info

    def update_project_config(self, project_id: str, **updates):
        config_path = self._pp(project_id, "config.json")
        if not os.path.isfile(config_path):
            return
        with open(config_path, encoding="utf-8") as f:
            config = json.load(f)
        config.update(updates)
        config["updated_at"] = datetime.now().isoformat()
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2)

    def delete_project(self, project_id: str):
        path = self._pp(project_id)
        if os.path.isdir(path):
            shutil.rmtree(path)

    # ── dataset ───────────────────────────────────────────────────────────────

    def save_dataset(self, project_id: str, df: pd.DataFrame, filename: str = ""):
        ds_dir = self._pp(project_id, "dataset")
        os.makedirs(ds_dir, exist_ok=True)
        df.to_csv(os.path.join(ds_dir, "data.csv"), index=False)
        if filename:
            self.update_project_config(project_id, original_filename=filename)

    def load_dataset(self, project_id: str) -> pd.DataFrame | None:
        csv_path = self._pp(project_id, "dataset", "data.csv")
        if not os.path.isfile(csv_path):
            return None
        return pd.read_csv(csv_path)

    # ── split ─────────────────────────────────────────────────────────────────

    def save_split(self, project_id: str, split_result: dict):
        splits_dir = self._pp(project_id, "splits")
        os.makedirs(splits_dir, exist_ok=True)

        serializable = {}
        for key, value in split_result.items():
            if key == "target_encoder":
                # Store sklearn encoder classes
                if hasattr(value, "classes_"):
                    serializable["target_encoder"] = {
                        "_type": "LabelEncoder",
                        "classes_": value.classes_.tolist(),
                    }
                continue
            if isinstance(value, np.ndarray):
                serializable[key] = value.tolist()
                serializable[f"_{key}_dtype"] = str(value.dtype)
                serializable[f"_{key}_shape"] = list(value.shape)
            elif isinstance(value, np.integer):
                serializable[key] = int(value)
            elif isinstance(value, np.floating):
                serializable[key] = float(value)
            elif isinstance(value, np.bool_):
                serializable[key] = bool(value)
            else:
                serializable[key] = value

        with open(os.path.join(splits_dir, "latest.json"), "w", encoding="utf-8") as f:
            json.dump(serializable, f, ensure_ascii=False, indent=2)

    def load_split(self, project_id: str) -> dict | None:
        json_path = self._pp(project_id, "splits", "latest.json")
        if not os.path.isfile(json_path):
            return None
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)

        # Reconstruct numpy arrays
        for key in list(data.keys()):
            dtype_key = f"_{key}_dtype"
            shape_key = f"_{key}_shape"
            if dtype_key in data and shape_key in data:
                arr = np.array(data[key], dtype=data[dtype_key])
                if data[shape_key]:
                    arr = arr.reshape(data[shape_key])
                data[key] = arr

        # Reconstruct target_encoder
        enc = data.get("target_encoder")
        if isinstance(enc, dict) and enc.get("_type") == "LabelEncoder":
            try:
                from sklearn.preprocessing import LabelEncoder
                le = LabelEncoder()
                le.fit(enc["classes_"])
                data["target_encoder"] = le
            except Exception:
                data["target_encoder"] = None

        # Cleanup metadata keys
        for key in list(data.keys()):
            if key.startswith("_"):
                del data[key]

        return data

    # ── model ─────────────────────────────────────────────────────────────────

    def next_model_id(self, project_id: str) -> str:
        models_dir = self._pp(project_id, "models")
        if not os.path.isdir(models_dir):
            os.makedirs(models_dir, exist_ok=True)
        existing = [d for d in os.listdir(models_dir)
                    if os.path.isdir(os.path.join(models_dir, d))]
        return f"model_{len(existing) + 1:03d}"

    def save_model(self, project_id: str, model_id: str,
                   state_dict: dict, meta: dict):
        model_dir = self._pp(project_id, "models", model_id)
        os.makedirs(model_dir, exist_ok=True)

        torch.save(state_dict, os.path.join(model_dir, "state_dict.pt"))

        meta["model_id"] = model_id
        meta.setdefault("created_at", datetime.now().isoformat())
        with open(os.path.join(model_dir, "config.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        # Bump model_count in project config
        count = len([d for d in os.listdir(self._pp(project_id, "models"))
                     if os.path.isdir(os.path.join(self._pp(project_id, "models"), d))])
        self.update_project_config(project_id, model_count=count)

    def load_model(self, project_id: str, model_id: str):
        model_dir = self._pp(project_id, "models", model_id)
        meta_path = os.path.join(model_dir, "config.json")
        state_path = os.path.join(model_dir, "state_dict.pt")
        if not os.path.isfile(meta_path) or not os.path.isfile(state_path):
            return None, None
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        state_dict = torch.load(state_path, map_location="cpu", weights_only=True)
        return state_dict, meta

    def list_models(self, project_id: str) -> list[dict]:
        models_dir = self._pp(project_id, "models")
        if not os.path.isdir(models_dir):
            return []
        models = []
        for name in os.listdir(models_dir):
            meta_path = os.path.join(models_dir, name, "config.json")
            if os.path.isfile(meta_path):
                with open(meta_path, encoding="utf-8") as f:
                    meta = json.load(f)
                meta["id"] = name
                models.append(meta)
        models.sort(key=lambda m: m.get("created_at", ""), reverse=True)
        return models
