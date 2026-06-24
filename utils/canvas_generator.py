"""
Canvas-to-model generator — orchestrates graph validation, template-driven
code generation, file persistence, and model-registry management.

P1 scope: single-chain topology only.  Branching / graph topology deferred to P2.
Pipeline: ``"large"`` (4-arg forward: x_enc, x_mark_enc, x_dec, x_mark_dec).
"""

import os
import importlib.util

from utils.models import MODEL_REGISTRY
from .canvas_templates import (
    COMPONENT_TEMPLATES, COMPONENT_DEFAULTS,
    ComponentTemplate,
)
from .canvas_graph import CanvasError, validate_canvas


GENERATED_DIR = os.path.join(os.path.dirname(__file__), "models", "generated")


# ═══════════════════════════════════════════════════════════════════
# Code generation
# ═══════════════════════════════════════════════════════════════════

CLASS_SKELETON = """import torch
from utils.models.base import BaseModel
{imports}


class CanvasModel_{project_id}_v{version}(BaseModel):
    pipeline = "large"

    def __init__(self, input_dim, output_dim,
                 seq_len=None, label_len=0, pred_len=None,
                 n_time_features=4, **kwargs):
        super().__init__(input_dim, output_dim, **kwargs)
        self.pred_len = pred_len or 1
{init_body}

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
{forward_body}
        # Select last pred_len steps → (batch, pred_len, d_model)
        x = x[:, -self.pred_len:, :]
        return x
"""


def _resolve_global_params(ordered_nodes, input_dim, output_dim, n_time_features):
    """Extract global params (d_model, dropout) from the first component
    that defines them, then verify consistency across the chain."""
    d_model = None
    dropout = 0.1

    for node in ordered_nodes:
        cfg = node.get("config", {})
        defaults = COMPONENT_DEFAULTS.get(node["type"], {})
        dm = cfg.get("d_model") or defaults.get("d_model")
        if d_model is None and dm:
            d_model = dm
        dp = cfg.get("dropout") if cfg.get("dropout") is not None else defaults.get("dropout")
        if dp is not None:
            dropout = dp

    if d_model is None:
        d_model = 256

    return {
        "input_dim": input_dim,
        "output_dim": output_dim,
        "d_model": d_model,
        "dropout": dropout,
        "n_time_features": n_time_features,
    }


def generate_model_source(canvas, project_id, version, input_dim, output_dim, n_time_features=4):
    """Generate complete model class source code from a validated canvas.

    Returns (source_code, model_type_key).
    Raises CanvasError on validation failure.
    """
    ordered = validate_canvas(canvas)
    global_params = _resolve_global_params(ordered, input_dim, output_dim, n_time_features)
    d_model = global_params["d_model"]
    model_type = f"canvas_{project_id}_v{version}"

    init_lines = []
    fwd_lines = []
    all_imports = set()

    prev_var = "x_enc"

    for idx, node in enumerate(ordered):
        ntype = node["type"]
        tmpl = COMPONENT_TEMPLATES.get(ntype)
        if not tmpl:
            raise CanvasError(f"未知组件类型: '{ntype}' — 画布生成器尚未支持")

        for imp in tmpl.imports:
            all_imports.add(imp)

        var_name = f"canvas_{node['id']}"
        cfg = node.get("config", {})

        err = tmpl.check_config(cfg)
        if err:
            raise CanvasError(f"组件 '{node.get('label', node['id'])}' 配置错误: {err}")

        try:
            if idx == 0 and ntype != "embedding":
                all_imports.add("import torch.nn as nn")
                init_lines.insert(0,
                    "# Input projection — map input_dim → d_model\n"
                    f"self.input_proj = nn.Linear(input_dim, {global_params['d_model']})"
                )
                fwd_proj = "x = self.input_proj(x_enc)"
                fwd_code = tmpl.build_forward(var_name, "x")
                fwd_lines.append(fwd_proj)
            else:
                fwd_code = tmpl.build_forward(var_name, prev_var)

            init_code = tmpl.build_init(node["id"], var_name, cfg, global_params, component_type=ntype)
        except KeyError as e:
            raise CanvasError(
                f"组件 '{node.get('label', node['id'])}' 缺少参数: {e}")

        init_lines.append(init_code)
        fwd_lines.append(fwd_code)
        prev_var = "x"

    imports_str = "\n".join(sorted(all_imports))

    def _indent_lines(lines, margin=8):
        out = []
        for chunk in lines:
            for line in chunk.splitlines():
                if line.strip():
                    out.append(" " * margin + line)
                else:
                    out.append("")
        return "\n".join(out)

    init_body_str = _indent_lines(init_lines)
    forward_body_str = _indent_lines(fwd_lines)

    source = CLASS_SKELETON.format(
        project_id=project_id,
        version=version,
        imports=imports_str,
        init_body="\n" + init_body_str,
        forward_body=forward_body_str,
    )

    return source, model_type


# ═══════════════════════════════════════════════════════════════════
# File persistence
# ═══════════════════════════════════════════════════════════════════

def write_model_file(source, project_id, version):
    """Write generated model source to disk and return the file path."""
    os.makedirs(GENERATED_DIR, exist_ok=True)
    filename = f"canvas_{project_id}_v{version}.py"
    filepath = os.path.join(GENERATED_DIR, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(source)
    return filepath


# ═══════════════════════════════════════════════════════════════════
# Model registration & lifecycle
# ═══════════════════════════════════════════════════════════════════

def register_model(filepath, model_type, project_id="", version=1, display_name=None):
    """Import a generated model file and register it in MODEL_REGISTRY.

    Returns the model class.
    """
    module_name = f"utils.models.generated.{model_type}"
    spec = importlib.util.spec_from_file_location(module_name, filepath)
    if spec is None or spec.loader is None:
        raise CanvasError(f"无法加载生成的模型文件: {filepath}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    model_class = _find_model_class(module)
    if model_class is None:
        raise CanvasError("生成的模型文件中未找到有效的模型类")

    pid_short = project_id[:8] if project_id else "unknown"
    name = display_name or f"Canvas ({pid_short}... v{version})"
    MODEL_REGISTRY[model_type] = {
        "class": model_class,
        "name": name,
        "pipeline": "large",
        "params": {},
    }

    return model_class


def _find_model_class(module):
    """Find the BaseModel subclass in a generated module."""
    from utils.models.base import BaseModel
    for attr_name in dir(module):
        attr = getattr(module, attr_name)
        if isinstance(attr, type) and issubclass(attr, BaseModel) and attr is not BaseModel:
            return attr
    return None


def list_generated_for_project(project_id):
    """Return list of dicts [{model_type, version}, ...] for a project."""
    if not os.path.isdir(GENERATED_DIR):
        return []
    result = []
    prefix = f"canvas_{project_id}_v"
    for fname in os.listdir(GENERATED_DIR):
        if not fname.startswith(prefix) or not fname.endswith(".py"):
            continue
        stem = fname[:-3]
        version_str = stem[len(f"canvas_{project_id}_v"):]
        try:
            version = int(version_str)
        except ValueError:
            version = 1
        result.append({"model_type": stem, "version": version})
    return sorted(result, key=lambda x: x["version"])


def register_all_generated():
    """Scan GENERATED_DIR and re-register all generated model files.
    Called at startup so models survive server restarts.
    """
    if not os.path.isdir(GENERATED_DIR):
        return []
    registered = []
    for fname in sorted(os.listdir(GENERATED_DIR)):
        if not fname.startswith("canvas_") or not fname.endswith(".py"):
            continue
        stem = fname[:-3]
        filepath = os.path.join(GENERATED_DIR, fname)

        try:
            module = importlib.import_module(f"utils.models.generated.{stem}")
            importlib.reload(module)
        except ImportError:
            spec = importlib.util.spec_from_file_location(f"utils.models.generated.{stem}", filepath)
            if spec is None or spec.loader is None:
                continue
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

        model_class = _find_model_class(module)
        if model_class is None:
            continue

        inner = stem[len("canvas_"):]
        parts = inner.rsplit("_v", 1)
        project_id = parts[0] if len(parts) == 2 else inner
        version = int(parts[1]) if len(parts) == 2 and parts[1].isdigit() else 1

        model_type = stem
        pid_short = project_id[:8] if project_id else "unknown"
        MODEL_REGISTRY.setdefault(model_type, {
            "class": model_class,
            "name": model_type,
            "pipeline": "large",
            "params": {},
        })
        registered.append(model_type)

    return registered


def unregister_model(model_type):
    """Remove a generated model from MODEL_REGISTRY and delete its .py file.

    Returns True on success, False if model_type is not a canvas-generated model.
    """
    if not model_type.startswith("canvas_"):
        return False

    MODEL_REGISTRY.pop(model_type, None)

    filepath = os.path.join(GENERATED_DIR, f"{model_type}.py")
    if os.path.isfile(filepath):
        os.remove(filepath)

    pycache = os.path.join(GENERATED_DIR, "__pycache__")
    if os.path.isdir(pycache):
        for fname in os.listdir(pycache):
            if fname.startswith(model_type) and fname.endswith((".pyc", ".pyo")):
                os.remove(os.path.join(pycache, fname))

    return True
