"""
Canvas-to-model generator — validates graph topology, generates PyTorch model
code, and registers it in the model registry.

P1 scope: single-chain topology only.  Branching / graph topology deferred to P2.

Pipeline: "large" (4-arg forward: x_enc, x_mark_enc, x_dec, x_mark_dec).
"""

import os
import re
import json
import importlib.util

from utils.models import MODEL_REGISTRY

GENERATED_DIR = os.path.join(os.path.dirname(__file__), "models", "generated")


# ═══════════════════════════════════════════════════════════════════
# Template definitions — each component type knows how to render
# its __init__ and forward code blocks
# ═══════════════════════════════════════════════════════════════════

def _fmt(val):
    """Format a config value for code generation — strings get quotes,
    booleans/numeric/None stay bare."""
    if isinstance(val, str):
        return repr(val)
    if val is None:
        return "None"
    return str(val)


class ComponentTemplate:
    """One component type's code-generation rules."""

    def __init__(self, imports, init_tpl, forward_tpl, check_config=None):
        self.imports = imports          # list of import strings
        self.init_tpl = init_tpl        # string template with {var} / {id} / {config}
        self.forward_tpl = forward_tpl  # string template with {var} / {id} / {in_var}
        self.check_config = check_config or (lambda cfg: None)

    def build_init(self, comp_id, var_name, cfg, global_params, component_type=""):
        """Return Python code for the __init__ block of one component."""
        # Merge saved config with component defaults to fill missing fields
        defaults = COMPONENT_DEFAULTS.get(component_type, {})
        merged = {**defaults, **cfg}
        ctx = dict(merged)
        ctx["id"] = comp_id
        ctx["var"] = var_name
        ctx.update(global_params)
        return self.init_tpl.format(**ctx)

    def build_forward(self, var_name, in_var):
        """Return Python code for the forward call of one component."""
        return self.forward_tpl.format(var=var_name, in_var=in_var)


# ── Component registry — single source of truth for the frontend ──
# The frontend fetches this via GET /api/canvas/component-defaults.
# When adding/editing a component type, only update THIS dict.
# The COMPONENT_TEMPLATES below reference defaults from here.

COMPONENT_REGISTRY = {
    "encoder": {
        "label": "Encoder",
        "category": "基础模块",
        "color": "#4CAF50",
        "inputs": 1,
        "outputs": 1,
        "defaults": {"d_model": 512, "n_heads": 8, "d_ff": 2048, "dropout": 0.1, "activation": "gelu"},
        "ports": {"input": {"label": "输入"}, "output": {"label": "输出"}},
    },
    "decoder": {
        "label": "Decoder",
        "category": "基础模块",
        "color": "#2196F3",
        "inputs": 1,
        "outputs": 1,
        "defaults": {"d_model": 512, "n_heads": 8, "d_ff": 2048, "dropout": 0.1, "activation": "gelu"},
        "ports": {"input": {"label": "编码器输出"}, "output": {"label": "输出"}},
    },
    "embedding": {
        "label": "DataEmbedding",
        "category": "输入模块",
        "color": "#FF9800",
        "inputs": 1,
        "outputs": 1,
        "defaults": {"d_model": 512, "dropout": 0.1},
        "ports": {"input": {"label": "原始数据"}, "output": {"label": "嵌入向量"}},
    },
    "linear": {
        "label": "Linear",
        "category": "基础模块",
        "color": "#9C27B0",
        "inputs": 1,
        "outputs": 1,
        "defaults": {"in_features": 512, "out_features": 1},
        "ports": {"input": {"label": "输入"}, "output": {"label": "输出"}},
    },
}

# Backward-compat alias for code that reads COMPONENT_DEFAULTS
COMPONENT_DEFAULTS = {k: v["defaults"] for k, v in COMPONENT_REGISTRY.items()}


# ── Registered components ────────────────────────────────────────

COMPONENT_TEMPLATES = {
    "embedding": ComponentTemplate(
        imports=[
            "from utils.models.shared_layers.Embed import DataEmbedding",
        ],
        init_tpl=(
            "# DataEmbedding — project raw features into d_model space\n"
            "self.{var} = DataEmbedding(\n"
            "    c_in={input_dim}, d_model={d_model},\n"
            "    n_time_features={n_time_features}, dropout={dropout},\n"
            ")"
        ),
        forward_tpl=(
            "x = self.{var}({in_var}, x_mark_enc)"
        ),
    ),

    "encoder": ComponentTemplate(
        imports=[
            "import torch.nn as nn",
            "from utils.models.shared_layers.Transformer_EncDec import Encoder, EncoderLayer",
            "from utils.models.shared_layers.SelfAttention_Family import AttentionLayer, FullAttention",
        ],
        init_tpl=(
            "# Encoder — {n_heads}-head FullAttention, {d_ff} FFN, {activation}\n"
            "_attn = AttentionLayer(\n"
            "    FullAttention(attention_dropout={dropout}),\n"
            "    d_model={d_model}, n_heads={n_heads},\n"
            ")\n"
            "_layer = EncoderLayer(_attn, d_model={d_model}, d_ff={d_ff},\n"
            "                     dropout={dropout}, activation={activation!r})\n"
            "self.{var} = Encoder([_layer], norm_layer=nn.LayerNorm({d_model}))"
        ),
        forward_tpl=(
            "x, _ = self.{var}({in_var})"
        ),
    ),

    "decoder": ComponentTemplate(
        imports=[
            "import torch.nn as nn",
            "from utils.models.shared_layers.Transformer_EncDec import Decoder, DecoderLayer",
            "from utils.models.shared_layers.SelfAttention_Family import AttentionLayer, FullAttention",
        ],
        init_tpl=(
            "# Decoder — self-attention + cross-attention, {n_heads} heads\n"
            "_self_attn = AttentionLayer(\n"
            "    FullAttention(attention_dropout={dropout}),\n"
            "    d_model={d_model}, n_heads={n_heads},\n"
            ")\n"
            "_cross_attn = AttentionLayer(\n"
            "    FullAttention(attention_dropout={dropout}),\n"
            "    d_model={d_model}, n_heads={n_heads},\n"
            ")\n"
            "_layer = DecoderLayer(_self_attn, _cross_attn,\n"
            "                     d_model={d_model}, d_ff={d_ff},\n"
            "                     dropout={dropout}, activation={activation!r})\n"
            "self.{var} = Decoder([_layer], norm_layer=nn.LayerNorm({d_model}))"
        ),
        forward_tpl=(
            "x = self.{var}({in_var}, cross={in_var})"
        ),
    ),

    "linear": ComponentTemplate(
        imports=[
            "import torch.nn as nn",
        ],
        init_tpl=(
            "# Linear projection — map d_model → output dimension\n"
            "self.{var} = nn.Linear({d_model}, {output_dim})"
        ),
        forward_tpl=(
            "x = self.{var}({in_var})"
        ),
    ),
}


# ═══════════════════════════════════════════════════════════════════
# Graph validation
# ═══════════════════════════════════════════════════════════════════

class CanvasError(Exception):
    """Raised for validation / generation failures with a user-facing message."""


def _assert(cond, msg):
    """Abort generation with a clear error message."""
    if not cond:
        raise CanvasError(msg)


def validate_canvas(canvas):
    """Validate canvas structure — returns ordered node list for single chain.

    Raises CanvasError with user-facing message on any violation.
    """
    nodes = canvas.get("nodes", [])
    edges = canvas.get("edges", [])

    _assert(len(nodes) >= 2,
            "画布至少需要 2 个组件节点（例如 Embedding → Linear）")

    # ── 1. Build adjacency + in-degree maps ──────────────────────
    in_degree = {n["id"]: 0 for n in nodes}
    out_degree = {n["id"]: 0 for n in nodes}
    adj = {n["id"]: [] for n in nodes}

    for e in edges:
        frm, to = e["from"], e["to"]
        _assert(frm in in_degree,
                f"边的起点 '{frm}' 在画布中不存在")
        _assert(to in in_degree,
                f"边的终点 '{to}' 在画布中不存在")
        in_degree[to] += 1
        out_degree[frm] += 1
        adj[frm].append(to)

    _assert(all(d <= 1 for d in in_degree.values()),
            "当前只支持单链拓扑（每个节点最多一个输入连接）— 分支/汇合留到 P2")
    _assert(all(d <= 1 for d in out_degree.values()),
            "当前只支持单链拓扑（每个节点最多一个输出连接）— 分支留到 P2")

    # Save a copy of initial in_degree before Kahn's algorithm mutates it
    orig_in_degree = dict(in_degree)

    # ── 2. Topological sort (Kahn) ───────────────────────────────
    q = [nid for nid, d in in_degree.items() if d == 0]
    ordered = []
    while q:
        nid = q.pop(0)
        ordered.append(nid)
        for nb in adj[nid]:
            in_degree[nb] -= 1
            if in_degree[nb] == 0:
                q.append(nb)

    _assert(len(ordered) == len(nodes),
            "画布中存在循环连接，请检查连线是否成环")
    _assert(orig_in_degree[ordered[0]] == 0,
            "画布第一个节点必须有一个输入端口（无输入连接）")

    # ── 3. Check for isolated nodes — use pre-Kahn in_degree (orig_in_degree) ──
    for nid in ordered:
        if orig_in_degree.get(nid, 0) == 0 and nid != ordered[0]:
            node = next(n for n in nodes if n["id"] == nid)
            raise CanvasError(
                f"组件 '{node.get('label', nid)}' 没有输入连接（孤立节点）。"
                f"\n请将其连接到上游组件，或删除。")

    # ── 4. Validate d_model consistency ──────────────────────────
    d_model = None
    for nid in ordered:
        node = next(n for n in nodes if n["id"] == nid)
        cfg = node.get("config", {})
        ntype = node["type"]

        if ntype in ("encoder", "decoder", "linear"):
            dm = cfg.get("d_model")
            if d_model is None and dm is not None:
                d_model = dm
            if dm is not None and dm != d_model:
                raise CanvasError(
                    f"组件 '{node.get('label', nid)}' 的 d_model={dm} "
                    f"与前面组件的 d_model={d_model} 不一致。"
                    f"\n请确保所有 Encode/Decoder/Linear 的 d_model 相同。")

    return [next(n for n in nodes if n["id"] == nid) for nid in ordered]


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

    # Build init and forward blocks
    init_lines = []
    fwd_lines = []
    all_imports = set()

    prev_var = "x_enc"

    for idx, node in enumerate(ordered):
        ntype = node["type"]
        tmpl = COMPONENT_TEMPLATES.get(ntype)
        if not tmpl:
            raise CanvasError(f"未知组件类型: '{ntype}' — 画布生成器尚未支持")

        # Collect imports
        for imp in tmpl.imports:
            all_imports.add(imp)

        var_name = f"canvas_{node['id']}"
        cfg = node.get("config", {})

        # Check config
        err = tmpl.check_config(cfg)
        if err:
            raise CanvasError(f"组件 '{node.get('label', node['id'])}' 配置错误: {err}")

        # Build code blocks
        try:
            # If the first component is NOT an embedding, add an input projection
            # to map input_dim → d_model (Encoder/Decoder expect d_model-dim input)
            if idx == 0 and ntype != "embedding":
                all_imports.add("import torch.nn as nn")
                init_lines.insert(0,
                    "# Input projection — map input_dim → d_model\n"
                    f"self.input_proj = nn.Linear({global_params['input_dim']}, {global_params['d_model']})"
                )
                # prev_var stays "x_enc", but we add a projection step before the first component
                fwd_proj = "x = self.input_proj(x_enc)"
                # The first component should now read from "x" (projected), not "x_enc"
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
        prev_var = "x"  # forward always stores result in x

    imports_str = "\n".join(sorted(all_imports))

    # Indent init body by 8 spaces (under __init__), preserving internal relative indent
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


def write_model_file(source, project_id, version):
    """Write generated model source to disk and return the file path."""
    os.makedirs(GENERATED_DIR, exist_ok=True)
    filename = f"canvas_{project_id}_v{version}.py"
    filepath = os.path.join(GENERATED_DIR, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(source)
    return filepath


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
        # canvas_<project_id>_v<N>.py
        stem = fname[:-3]  # remove .py
        filepath = os.path.join(GENERATED_DIR, fname)

        try:
            module = importlib.import_module(f"utils.models.generated.{stem}")
            # Need to reload since importlib caches
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

        # Parse project_id and version from filename
        # canvas_<project_id>_v<N>.py
        inner = stem[len("canvas_"):]  # <project_id>_v<N>
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
    # Only allow deleting canvas-generated models
    if not model_type.startswith("canvas_"):
        return False

    # Remove from registry
    MODEL_REGISTRY.pop(model_type, None)

    # Delete the .py file
    filepath = os.path.join(GENERATED_DIR, f"{model_type}.py")
    if os.path.isfile(filepath):
        os.remove(filepath)

    # Also remove compiled cache
    pycache = os.path.join(GENERATED_DIR, "__pycache__")
    cache_file = os.path.join(pycache, f"{model_type}.cpython-*.pyc")
    # Simple glob-free approach: remove any matching .pyc
    if os.path.isdir(pycache):
        for fname in os.listdir(pycache):
            if fname.startswith(model_type) and fname.endswith((".pyc", ".pyo")):
                os.remove(os.path.join(pycache, fname))

    return True
