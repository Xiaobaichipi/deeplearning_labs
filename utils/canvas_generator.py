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

    def build_init(self, comp_id, var_name, cfg, global_params):
        """Return Python code for the __init__ block of one component."""
        # Merge component defaults with global pipeline params
        ctx = dict(cfg)
        ctx["id"] = comp_id
        ctx["var"] = var_name
        ctx.update(global_params)
        return self.init_tpl.format(**ctx)

    def build_forward(self, var_name, in_var):
        """Return Python code for the forward call of one component."""
        return self.forward_tpl.format(var=var_name, in_var=in_var)


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
            "x, _ = self.{var}(x)"
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
            "x = self.{var}(x, cross=x)"
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
            "x = self.{var}(x)"
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
    _assert(in_degree[ordered[0]] == 0,
            "画布第一个节点必须有一个输入端口（无输入连接）")

    # ── 3. Validate d_model consistency ──────────────────────────
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
        if d_model is None and cfg.get("d_model"):
            d_model = cfg["d_model"]
        if cfg.get("dropout") is not None:
            dropout = cfg["dropout"]

    # Default d_model if no component defines one
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
            init_code = tmpl.build_init(node["id"], var_name, cfg, global_params)
            fwd_code = tmpl.build_forward(var_name, prev_var)
        except KeyError as e:
            raise CanvasError(
                f"组件 '{node.get('label', node['id'])}' 缺少参数: {e}")

        init_lines.append(init_code)
        fwd_lines.append(fwd_code)
        prev_var = var_name

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


def register_model(filepath, model_type, project_id="", version=1):
    """Import a generated model file and register it in MODEL_REGISTRY.

    Returns the model class.
    """
    module_name = f"utils.models.generated.{model_type}"
    spec = importlib.util.spec_from_file_location(module_name, filepath)
    if spec is None or spec.loader is None:
        raise CanvasError(f"无法加载生成的模型文件: {filepath}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Find the model class (should be the only BaseModel subclass)
    model_class = None
    for attr_name in dir(module):
        attr = getattr(module, attr_name)
        if isinstance(attr, type) and issubclass(attr, object):
            from utils.models.base import BaseModel
            if issubclass(attr, BaseModel) and attr is not BaseModel:
                model_class = attr
                break

    if model_class is None:
        raise CanvasError("生成的模型文件中未找到有效的模型类")

    pid_short = project_id[:8] if project_id else "unknown"
    MODEL_REGISTRY[model_type] = {
        "class": model_class,
        "name": f"Canvas ({pid_short}... v{version})",
        "pipeline": "large",
        "params": {},
    }

    return model_class
