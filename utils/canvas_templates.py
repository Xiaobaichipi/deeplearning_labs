"""
Canvas component templates — defines how each component type renders its
``__init__`` and ``forward`` code blocks.

Single source of truth for component metadata (``COMPONENT_REGISTRY``)
and code-generation rules (``COMPONENT_TEMPLATES``).
"""


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


# ── Component code-generation templates ───────────────────────────

COMPONENT_TEMPLATES = {
    "embedding": ComponentTemplate(
        imports=[
            "from utils.models.shared_layers.Embed import DataEmbedding",
        ],
        init_tpl=(
            "# DataEmbedding — project raw features into d_model space\n"
            "self.{var} = DataEmbedding(\n"
            "    c_in=input_dim, d_model={d_model},\n"
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
            "self.{var} = nn.Linear({d_model}, output_dim)"
        ),
        forward_tpl=(
            "x = self.{var}({in_var})"
        ),
    ),
}
