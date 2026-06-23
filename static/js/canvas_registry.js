/* Canvas Component Registry — defines draggable building blocks.
 *
 * ═══════════════════════════════════════════════════════════════════
 *   ⚠️  修改默认参数时请务必同步更新:
 *       utils/canvas_generator.py 中的 COMPONENT_DEFAULTS 字典
 *   ═══════════════════════════════════════════════════════════════════
 *   两端不一致会导致前后端默认行为不同步（例如前端显示 n_heads=8
 *   但后端代码生成时使用错误的默认值）。
 *   未来可考虑通过 API /api/canvas/component-defaults 提供单一数据源。
 */

const COMPONENT_TYPES = {
  encoder: {
    label: "Encoder",
    category: "基础模块",
    color: "#4CAF50",
    inputs: 1,
    outputs: 1,
    defaults: {
      d_model: 512,
      n_heads: 8,
      d_ff: 2048,
      dropout: 0.1,
      activation: "gelu",
    },
    ports: { input: { label: "输入" }, output: { label: "输出" } },
  },
  decoder: {
    label: "Decoder",
    category: "基础模块",
    color: "#2196F3",
    inputs: 1,
    outputs: 1,
    defaults: {
      d_model: 512,
      n_heads: 8,
      d_ff: 2048,
      dropout: 0.1,
      activation: "gelu",
    },
    ports: { input: { label: "编码器输出" }, output: { label: "输出" } },
  },
  embedding: {
    label: "DataEmbedding",
    category: "输入模块",
    color: "#FF9800",
    inputs: 1,
    outputs: 1,
    defaults: {
      d_model: 512,
      dropout: 0.1,
    },
    ports: { input: { label: "原始数据" }, output: { label: "嵌入向量" } },
  },
  linear: {
    label: "Linear",
    category: "基础模块",
    color: "#9C27B0",
    inputs: 1,
    outputs: 1,
    defaults: {
      in_features: 512,
      out_features: 1,
    },
    ports: { input: { label: "输入" }, output: { label: "输出" } },
  },
};
