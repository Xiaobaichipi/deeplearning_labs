# 组件注册表去重 — `COMPONENT_REGISTRY` 单一数据源

## 背景

`canvas_registry.js` 和 `canvas_generator.py` 各自维护了一份组件默认参数，容易不同步。

## 改动

- **`utils/canvas_generator.py`**: `COMPONENT_DEFAULTS` 替换为 `COMPONENT_REGISTRY`（含 label/color/category/ports 等全部元信息）。`COMPONENT_DEFAULTS` 保留为从 registry 自动生成的别名
- **`routes/projects.py`**: 新增 `GET /api/canvas/component-defaults` 端点，返回 `COMPONENT_REGISTRY` 作为唯一数据源
- **`static/js/api.js`**: 新增 `_loadComponentRegistry()` 函数
- **`static/js/canvas.js`**: 移除对 `canvas_registry.js` 的依赖。内联 fallback registry，启动时通过 `_loadComponentRegistry()` 异步升级
- **`static/js/canvas_registry.js`**: **已删除**
- **`templates/index.html`**: 移除 `<script src="canvas_registry.js">` 引用

## 新增参数同步方式

```
画布组件定义: COMPONENT_REGISTRY (Python)
                    ↓
GET /api/canvas/component-defaults
                    ↓
          COMPONENT_TYPES (JavaScript)
```

只需维护 Python 端一份，前端启动时自动同步。

## 验证

232 测试全部通过，无回归。
