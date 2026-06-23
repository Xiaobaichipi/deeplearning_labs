# HTML 模板提取 — `populateModelList` 改用 `<template>` 标签

## 背景

`populateModelList` 函数中约 30 行 HTML 字符串内联在 JS 里，结构难读难改。

## 改动

- **`templates/index.html`**: 新增 `#tpl-model-card` 和 `#tpl-model-empty` 两个 `<template>` 元素
- **`static/js/ui.js`**: `populateModelList` 从模板字符串改为 `tpl.content.cloneNode(true)` + DOM API 填充数据。新增 `setChip()` 辅助函数
- **`docs/work_log/html_template_extract_1.md`**: 本文件

## 验证

232 测试全部通过，无回归。
