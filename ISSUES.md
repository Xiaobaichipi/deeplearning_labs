# Issues Log

## 2026-06-05: 新增归一化功能后上传数据不显示

### Bug描述

在 Model Config 步骤新增归一化（Min-Max / Mean）选项后，用户上传数据集时后端返回成功，但前端页面内容不加载（Step 2 不显示数据预览、统计等）。

### 根因分析

`static/js/app.js` 中 `startTraining()` 函数的 `params` 对象定义缺少闭合符号 `};`。

具体过程：在添加 `normalization` 字段到 `params` 对象时，Edit 替换操作意外删除了对象末尾的闭合 `    };`。导致 `startTraining()` 函数体内的对象字面量变为语法错误：

```javascript
const params = {
    ...
    normalization: document.getElementById("normalization").value,
                                                    // ← 缺少 }; 
try {                                               // ← 解析器在此处报错
```

由于整个 `<script>app.js` 在解析阶段就失败，**所有函数**（`handleUpload`、`populateStep2`、`goToStep` 等）均不会被定义。因此：

- `/api/upload` 后端接口正常（curl 测试通过）
- 但上传成功后前端无反应，因为 `handleUpload` 从未执行
- 任何交互函数（`runClean`、`runFill`、`startTraining`）也都不可用

### 修复

在 `app.js:308` 补上缺失的 `    };`，闭合 params 对象：

```javascript
        normalization: document.getElementById("normalization").value,
    };                         // ← 补上这一行

    try {
```

### 教训

编辑 JS 代码后，应在浏览器控制台或通过 Node.js 检查语法：

```bash
node -e "new Function(require('fs').readFileSync('static/js/app.js','utf8')); console.log('OK')"
```

Flask debug reloader 不检测前端文件变化，浏览器也可能缓存旧版 JS。修改后应 Ctrl+F5 强制刷新。

---

## Prior Issues (前序会话已解决)

- NaN JSON 序列化：`clean_nan()` 递归转换
- CSV 中文编码：增加 gbk/gb2312/gb18030 编码回退链
- ReduceLROnPlateau 参数：移除 `verbose`
- 混淆矩阵 thresh：`max(cm)` → `max(max(row) for row in cm)`
- 空 DataFrame 清理保护：至少保留 1 列，IQR 跳过小数据集
- Toggle CSS 重叠：`display: inline-block` + `box-sizing: content-box`
- 中文字体自动检测：`utils/fonts.py`
- 会话/缓存失联：固定 `secret_key` + `get_data()` 磁盘回退
