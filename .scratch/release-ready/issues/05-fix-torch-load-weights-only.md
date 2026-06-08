---
title: 修复 torch.load weights_only FutureWarning
status: ready-for-agent
---

## What to build

`utils/project_manager.py:210` 调用 `torch.load(state_path, map_location="cpu")` 时没有指定 `weights_only=True`，在当前 PyTorch 版本中产生 FutureWarning，下个主版本会变成错误。

## Acceptance criteria

- [ ] `torch.load(state_path, map_location="cpu", weights_only=True)` 调用正常
- [ ] 模型加载功能（项目激活、加载模型到 session）不受影响
- [ ] 测试 `test_activate_and_train_persists_model` 和 `test_export_model` 仍然通过

## Blocked by

None - can start immediately

## 涉及文件

- `utils/project_manager.py` — line 210
