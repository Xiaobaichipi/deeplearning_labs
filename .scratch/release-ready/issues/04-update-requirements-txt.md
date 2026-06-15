---
title: 更新 requirements.txt 适配公开发布
status: ready-for-agent
---

## What to build

当前 `requirements.txt` 使用宽版本范围（`torch>=2.0`），首次使用的用户可能装到不兼容的版本。需要：

1. 锁定经过测试的版本范围，避免因新版本 API 变更导致不可用
2. 添加 CUDA 版本安装指引注释（指引用户按自己的驱动选择合适的 PyTorch CUDA 版本）
3. 考虑移除 `torchvision` 依赖（项目本身不用 torchvision，但 torch 依赖它）
4. 添加 `openpyxl` 依赖（用于 XLSX 导出）

## Acceptance criteria

- [ ] 版本范围锁定已验证兼容的范围
- [ ] 添加注释说明不同 CUDA 版本的安装方式
- [ ] `openpyxl` 加入依赖（目前是运行时 import 时会优雅降级，但用户期望 XLSX 正常工作）
- [ ] 用 `pip install -r requirements.txt` 在新 venv 中测试安装成功
- [ ] 152 测试全部通过

## Blocked by

None - can start immediately

## 涉及文件

- `requirements.txt`
