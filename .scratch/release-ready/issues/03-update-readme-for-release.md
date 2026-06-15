---
title: 更新 README 适配公开发布
status: ready-for-agent
---

## What to build

README 中有几处信息与当前 v2-project-system 分支不匹配，需要更新：

1. **分支名** — 项目状态引用 `jiagou_youhua` 分支，应改为 `v2-project-system` 或 `master`
2. **GPU 安装指南** — 目前指向 CUDA 11.8（`cu118`），应更新到 CUDA 12.1（`cu121`）以匹配当前推荐的 PyTorch 版本
3. **架构图** — 缺少 `routes/projects.py` 和 `utils/session.py`（对比代码评审两个文件的 SessionManager 职责）
4. **项目状态** — 更新为更正式的描述而不是"Active development"
5. **开发指引** — 测试章节已有测试，去掉"(once added)"备注

## Acceptance criteria

- [ ] 分支名更新为 `master`（发布分支）
- [ ] GPU 安装命令指向 `cu121`
- [ ] 架构图补充 `routes/projects.py` 和 `utils/session.py`
- [ ] 项目状态更新
- [ ] 测试章节不再说"(once added)"

## Blocked by

None - can start immediately

## 涉及文件

- `README.md`
