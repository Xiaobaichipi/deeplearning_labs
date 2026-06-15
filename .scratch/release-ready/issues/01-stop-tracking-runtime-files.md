---
title: 停止跟踪运行时文件（.server.pid / server.log）
status: ready-for-agent
---

## What to build

`.server.pid` 和 `server.log` 是运行时产生的临时文件，虽然已在 `.gitignore` 中，但之前被提交过，现在仍被 git 跟踪。需要将它们从 git 跟踪中移除。

## Acceptance criteria

- [ ] `git rm --cached .server.pid server.log` 执行成功
- [ ] `.gitignore` 中确认这两个文件已被忽略
- [ ] `git status` 不再显示这两个文件的修改
- [ ] 后续重启服务器时，新生成的 `.server.pid` 和 `server.log` 不会被 git 跟踪

## Blocked by

None - can start immediately

## 涉及文件

- `.server.pid` — git rm --cached
- `server.log` — git rm --cached
- `.gitignore` — 确认已有条目
