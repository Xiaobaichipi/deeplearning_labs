---
title: Docker 部署支持
status: ready-for-agent
---

## What to build

提供 Dockerfile + docker-compose.yml，用户可以通过一条命令启动整个应用。这降低了环境配置门槛（无需手动安装 PyTorch、处理 CUDA 驱动兼容性等）。

## Acceptance criteria

- [ ] `Dockerfile` — 基于 Python 3.12 slim 镜像，安装依赖，暴露 5000 端口
- [ ] `docker-compose.yml` — 映射端口和卷（`uploads/`、`outputs/`、`projects/` 持久化），支持 `docker compose up -d` 一键启动
- [ ] Dockerfile 中包含 CPU 版本的 PyTorch 安装（最广兼容），用户可以自行替换为 GPU 版本
- [ ] `nvidia-docker` 可选支持（GPU 加速）在 docker-compose 中注释提供
- [ ] 验证：`docker compose up -d` 后访问 `http://localhost:5000` 正常工作

## Blocked by

None - can start immediately

## 涉及文件

- `Dockerfile` — 新建
- `.dockerignore` — 新建
- `docker-compose.yml` — 新建
