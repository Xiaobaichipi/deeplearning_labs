"""
Canvas graph validation — topological sort, single-chain checks,
d_model consistency, and isolated-node detection.
"""


class CanvasError(Exception):
    """Raised for validation / generation failures with a user-facing message."""


def _assert(cond, msg):
    if not cond:
        raise CanvasError(msg)


def validate_canvas(canvas):
    """Validate canvas structure — returns ordered node list for single chain.

    Raises CanvasError with user-facing message on any violation.
    """
    nodes = canvas.get("nodes", [])
    edges = canvas.get("edges", [])

    _assert(len(nodes) >= 2,
            "画布至少需要 2 个组件节点（例如 Embedding → Linear）")

    # ── 1. Build adjacency + in-degree maps ──────────────────────
    in_degree = {n["id"]: 0 for n in nodes}
    out_degree = {n["id"]: 0 for n in nodes}
    adj = {n["id"]: [] for n in nodes}

    for e in edges:
        frm, to = e["from"], e["to"]
        _assert(frm in in_degree,
                f"边的起点 '{frm}' 在画布中不存在")
        _assert(to in in_degree,
                f"边的终点 '{to}' 在画布中不存在")
        in_degree[to] += 1
        out_degree[frm] += 1
        adj[frm].append(to)

    _assert(all(d <= 1 for d in in_degree.values()),
            "当前只支持单链拓扑（每个节点最多一个输入连接）— 分支/汇合留到 P2")
    _assert(all(d <= 1 for d in out_degree.values()),
            "当前只支持单链拓扑（每个节点最多一个输出连接）— 分支留到 P2")

    # Save a copy of initial in_degree before Kahn's algorithm mutates it
    orig_in_degree = dict(in_degree)

    # ── 2. Topological sort (Kahn) ───────────────────────────────
    q = [nid for nid, d in in_degree.items() if d == 0]
    ordered = []
    while q:
        nid = q.pop(0)
        ordered.append(nid)
        for nb in adj[nid]:
            in_degree[nb] -= 1
            if in_degree[nb] == 0:
                q.append(nb)

    _assert(len(ordered) == len(nodes),
            "画布中存在循环连接，请检查连线是否成环")
    _assert(orig_in_degree[ordered[0]] == 0,
            "画布第一个节点必须有一个输入端口（无输入连接）")

    # ── 3. Check for isolated nodes — use pre-Kahn in_degree (orig_in_degree) ──
    for nid in ordered:
        if orig_in_degree.get(nid, 0) == 0 and nid != ordered[0]:
            node = next(n for n in nodes if n["id"] == nid)
            raise CanvasError(
                f"组件 '{node.get('label', nid)}' 没有输入连接（孤立节点）。"
                f"\n请将其连接到上游组件，或删除。")

    # ── 4. Validate d_model consistency ──────────────────────────
    d_model = None
    for nid in ordered:
        node = next(n for n in nodes if n["id"] == nid)
        cfg = node.get("config", {})
        ntype = node["type"]

        if ntype in ("encoder", "decoder", "linear"):
            dm = cfg.get("d_model")
            if d_model is None and dm is not None:
                d_model = dm
            if dm is not None and dm != d_model:
                raise CanvasError(
                    f"组件 '{node.get('label', nid)}' 的 d_model={dm} "
                    f"与前面组件的 d_model={d_model} 不一致。"
                    f"\n请确保所有 Encode/Decoder/Linear 的 d_model 相同。")

    return [next(n for n in nodes if n["id"] == nid) for nid in ordered]
