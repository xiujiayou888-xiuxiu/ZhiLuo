# -*- coding: utf-8 -*-
"""
知络 v8.0 工具集 — 移植自 v6.0 的 4 个实用能力
===============================================
to_mermaid   : 知识图谱 → Mermaid 可视化代码
export_as    : 多格式导出 (json / csv / markdown / graphml)
pagerank     : PageRank 知识枢纽排序（纯 Python）
backup/restore : 备份与恢复
"""
import json
import csv
import os
import shutil
import math
from datetime import datetime
from pathlib import Path
from collections import defaultdict


# ══════════════════════════════════════════════════════════════
#  1. Mermaid 可视化
# ══════════════════════════════════════════════════════════════

def to_mermaid(brain, focus=None, max_nodes=50):
    """
    将知识图谱转为 Mermaid 图代码

    Args:
        brain: BrainEngine 实例
        focus: 指定中心实体名，只展示其 2 跳内的子图
        max_nodes: 不指定 focus 时，全图最大展示节点数

    Returns:
        Mermaid 图代码字符串，可直接复制到 mermaid.live 查看
    """
    G = brain.G.G
    if G.number_of_nodes() == 0:
        return "graph TD\n    empty[\"(空图谱)\"]"

    if focus:
        # 找到 focus 节点，取 2 跳子图
        matches = brain.find_entity(focus)
        if not matches:
            return f"graph TD\n    empty[\"(未找到实体: {focus})\"]"

        start_nid = matches[0][0]
        visited = set()
        queue = [(start_nid, 0)]

        while queue:
            node, hops = queue.pop(0)
            if node in visited or hops > 2:
                continue
            visited.add(node)
            for neighbor in G.successors(node):
                if neighbor not in visited:
                    queue.append((neighbor, hops + 1))
            for neighbor in G.predecessors(node):
                if neighbor not in visited:
                    queue.append((neighbor, hops + 1))

        node_ids = visited
    else:
        # 取前 max_nodes 个节点（按度数排序）
        nodes_sorted = sorted(
            G.nodes(data=True),
            key=lambda x: G.degree(x[0]),
            reverse=True
        )
        node_ids = {nid for nid, _ in nodes_sorted[:max_nodes]}

    if not node_ids:
        return "graph TD\n    empty[\"(无节点)\"]"

    lines = ["graph TD"]

    # 节点定义
    for nid in node_ids:
        data = G.nodes[nid]
        name = data.get("name", str(nid))
        etype = data.get("type", "entity")
        # 截断过长的名称
        label = name[:40].replace('"', "'")
        if len(name) > 40:
            label += "..."
        lines.append(f'    N{nid}["[{etype}] {label}"]')

    # 边定义
    edge_styles = {
        "cause":       "-->",
        "affect":      "-->",
        "part_of":     "-.->",
        "synonym":     "==>",
        "substitute":  "-->",
        "depend_on":   "-.->",
        "belongs_to":  "---",
        "related":     "---",
        "refines":     "-..->",
        "contradicts": "--x",
    }

    seen_edges = set()
    for s, t, data in G.edges(data=True):
        if s not in node_ids or t not in node_ids:
            continue
        # 去重（双向边只画一条）
        pair = (min(s, t), max(s, t))
        if pair in seen_edges:
            continue
        seen_edges.add(pair)

        rel = data.get("relation", "related")
        weight = data.get("weight", 1.0)
        style = edge_styles.get(rel, "---")
        # 边标签：关系类型 + 权重
        if weight != 1.0:
            label = f"{rel} ({weight:.1f})"
        else:
            label = rel
        lines.append(f"    N{s} {style}|{label}| N{t}")

    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════
#  2. 多格式导出
# ══════════════════════════════════════════════════════════════

def export_as(brain, fmt="json", output_path=None):
    """
    导出知识图谱

    Args:
        brain: BrainEngine 实例
        fmt: 导出格式 — json / csv / markdown / graphml
        output_path: 可选，指定输出路径；不指定则自动生成到 data 目录

    Returns:
        导出文件的绝对路径
    """
    G = brain.G.G
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    if output_path:
        out_path = Path(output_path)
    else:
        out_dir = brain._data_dir
        out_path = out_dir / f"export_{brain.workspace}_{ts}.{fmt}"

    out_path.parent.mkdir(parents=True, exist_ok=True)

    if fmt == "json":
        _export_json(brain, out_path)
        result_path = str(out_path.resolve())
    elif fmt == "csv":
        nodes_path, edges_path = _export_csv(brain, out_path)
        result_path = str(nodes_path)  # 返回 nodes.csv 路径
    elif fmt in ("markdown", "md"):
        _export_markdown(brain, out_path)
        result_path = str(out_path.resolve())
    elif fmt == "graphml":
        _export_graphml(brain, out_path)
        result_path = str(out_path.resolve())
    else:
        raise ValueError(f"不支持的导出格式: {fmt}，支持 json / csv / markdown / graphml")

    return result_path


def _export_json(brain, path):
    """导出完整 JSON"""
    data = {
        "workspace": brain.workspace,
        "version": "8.0",
        "exported_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "graph": brain.G.to_dict(),
        "entity_meta": {
            str(k): v.to_dict() for k, v in brain._entity_meta.items()
        },
        "edge_meta": {
            str(k): v.to_dict() for k, v in brain._edge_meta.items()
        },
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _export_csv(brain, path):
    """导出 nodes.csv 和 edges.csv"""
    G = brain.G.G

    # nodes.csv
    nodes_path = path.with_suffix("").parent / (path.stem + "_nodes.csv")
    with open(nodes_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["node_id", "name", "type", "degree"])
        for nid, data in G.nodes(data=True):
            writer.writerow([
                nid,
                data.get("name", ""),
                data.get("type", "entity"),
                G.degree(nid),
            ])

    # edges.csv
    edges_path = path.with_suffix("").parent / (path.stem + "_edges.csv")
    with open(edges_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["source_id", "source_name", "target_id", "target_name",
                         "relation", "weight"])
        for s, t, data in G.edges(data=True):
            writer.writerow([
                s,
                G.nodes[s].get("name", ""),
                t,
                G.nodes[t].get("name", ""),
                data.get("relation", "related"),
                data.get("weight", 1.0),
            ])

    # 返回实际写入路径
    return str(nodes_path), str(edges_path)


def _export_markdown(brain, path):
    """导出 Markdown，按实体类型分组"""
    G = brain.G.G

    # 按类型分组
    type_groups = defaultdict(list)
    for nid, data in G.nodes(data=True):
        etype = data.get("type", "entity")
        name = data.get("name", str(nid))
        degree = G.degree(nid)
        type_groups[etype].append((nid, name, degree))

    lines = []
    lines.append(f"# 知识图谱导出 — {brain.workspace}")
    lines.append(f"\n导出时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"总计: {G.number_of_nodes()} 个实体, {G.number_of_edges()} 条关系\n")

    # 目录
    lines.append("## 目录\n")
    for etype in sorted(type_groups.keys()):
        lines.append(f"- [{etype}](#{etype}) ({len(type_groups[etype])}个)")

    lines.append("\n---\n")

    # 详细内容
    for etype in sorted(type_groups.keys()):
        items = sorted(type_groups[etype], key=lambda x: -x[2])
        lines.append(f"## {etype} ({len(items)}个)\n")

        for nid, name, degree in items:
            # 关联关系
            relations = []
            for neighbor in G.successors(nid):
                edge = G[nid][neighbor]
                n_name = G.nodes[neighbor].get("name", str(neighbor))
                rel = edge.get("relation", "related")
                relations.append(f"→ {n_name} ({rel})")
            for neighbor in G.predecessors(nid):
                edge = G[neighbor][nid]
                n_name = G.nodes[neighbor].get("name", str(neighbor))
                rel = edge.get("relation", "related")
                relations.append(f"← {n_name} ({rel})")

            rel_str = "  \\|  ".join(relations[:10]) if relations else "无关联"
            if len(relations) > 10:
                rel_str += f" ...(+{len(relations) - 10})"

            lines.append(f"- **{name}** (度:{degree})")
            lines.append(f"  - {rel_str}")

        lines.append("")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def _export_graphml(brain, path):
    """导出 GraphML 标准 XML 格式，可导入 Gephi / Neo4j"""
    G = brain.G.G

    with open(path, "w", encoding="utf-8") as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        f.write('<graphml xmlns="http://graphml.graphdrawing.org/xmlns"\n')
        f.write('         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"\n')
        f.write('         xsi:schemaLocation="http://graphml.graphdrawing.org/xmlns\n')
        f.write('         http://graphml.graphdrawing.org/xmlns/1.0/graphml.xsd">\n')

        # Key 定义
        f.write('  <key id="name" for="node" attr.name="name" attr.type="string"/>\n')
        f.write('  <key id="type" for="node" attr.name="type" attr.type="string"/>\n')
        f.write('  <key id="relation" for="edge" attr.name="relation" attr.type="string"/>\n')
        f.write('  <key id="weight" for="edge" attr.name="weight" attr.type="double"/>\n')

        f.write(f'  <graph id="G" edgedefault="directed">\n')

        # 节点
        for nid, data in G.nodes(data=True):
            name = data.get("name", str(nid))
            etype = data.get("type", "entity")
            # XML 转义
            name_esc = name.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")
            f.write(f'    <node id="n{nid}">\n')
            f.write(f'      <data key="name">{name_esc}</data>\n')
            f.write(f'      <data key="type">{etype}</data>\n')
            f.write(f'    </node>\n')

        # 边
        for s, t, data in G.edges(data=True):
            rel = data.get("relation", "related")
            weight = data.get("weight", 1.0)
            f.write(f'    <edge source="n{s}" target="n{t}">\n')
            f.write(f'      <data key="relation">{rel}</data>\n')
            f.write(f'      <data key="weight">{weight}</data>\n')
            f.write(f'    </edge>\n')

        f.write('  </graph>\n')
        f.write('</graphml>\n')


# ══════════════════════════════════════════════════════════════
#  3. PageRank 排序
# ══════════════════════════════════════════════════════════════

def pagerank(brain, top_k=20, damping=0.85, max_iter=100):
    """
    计算知识枢纽排序（纯 Python 实现，不依赖 numpy）

    基于 NetworkX 有向图，入边越多 + 入边来源节点越重要 → score 越高。

    Args:
        brain: BrainEngine 实例
        top_k: 返回前 K 个最高分节点
        damping: 阻尼系数，默认 0.85
        max_iter: 最大迭代次数

    Returns:
        [(node_name, score, degree), ...] 按 score 降序
    """
    G = brain.G.G
    n = G.number_of_nodes()
    if n == 0:
        return []
    if n == 1:
        nid = list(G.nodes())[0]
        name = G.nodes[nid].get("name", str(nid))
        return [(name, 1.0, 0)]

    # 构建入边邻接表（只存 node_id）
    nids = list(G.nodes())
    in_edges = {nid: [] for nid in nids}
    out_degree = {nid: max(G.out_degree(nid), 1) for nid in nids}

    for s, t in G.edges():
        in_edges[t].append(s)

    # 初始化 PageRank
    pr = {nid: 1.0 / n for nid in nids}

    for _ in range(max_iter):
        new_pr = {}
        for nid in nids:
            rank_sum = sum(pr[src] / out_degree[src] for src in in_edges[nid])
            new_pr[nid] = (1 - damping) / n + damping * rank_sum
        pr = new_pr

    # 排序输出
    results = []
    for nid in nids:
        name = G.nodes[nid].get("name", str(nid))
        degree = G.degree(nid)
        results.append((name, round(pr[nid], 6), degree))

    results.sort(key=lambda x: -x[1])
    return results[:top_k]


# ══════════════════════════════════════════════════════════════
#  4. 备份与恢复
# ══════════════════════════════════════════════════════════════

def backup(brain, output_dir=None):
    """
    备份当前知识图谱到文件

    将 brain 的 JSON 持久化文件复制到备份目录，文件名带时间戳。
    备份目录默认为 brain._data_dir 下的 backups/。

    Args:
        brain: BrainEngine 实例
        output_dir: 可选，指定备份目录

    Returns:
        备份文件路径
    """
    # 先确保当前数据已保存
    brain.save()

    source = brain._graph_file
    if not source.exists():
        raise FileNotFoundError(f"图谱文件不存在: {source}")

    # 备份目录
    if output_dir:
        backup_dir = Path(output_dir)
    else:
        backup_dir = brain._data_dir / "backups"
    backup_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    dest = backup_dir / f"brain_{brain.workspace}_{ts}.json"
    shutil.copy2(str(source), str(dest))

    # 同时备份纠错文件（如果存在）
    if brain._correction_file.exists():
        corr_dest = backup_dir / f"corrections_{brain.workspace}_{ts}.json"
        shutil.copy2(str(brain._correction_file), str(corr_dest))

    return str(dest.resolve())


def restore(brain, backup_path):
    """
    从备份文件恢复知识图谱

    恢复前自动备份当前数据为安全措施（.pre_restore 后缀）。
    恢复后重建索引。

    Args:
        brain: BrainEngine 实例
        backup_path: 备份文件路径

    Returns:
        {"status": "ok", "nodes": int, "edges": int, "safety_backup": str}
    """
    backup_file = Path(backup_path)
    if not backup_file.exists():
        return {"status": "error", "message": f"备份文件不存在: {backup_path}"}

    # ── 安全措施：先备份当前数据 ──
    safety_path = None
    if brain._graph_file.exists():
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        safety_dir = brain._data_dir / "backups"
        safety_dir.mkdir(parents=True, exist_ok=True)
        safety_path = safety_dir / f"brain_{brain.workspace}_pre_restore_{ts}.json"
        shutil.copy2(str(brain._graph_file), str(safety_path))

    # ── 加载备份数据 ──
    with open(backup_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 恢复图数据
    graph_data = data.get("graph", {})
    brain.G.from_dict(graph_data)

    # 恢复元数据（使用 knowledge_schema 中的 EntityMeta / EdgeMeta）
    from knowledge_schema import EntityMeta, EdgeMeta
    brain._entity_meta = {
        int(k): EntityMeta.from_dict(v)
        for k, v in data.get("entity_meta", {}).items()
    }

    brain._edge_meta = {
        _parse_edge_key(k): EdgeMeta.from_dict(v)
        for k, v in data.get("edge_meta", {}).items()
    }

    # 恢复纠错规则
    brain._corrections = data.get("corrections", [])
    brain._save_corrections()

    # 重建索引
    brain._rebuild_index()

    # 保存
    brain.save()

    return {
        "status": "ok",
        "nodes": brain.G.G.number_of_nodes(),
        "edges": brain.G.G.number_of_edges(),
        "safety_backup": str(safety_path.resolve()) if safety_path else None,
    }


# ── 辅助函数 ──

def _parse_edge_key(key_str):
    """解析边元数据键: '(sid, tid, rtype)' → tuple"""
    if isinstance(key_str, tuple):
        return key_str
    if isinstance(key_str, str):
        if key_str.startswith("("):
            try:
                return eval(key_str)
            except Exception:
                pass
    return key_str


def _entity_meta_from_dict(d):
    """回退：简单的 dict→EntityMeta 转换（不依赖 knowledge_schema）"""
    return type("EntityMeta", (), {
        "name": d.get("name", ""),
        "entity_type": d.get("entity_type", "entity"),
        "confidence": d.get("confidence", 1.0),
        "source": d.get("source", "manual"),
        "created_at": d.get("created_at", ""),
        "tags": d.get("tags", []),
        "meta": d.get("meta", {}),
        "to_dict": lambda self=None: d,
    })()


def _edge_meta_from_dict(d):
    """回退：简单的 dict→EdgeMeta 转换"""
    return type("EdgeMeta", (), {
        "relation": d.get("relation", "related"),
        "weight": d.get("weight", 1.0),
        "confidence": d.get("confidence", 0.5),
        "trigger": d.get("trigger", ""),
        "change_pct": d.get("change_pct", 0.0),
        "created_at": d.get("created_at", ""),
        "source": d.get("source", "auto"),
        "meta": d.get("meta", {}),
        "to_dict": lambda self=None: d,
    })()
