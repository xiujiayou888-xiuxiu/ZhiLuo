# -*- coding: utf-8 -*-
"""
知络图推理包装器 v7.1 — 组合包装器 + 降噪管道
拦截 ZhiLuo.learn 自动建边（使用降噪后的关系）+ 新增 reason/graph_visualize 命令
"""
import os, json

from graph_engine import GraphEngine
from relation_extractor import extract_filtered, extract_relations, extract_entities
from engine import ZhiLuo, WS_DIR

_SKILL_DIR = os.path.dirname(os.path.abspath(__file__))
_GRAPH_FILE = os.path.join(_SKILL_DIR, "data", "workspaces", "graph.json")


class ZhiLuoGraph:
    def __init__(self, inner, llm_func=None, workspace="global"):
        if llm_func is None and hasattr(inner, "_llm_func") and inner._llm_func is not None:
            llm_func = inner._llm_func
        self._llm = llm_func
        self._workspace = workspace
        self._graph_file = os.path.join(_SKILL_DIR, "data", "workspaces", f"graph_{workspace}.json")
        self._inner = inner
        self.graph = GraphEngine()
        self._load_graph()

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def workbuddy_main(self, cmd, args=""):
        if cmd == "learn":
            result = self._inner.workbuddy_main(cmd, args)
            self._auto_build_edges_filtered(args)
            return result
        if cmd == "reason":
            return self._do_reason(args)
        if cmd == "graph_stats":
            return self._do_graph_stats()
        if cmd == "graph_visualize":
            return self._do_graph_visualize(args)
        return self._inner.workbuddy_main(cmd, args)

    # ── 降噪自动建边 ──
    def _auto_build_edges_filtered(self, text, source_type="用户"):
        """LLM优先提取，规则保底"""
        if self._llm is not None:
            from relation_extractor import extract_with_llm
            llm_rels = extract_with_llm(text, self._llm)
            if llm_rels and len(llm_rels) > 0:
                for r in llm_rels:
                    self.graph.add_relation(r["source"], r["target"], rtype=r.get("relation","related"), weight=r.get("confidence", r.get("weight", 0.5)), source_confidence=r.get("confidence",0.5), target_confidence=r.get("confidence",0.5))
                self._save_graph()
                return len(llm_rels), {}
        result = extract_filtered(text, source_type=source_type)
        relations = result["relations"]
        added = 0
        for r in relations:
            self.graph.add_relation(
                r["source"], r["target"],
                rtype=r["relation"],
                weight=r.get("confidence", r.get("weight", 0.5))
            )
            added += 1
        if added > 0:
            self._save_graph()
        return added, result.get("dropped", {})

    # ── 批量采集建边（自动采集专用，批量降噪） ──
    def batch_learn(self, texts, source_type="采集"):
        total_added = 0
        total_raw = 0
        total_dropped = {"low_confidence": 0, "low_freq": 0}
        for text in texts:
            result = extract_filtered(text, source_type=source_type)
            total_raw += result["raw_count"]
            for k, v in result.get("dropped", {}).items():
                total_dropped[k] = total_dropped.get(k, 0) + v
            for r in result["relations"]:
                self.graph.add_relation(
                    r["source"], r["target"],
                    rtype=r["relation"],
                    weight=r.get("confidence", r.get("weight", 0.5)),
                    source_confidence=r.get("confidence", 0.5),
                    target_confidence=r.get("confidence", 0.5),
                )
                total_added += 1
        if total_added > 0:
            self._save_graph()
        return {
            "added": total_added,
            "raw": total_raw,
            "dropped": total_dropped,
        }

    def _do_reason(self, query):
        result = self.graph.reason(query)
        if "error" in result:
            return f"[Z] {result['error']}"
        if "impacts" in result:
            lines = [f"[Z] 图推理：{result['source']}变化 {result['initial_change']*100:+.1f}%，影响 {result['affected_count']} 个实体"]
            for imp in result["impacts"][:10]:
                pct = imp["change"] * 100
                path_str = " -> ".join(imp["path"])
                lines.append(f"  {imp['name']}: {pct:+.1f}% (hops={imp['hops']}, {path_str})")
            return "\n".join(lines)
        if "paths" in result:
            paths = result["paths"]
            if not paths:
                return f"[Z] {result.get('source','')} 与 {result.get('target','')} 之间无路径"
            lines = [f"[Z] {result['source']} -> {result['target']}：{len(paths)} 条路径"]
            for i, path in enumerate(paths[:5], 1):
                parts = []
                for item in path:
                    if "name" in item: parts.append(item["name"])
                    elif "rel" in item: parts.append(f"-{item['rel']}->")
                lines.append(f"  路径{i}: {' '.join(parts)}")
            return "\n".join(lines)
        if "diffuse" in result:
            diff = result["diffuse"]
            if not diff: return f"[Z] 未找到 {result.get('entity','')} 的关联"
            lines = [f"[Z] {result['entity']} 的关联扩散："]
            for d in diff[:10]:
                lines.append(f"  - {d['name']} [{d['type']}] (hops={d['hops']})")
            return "\n".join(lines)
        return json.dumps(result, ensure_ascii=False, indent=2)

    def _do_graph_stats(self):
        st = self.graph.stats()
        return f"[Z] 图统计：{st['nodes']} 个实体节点，{st['edges']} 条关系边"

    def _do_graph_visualize(self, keyword=""):
        if keyword:
            nodes = self.graph.diffuse(keyword, max_hops=2)
            if not nodes: return f"[Z] 未找到 {keyword} 的图"
            node_ids = {n["id"] for n in nodes}
        else:
            node_ids = None
        lines = ["graph LR"]
        for s, t, data in self.graph.G.edges(data=True):
            if node_ids and (s not in node_ids or t not in node_ids): continue
            s_name = self.graph.G.nodes[s].get("name", str(s))
            t_name = self.graph.G.nodes[t].get("name", str(t))
            lines.append(f'  {s_name} -->|{data.get("relation","related")} {data.get("weight",1.0):.1f}| {t_name}')
        if len(lines) == 1: return "[Z] 图为空"
        return "\n".join(lines)

    def _save_graph(self):
        os.makedirs(os.path.dirname(self._graph_file) if hasattr(self, '_graph_file') else os.path.dirname(_GRAPH_FILE), exist_ok=True)
        data = self.graph.to_dict()
        with open(self._graph_file if hasattr(self, "_graph_file") else _GRAPH_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def switch_ws(self, workspace):
        """切换工作区（每个工作区独立 .db 文件 + 独立图数据）"""
        old_ws = getattr(self, "_workspace", "global")
        if workspace == old_ws:
            return f"[Z] 已在 {workspace} 工作区"
        # 保存当前图
        self._save_graph()
        # 重新初始化（新工作区路径）
        self._inner = ZhiLuo(path=str(WS_DIR / f"{workspace}.db"), workspace=workspace)
        self._workspace = workspace
        self._graph_file = os.path.join(_SKILL_DIR, "data", "workspaces", f"graph_{workspace}.json")
        # 切换图数据
        self._GRAPH_FILE = os.path.join(_SKILL_DIR, "data", "workspaces", f"graph_{workspace}.json")
        self.graph = GraphEngine()
        self._load_graph()
        return f"[Z] 已切换到 {workspace} 工作区"

    def _load_graph(self):
        _gf = self._graph_file if hasattr(self, "_graph_file") else _GRAPH_FILE
        if os.path.exists(_gf):
            try:
                with open(_gf, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.graph.from_dict(data)
            except Exception:
                pass

__all__ = ["ZhiLuoGraph"]
