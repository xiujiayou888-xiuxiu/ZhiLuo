# -*- coding: utf-8 -*-
"""
知络 v8.3 — 大脑包装层
移植 v8.2 的增量能力到 v7.1 底座：
  - 共现网络 + 语义发散 (semantic_bridge)
  - 三层上下文记忆 (context_memory)
  - 体裁感知检索 (genre_retrieval)
  - 通用提取器 (generic_extractor)
  - 三级索引 (fast_index)
  - 关系注册表 (knowledge_schema)
  - 工具集 (tools)
保持 v7.1 所有 API 不变，只做增量扩展
"""
import os, json, re
from pathlib import Path

# 增量模块
from knowledge_schema import RelationRegistry, RelationType, EntityMeta, EdgeMeta
from generic_extractor import extract_entities, extract_relations, extract_all as generic_extract_all
from context_memory import ContextMemory, create_context_memory
from genre_retrieval import GenreClassifier, GenreAwareRetriever, SkeletonExtractor
from semantic_bridge import CoOccurrenceNetwork, SemanticBridge
from fast_index import FastIndex
from graph_engine import GraphEngine
from tools import to_mermaid, export_as, pagerank, backup, restore


class BrainWrapper:
    """
    知络 v8.3 BrainWrapper
    包装 MemoryStore/ZhiLuo + 全部增量模块
    所有增量能力通过 brain.xxx() 调用，不破坏原 ZhiLuo 接口
    """

    def __init__(self, memory_store, workspace="global", data_dir=None):
        self.ms = memory_store  # 底层 MemoryStore (v7.1 底座)
        self.ws = workspace

        # 数据目录
        if data_dir:
            self._data_dir = Path(data_dir)
        elif hasattr(memory_store, "path") and memory_store.path and os.path.isdir(memory_store.path):
            self._data_dir = Path(memory_store.path) / "brain"
        else:
            self._data_dir = Path(os.path.dirname(os.path.abspath(__file__))) / "brain_data"
        self._data_dir.mkdir(parents=True, exist_ok=True)

        # === v8.3 增量模块初始化 ===

        # 1. 关系注册表
        self.registry = RelationRegistry()

        # 2. 三级索引 (FastIndex) — 基于 SQLite+FTS5
        self.idx = FastIndex(workspace=workspace, data_dir=str(self._data_dir))

        # 3. 共现网络 + 语义发散
        self.co_net = CoOccurrenceNetwork(data_dir=str(self._data_dir))
        self.bridge = SemanticBridge(self.co_net)

        # 4. 三层上下文记忆
        self.context_memory = ContextMemory(data_dir=str(self._data_dir / "context"))

        # 5. 体裁感知检索
        self.genre_retriever = GenreAwareRetriever(self.idx)

        # 6. 实体/关系元数据缓存 (与 v8.2 brain.py 兼容)
        self._entity_meta: dict = {}
        self._edge_meta: dict = {}

        # 7. 图引擎 (tools.py 依赖 brain.G.G)
        self.G = GraphEngine()
        self.workspace = workspace
        self._graph_file = self._data_dir / f"graph_{workspace}.json"
        self._correction_file = self._data_dir / f"corrections_{workspace}.json"
        self._corrections = []

    def sync_index(self):
        """从 MemoryStore 实时同步到 FastIndex"""
        try:
            # 用 MemoryStore 实际的 workspace 而非 brain.ws
            ws = getattr(self.ms, 'ws', self.ws)
            nodes = self.ms.valid(ws)
            if nodes:
                rebuild_nodes = []
                for n in nodes:
                    rebuild_nodes.append({
                        "id": n["id"],
                        "name": n.get("text", "")[:80],
                        "text": n.get("text", ""),
                        "type": n.get("category", "knowledge"),
                    })
                self.idx.rebuild(rebuild_nodes)
                return len(rebuild_nodes)
        except Exception:
            pass
        return 0

    def _rebuild_index_if_needed(self):
        """从 MemoryStore 已有数据重建 FastIndex"""
        try:
            nodes = self.ms.valid(self.ws)
            if nodes:
                rebuild_nodes = []
                for n in nodes:
                    rebuild_nodes.append({
                        "id": n["id"],
                        "name": n.get("text", "")[:80],
                        "text": n.get("text", ""),
                        "type": n.get("category", "knowledge"),
                    })
                self.idx.rebuild(rebuild_nodes)
        except Exception:
            pass

    def save(self):
        """保存图数据到JSON（tools.py backup依赖）"""
        try:
            data = {
                "graph": self.G.to_dict() if hasattr(self.G, 'to_dict') else {},
                "entity_meta": {k: (v.to_dict() if hasattr(v, 'to_dict') else v) for k, v in self._entity_meta.items()},
                "edge_meta": {str(k): (v.to_dict() if hasattr(v, 'to_dict') else v) for k, v in self._edge_meta.items()},
                "corrections": self._corrections,
            }
            with open(str(self._graph_file), "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def find_entity(self, name):
        """查找实体（tools.py to_mermaid依赖）"""
        return self.G.find_entity(name)

    def _rebuild_index(self):
        """重建索引（tools.py restore依赖）"""
        self.sync_index()

    # ========== 语义发散 (v8.2 核心) ==========

    def explore(self, query, top_k=15):
        """语义发散：从给定概念出发发现关联概念"""
        return self.bridge.explore(query, top_k=top_k)

    def explore_multi(self, queries, top_k=20):
        """多查询联合发散"""
        return self.bridge.explore_multi(queries, top_k=top_k)

    def bridge_network(self, seed=None, max_nodes=50):
        """共现网络结构"""
        return self.co_net.get_network(seed=seed, max_nodes=max_nodes)

    def bridge_stats(self):
        """共现网络统计"""
        return self.co_net.stats()

    # ========== 上下文记忆 (v8.1) ==========

    def remember_session(self, keyword, entities=None, snapshot=""):
        """L1 短程：记住会话上下文"""
        self.context_memory.remember_query(keyword, entities or [], snapshot)

    def get_context(self, query):
        """三层上下文：返回格式化的上下文字符串"""
        result = self.context_memory.recall_context(query)
        if not result.get("found"):
            return ""
        return self.context_memory.inherit_context(query)

    def inherit_context(self, query):
        """生成上下文继承提示文本"""
        return self.context_memory.inherit_context(query)

    def get_topic_knowledge(self, query):
        """获取查询相关的主题聚合知识（L2 中程）"""
        return self.context_memory.get_topic_knowledge(query)

    def topic_evolution(self, topic=None):
        """主题演化历史（L3 长程）"""
        return self.context_memory.get_topic_evolution(topic)

    def context_stats(self):
        """上下文记忆统计"""
        return self.context_memory.stat()

    # ========== 体裁感知检索 (v8.1) ==========

    def classify_genre(self, text):
        """分类文本体裁"""
        return GenreClassifier.classify(text)

    def genre_search(self, query, top_k=10):
        """体裁感知搜索"""
        return self.genre_retriever.retrieve(query, top_k=top_k)

    def extract_skeleton(self, text):
        """提取文本骨架（标题/关键点/数字）"""
        genre, _ = GenreClassifier.classify(text)
        return SkeletonExtractor.extract(text, genre)

    # ========== 通用提取器 (v8.0) ==========

    def extract(self, text):
        """通用实体+关系提取（不绑定行业词典）"""
        return generic_extract_all(text)

    # ========== 工具集 (v8.2 tools.py) ==========

    def to_mermaid(self, focus=None, max_nodes=50):
        """知识图谱 → Mermaid 可视化代码"""
        return to_mermaid(self, focus=focus, max_nodes=max_nodes)

    def export(self, fmt, output_path=None):
        """多格式导出 (json/csv/markdown/graphml)"""
        return export_as(self, fmt, output_path)

    def page_rank(self, alpha=0.85, max_iter=20):
        """PageRank 知识枢纽排序"""
        return pagerank(self, top_k=20, damping=alpha, max_iter=max_iter)

    def backup(self, output_dir=None):
        """备份"""
        return backup(self, output_dir or str(self._data_dir / "backups"))

    def restore(self, backup_path):
        """恢复"""
        return restore(self, backup_path)

    # ========== v8.2 brain.py 兼容接口 ==========

    def stats(self):
        """综合统计"""
        result = {
            "workspace": self.ws,
            "memory_nodes": len(self.ms.valid(self.ws)) if hasattr(self.ms, "valid") else 0,
            "co_net": self.bridge_stats(),
            "context": self.context_stats(),
        }
        try:
            result["index"] = self.idx.stats()
        except Exception:
            pass
        return result

    def decay(self, half_life=30, dry_run=True):
        "置信度衰减：长期未访问的知识降低置信度"
        from engine import decay as _decay
        return _decay(self.ms, half_life=half_life, dry_run=dry_run)

    def snapshot(self):
        """当前知识图谱概览"""
        nodes = self.ms.valid(self.ws) if hasattr(self.ms, "valid") else []
        edges = []
        seen = set()
        # Bug2修复：边存在store.adj中，不在node["edges"]里
        adj = getattr(self.ms, 'adj', {})
        for src_id, edge_list in adj.items():
            for e in edge_list:
                k = (src_id, e["target"])
                if k not in seen and (e["target"], src_id) not in seen:
                    seen.add(k)
                    edges.append({"source": src_id, "target": e["target"], "relation": e.get("type", "related")})
        return {
            "workspace": self.ws,
            "nodes_count": len(nodes),
            "edges_count": len(edges),
        }


    def find_semantic_conflicts(self, threshold=0.8):
        """语义相似度冲突检测（首选VectorStore向量检索，降级SimHash海明距离）"""
        from collections import defaultdict
        from engine import SimHash

        nodes = self.ms.valid(self.ws) if hasattr(self.ms, "valid") else []
        if not nodes:
            return []

        # 首选：向量引擎精确语义匹配
        vec_conflicts = []
        try:
            from pathlib import Path
            data_dir = getattr(self, "_data_dir", Path(__file__).parent / "data")
            db_path = str(Path(data_dir) / "workspaces" / "%s_vec.db" % self.ws)
            from vector_store import VectorStore
            vs = VectorStore(db_path)

            by_cat = defaultdict(list)
            for n in nodes:
                by_cat[n.get("category", "未分类")].append(n)

            for cat, group in by_cat.items():
                for i in range(len(group)):
                    for j in range(i + 1, len(group)):
                        a, b = group[i], group[j]
                        ta, tb = a.get("text", ""), b.get("text", "")
                        if len(ta) < 10 or len(tb) < 10:
                            continue
                        try:
                            results = vs.search(ta, top_k=5)
                            for rid, sim, _ in results:
                                if rid == b.get("id"):
                                    if sim > threshold:
                                        vec_conflicts.append({
                                            "text_a": ta[:50],
                                            "text_b": tb[:50],
                                            "similarity": sim,
                                            "category": cat,
                                        })
                                    break
                        except Exception:
                            pass
        except Exception:
            pass

        if vec_conflicts:
            return vec_conflicts[:20]

        # 降级：SimHash海明距离
        by_cat2 = defaultdict(list)
        for n in nodes:
            by_cat2[n.get("category", "未分类")].append(n)

        conflicts = []
        for cat, group in by_cat2.items():
            for i in range(len(group)):
                for j in range(i + 1, len(group)):
                    a, b = group[i], group[j]
                    if len(a.get("text", "")) < 10 or len(b.get("text", "")) < 10:
                        continue
                    sh_a = a.get("simhash", 0)
                    sh_b = b.get("simhash", 0)
                    if sh_a and sh_b:
                        dist = SimHash.hamming(sh_a, sh_b)
                        if dist <= threshold * 3:
                            conflicts.append({
                                "text_a": a["text"][:50],
                                "text_b": b["text"][:50],
                                "similarity": 1.0 - dist / 64,
                                "category": cat,
                            })
        return conflicts[:20]


def create_brain(memory_store, workspace="global", data_dir=None):
    """便捷工厂函数"""
    return BrainWrapper(memory_store, workspace=workspace, data_dir=data_dir)
