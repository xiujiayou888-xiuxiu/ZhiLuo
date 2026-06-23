# -*- coding: utf-8 -*-
"""
知络 v8.2 语义桥模块 — SemanticBridge
=====================================
基于真实共现统计 + 图遍历的语义发散引擎。

与左脑 EntanglementBridge 的设计差异：
  左脑：SHA256哈希→随机8维向量→向量内积，语义是随机的
  知络：learn()时记录实体共现→共现网络 + 图边遍历，语义来自真实数据

核心能力：
  1. CoOccurrenceNetwork — 实体共现网络（同一条 learn() 中出现过的实体对）
  2. SemanticBridge — 多路径发散：共现网络 + 知识图谱边 + 关键词扩展
  3. 时间衰减 — 近期的共现权重更高

使用场景：
  - 给定一个概念，发现关联概念（语义发散）
  - 在 search 时扩展查询范围
  - 为 Agent 提供"你可能还想知道"的推荐

不侵权声明：本模块基于通用图论和共现统计方法论独立实现，
与左脑的哈希向量纠缠机制完全不同。
"""

import json
import re
import time
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set
from collections import defaultdict, Counter
from dataclasses import dataclass, field

try:
    import jieba
    _HAS_JIEBA = True
except ImportError:
    _HAS_JIEBA = False


# ═══════════════════════════════════════════════════════════
#  数据结构
# ═══════════════════════════════════════════════════════════

@dataclass
class CoEdge:
    """共现边"""
    count: int = 0           # 共现次数
    first_seen: float = 0.0  # 首次共现时间戳
    last_seen: float = 0.0   # 最近共现时间戳
    sources: Set[str] = field(default_factory=set)  # 来源列表


@dataclass
class BridgeNode:
    """桥节点"""
    term: str
    frequency: int = 0       # 出现总次数
    last_seen: float = 0.0   # 最近出现时间
    sources: Set[str] = field(default_factory=set)
    first_seen: float = 0.0


# ═══════════════════════════════════════════════════════════
#  共现网络
# ═══════════════════════════════════════════════════════════

class CoOccurrenceNetwork:
    """
    实体共现网络
    
    核心逻辑：learn() 时提取文本中的实体词，同一条 learn() 
    中出现的实体对形成共现边。边权重 = 共现次数 × 时间衰减。
    """

    # 中文分词正则（不依赖 jieba，保持零依赖）
    _WORD_RE = re.compile(r'[\u4e00-\u9fff]{2,6}')
    # 停用词
    _STOP_WORDS = {
        '这个', '那个', '一个', '一些', '什么', '怎么', '为什么',
        '可以', '需要', '应该', '能够', '可能', '已经', '还是',
        '不是', '没有', '不会', '不能', '就是', '还是', '或者',
        '以及', '而且', '但是', '因为', '所以', '如果', '虽然',
        '进行', '使用', '通过', '根据', '关于', '对于', '经过',
        '目前', '现在', '然后', '之后', '之前', '以后', '以上',
        '很多', '非常', '比较', '特别', '十分', '更加', '最为',
        '我们', '他们', '你们', '自己', '大家', '别人', '有人',
        '其中', '其他', '另外', '所有', '全部', '各种', '不同',
        '问题', '方法', '方式', '方面', '情况', '过程', '结果',
        '时间', '地方', '东西', '事情', '关系', '作用', '影响',
        # v8.2 扩展：高频虚词/动词
        '今天', '昨天', '明天', '今年', '去年', '明年', '最近',
        '受到', '加大', '改变', '正在', '越来越', '广泛', '保持',
        '面临', '提高', '包括', '重要', '关键', '主要', '基本',
        '一定', '必须', '不断', '继续', '开始', '结束', '完成',
        '具有', '存在', '发生', '出现', '成为', '形成', '产生',
        '带来', '造成', '引起', '导致', '实现', '达到', '取得',
        '来看', '来说', '而言', '来看', '来讲', '显得',
        '领域', '方面', '层面', '维度', '角度', '程度',
    }

    def __init__(self, data_dir: str = "./data"):
        self._nodes: Dict[str, BridgeNode] = {}
        self._edges: Dict[Tuple[str, str], CoEdge] = {}
        self._data_dir = Path(data_dir)
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._dirty = False
        self._load()

    # ── 实体提取 ──
    @classmethod
    def extract_terms(cls, text: str) -> List[str]:
        """从文本中提取有意义的实体词（jieba分词，不依赖LLM）"""
        if _HAS_JIEBA:
            words = jieba.lcut(text)
        else:
            words = cls._WORD_RE.findall(text)
        
        result = []
        for w in words:
            w = w.strip()
            if len(w) < 2:
                continue
            if w in cls._STOP_WORDS:
                continue
            if re.match(r'^[\d零一二三四五六七八九十百千万亿.%％]+$', w):
                continue
            # 过滤纯标点/空白
            if re.match(r'^[\s，。！？、；：""''（）【】《》…—\-,\.!?;:()\[\]{}]+$', w):
                continue
            result.append(w)
        # 去重保序
        seen = set()
        unique = []
        for w in result:
            if w not in seen:
                seen.add(w)
                unique.append(w)
        return unique

    # ── 学习共现 ──
    def learn(self, text: str, source: str = "", timestamp: float = None):
        """从一条文本中学习实体共现关系"""
        if timestamp is None:
            timestamp = time.time()
        
        terms = self.extract_terms(text)
        if len(terms) < 2:
            # 单个实体也要记录
            for t in terms:
                self._add_node(t, source, timestamp)
            return

        # 记录节点
        for t in terms:
            self._add_node(t, source, timestamp)

        # 记录共现边（同一条文本中出现的所有实体对）
        for i in range(len(terms)):
            for j in range(i + 1, len(terms)):
                a, b = terms[i], terms[j]
                key = (min(a, b), max(a, b))
                if key not in self._edges:
                    self._edges[key] = CoEdge(
                        count=0,
                        first_seen=timestamp,
                        last_seen=timestamp,
                        sources=set()
                    )
                edge = self._edges[key]
                edge.count += 1
                edge.last_seen = max(edge.last_seen, timestamp)
                edge.sources.add(source)

        self._dirty = True

    def _add_node(self, term: str, source: str, timestamp: float):
        if term not in self._nodes:
            self._nodes[term] = BridgeNode(
                term=term,
                frequency=0,
                last_seen=timestamp,
                sources=set(),
                first_seen=timestamp
            )
        node = self._nodes[term]
        node.frequency += 1
        node.last_seen = max(node.last_seen, timestamp)
        node.sources.add(source)

    # ── 语义发散 ──
    def explore(self, seed: str, top_k: int = 15,
                min_strength: float = 0.01,
                now: float = None) -> List[Tuple[str, float]]:
        """
        从种子词出发，发现关联概念
        
        策略：
        1. 直接共现（与种子词在同一文本中出现过的词）
        2. 间接共现（共现词的共现词，二阶扩散）
        3. 权重 = 共现次数 × 时间衰减 × 频率归一化
        
        Returns:
            [(term, strength), ...] 按强度降序
        """
        # 空查询/纯数字/纯标点过滤
        if not seed or not seed.strip():
            return []
        if seed.strip().isdigit():
            return []
        cleaned = re.sub(r'[^\w\u4e00-\u9fff]', '', seed)
        if not cleaned:
            return []

        if now is None:
            now = time.time()

        if seed not in self._nodes:
            # 种子词不在网络中：用 jieba 分词，找网络中最优匹配节点
            results = {}
            if _HAS_JIEBA:
                tokens = [w for w in jieba.lcut(seed) if len(w) >= 2 and w not in self._STOP_WORDS]
                in_net = [(tok, self._nodes[tok]) for tok in tokens if tok in self._nodes]
                if in_net:
                    # 取所有在网 token 的 source 交集做限定
                    common_sources = None
                    for tok, node in in_net:
                        if common_sources is None:
                            common_sources = node.sources.copy()
                        else:
                            common_sources &= node.sources
                    # 交集为空时，用频率最高节点的 sources
                    if not common_sources:
                        best_node_name = max(in_net, key=lambda x: x[1].frequency)[0]
                        common_sources = self._nodes[best_node_name].sources.copy()
                    
                    # 频率最高的节点做桥接
                    best_node = max(in_net, key=lambda x: x[1].frequency)[0]
                    for term, strength in self.explore(best_node, top_k * 2, min_strength, now):
                        if term != best_node:
                            # 用 source 交集过滤跨域噪声
                            if common_sources and term in self._nodes:
                                term_sources = self._nodes[term].sources
                                if term_sources and not (common_sources & term_sources):
                                    continue
                            results[term] = max(results.get(term, 0), strength * 0.8)
            # 回退：直接子串匹配
            if not results:
                for node_name, node in self._nodes.items():
                    if seed in node_name or node_name in seed:
                        for term, strength in self.explore(node_name, top_k, min_strength, now):
                            if term != node_name:
                                results[term] = max(results.get(term, 0), strength * 0.8)
                        break  # 只取第一个匹配，避免混源
            if results:
                return sorted(results.items(), key=lambda x: -x[1])[:top_k]
            return []

        seed_node = self._nodes[seed]
        seed_sources = seed_node.sources
        scores: Dict[str, float] = {}

        # 一阶：直接共现
        for (a, b), edge in self._edges.items():
            if seed not in (a, b):
                continue
            other = b if a == seed else a
            if other not in self._nodes:
                continue

            # 基础强度：共现次数 × 时间衰减
            age_days = (now - edge.last_seen) / 86400
            time_decay = math.exp(-age_days / 30)  # 30天半衰期
            freq_norm = math.log(1 + edge.count) / math.log(1 + max(1, self._nodes[other].frequency))
            
            strength = freq_norm * time_decay
            strength *= (0.5 + 0.5 * min(1.0, edge.count / 10))  # 低频共现打折

            if strength >= min_strength:
                scores[other] = max(scores.get(other, 0), strength)

        # 后置全局：source 过滤（跨域词直接排除）
        if seed_sources:
            for term in list(scores.keys()):
                if term in self._nodes:
                    term_sources = self._nodes[term].sources
                    if term_sources and not (seed_sources & term_sources):
                        scores[term] = 0.0  # 无交集直接归零

        # 二阶：共现词的共现词（轻度扩散，权重打折）
        first_order = list(scores.keys())
        for fo in first_order[:5]:  # 只扩散前5个最强的
            base = scores[fo]
            for (a, b), edge in self._edges.items():
                if fo not in (a, b):
                    continue
                other = b if a == fo else a
                if other == seed or other in scores:
                    continue
                if other not in self._nodes:
                    continue

                age_days = (now - edge.last_seen) / 86400
                time_decay = math.exp(-age_days / 30)
                strength = base * 0.3 * time_decay * min(1.0, edge.count / 5)
                
                if strength >= min_strength:
                    scores[other] = max(scores.get(other, 0), strength)

        # 二阶后再次 source 过滤
        if seed_sources:
            for term in list(scores.keys()):
                if term in self._nodes:
                    term_sources = self._nodes[term].sources
                    if term_sources and not (seed_sources & term_sources):
                        scores[term] = 0.0  # 无交集归零

        # 排序前移除零分项
        scores = {k: v for k, v in scores.items() if v > 0}
        # 排序
        sorted_results = sorted(scores.items(), key=lambda x: -x[1])
        return sorted_results[:top_k]

    def explore_multi(self, seeds: List[str], top_k: int = 20,
                      now: float = None) -> List[Tuple[str, float]]:
        """多种子联合发散"""
        merged: Dict[str, float] = {}
        weight = 1.0
        for seed in seeds:
            results = self.explore(seed, top_k=30, now=now)
            for term, strength in results:
                if term in seeds:
                    continue
                merged[term] = merged.get(term, 0) + strength * weight
            weight *= 0.7  # 后面的种子权重递减
        
        return sorted(merged.items(), key=lambda x: -x[1])[:top_k]

    # ── 网络查询 ──
    def get_network(self, seed: str = None, max_nodes: int = 50) -> dict:
        """获取网络结构（用于可视化）"""
        nodes = []
        edges = []

        if seed and seed in self._nodes:
            # 以种子为中心的子图
            related = self.explore(seed, top_k=max_nodes)
            node_set = {seed} | {r[0] for r in related}
        else:
            # 全图Top节点
            top_nodes = sorted(self._nodes.items(),
                             key=lambda x: -x[1].frequency)[:max_nodes]
            node_set = {n[0] for n in top_nodes}

        for term in node_set:
            if term in self._nodes:
                n = self._nodes[term]
                nodes.append({
                    "term": term,
                    "frequency": n.frequency,
                    "last_seen": n.last_seen,
                    "source_count": len(n.sources)
                })

        added_edges = set()
        for (a, b), edge in self._edges.items():
            if a in node_set and b in node_set:
                key = (min(a, b), max(a, b))
                if key not in added_edges:
                    added_edges.add(key)
                    edges.append({
                        "source": a,
                        "target": b,
                        "weight": edge.count,
                        "last_seen": edge.last_seen,
                        "source_count": len(edge.sources)
                    })

        return {
            "nodes": nodes,
            "edges": edges,
            "total_nodes": len(self._nodes),
            "total_edges": len(self._edges)
        }

    def stats(self) -> dict:
        return {
            "total_nodes": len(self._nodes),
            "total_edges": len(self._edges),
            "top_terms": sorted(
                [(t, n.frequency) for t, n in self._nodes.items()],
                key=lambda x: -x[1]
            )[:20],
            "densest_pairs": sorted(
                [((a, b), e.count) for (a, b), e in self._edges.items()],
                key=lambda x: -x[1]
            )[:10]
        }

    # ── 持久化 ──
    def save(self):
        if not self._dirty:
            return
        data = {
            "nodes": {
                t: {
                    "frequency": n.frequency,
                    "last_seen": n.last_seen,
                    "sources": list(n.sources),
                    "first_seen": n.first_seen
                }
                for t, n in self._nodes.items()
            },
            "edges": {
                f"{a}|||{b}": {
                    "count": e.count,
                    "first_seen": e.first_seen,
                    "last_seen": e.last_seen,
                    "sources": list(e.sources)
                }
                for (a, b), e in self._edges.items()
            }
        }
        path = self._data_dir / "bridge_network.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        self._dirty = False

    def _load(self):
        path = self._data_dir / "bridge_network.json"
        if not path.exists():
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            self._nodes = {}
            for t, d in data.get("nodes", {}).items():
                self._nodes[t] = BridgeNode(
                    term=t,
                    frequency=d["frequency"],
                    last_seen=d["last_seen"],
                    sources=set(d.get("sources", [])),
                    first_seen=d.get("first_seen", d["last_seen"])
                )

            self._edges = {}
            for key, d in data.get("edges", {}).items():
                a, b = key.split("|||")
                self._edges[(a, b)] = CoEdge(
                    count=d["count"],
                    first_seen=d["first_seen"],
                    last_seen=d["last_seen"],
                    sources=set(d.get("sources", []))
                )
        except Exception:
            pass  # 损坏文件，从零开始

    def clear(self):
        self._nodes.clear()
        self._edges.clear()
        self._dirty = True
        self.save()


# ═══════════════════════════════════════════════════════════
#  SemanticBridge — 多路径语义发散
# ═══════════════════════════════════════════════════════════

class SemanticBridge:
    """
    语义桥 — 多路径发散引擎
    
    综合三种信号：
    1. 共现网络（CoOccurrenceNetwork）— 统计共现
    2. 知识图谱边（graph_engine）— 显式关系
    3. 关键词扩展（规则）— 补漏
    
    与左脑的核心差异：
    - 左脑：随机哈希向量内积（语义无意义）
    - 知络：真实共现数据 + 图边关系（语义来自实际数据）
    """

    def __init__(self, co_net: CoOccurrenceNetwork, graph_engine=None):
        self.co_net = co_net
        self.graph = graph_engine

    def explore(self, query: str, top_k: int = 15,
                include_graph: bool = True) -> dict:
        """
        多路径语义发散
        
        Returns:
            {
                "seed": str,
                "co_occurrence": [(term, strength), ...],
                "graph_edges": [(term, relation), ...],
                "merged": [(term, strength), ...],
                "injection": str  # 可直接注入Agent上下文的文本
            }
        """
        # 空查询/纯数字/纯标点过滤
        if not query or not query.strip():
            return {"seed": query, "co_occurrence": [], "graph_edges": [], "merged": [], "injection": ""}
        if query.strip().isdigit():
            return {"seed": query, "co_occurrence": [], "graph_edges": [], "merged": [], "injection": ""}
        cleaned = re.sub(r'[^\w\u4e00-\u9fff]', '', query)
        if not cleaned:
            return {"seed": query, "co_occurrence": [], "graph_edges": [], "merged": [], "injection": ""}

        now = time.time()

        # 路径1：共现网络
        co_results = self.co_net.explore(query, top_k=top_k * 2, now=now)

        # 路径2：知识图谱边
        graph_results = []
        if include_graph and self.graph:
            try:
                # 尝试从图中找关联节点
                entities = self.graph.find_entity(query)
                if entities:
                    for eid, name, etype in entities[:3]:
                        neighbors = self.graph.get_neighbors(eid, max_hops=1)
                        for n in neighbors[:10]:
                            graph_results.append((n.get("name", ""), n.get("relation", "related")))
            except Exception:
                pass

        # 合并排序
        merged: Dict[str, float] = {}
        for term, strength in co_results:
            merged[term] = strength

        for term, rel in graph_results:
            if term not in merged:
                merged[term] = 0.4  # 图边基础分
            else:
                merged[term] = max(merged[term], 0.5)

        sorted_merged = sorted(merged.items(), key=lambda x: -x[1])[:top_k]

        # 生成注入文本
        if sorted_merged:
            top_terms = sorted_merged[:8]
            injection = (
                f"【语义桥】与「{query}」关联的概念：\n" +
                "\n".join(f"  • {t}（关联度 {s:.2f}）" for t, s in top_terms)
            )
        else:
            injection = ""

        return {
            "seed": query,
            "co_occurrence": co_results[:top_k],
            "graph_edges": graph_results[:top_k],
            "merged": sorted_merged,
            "injection": injection,
            "total_network_size": self.co_net.stats()["total_nodes"]
        }

    def explore_multi(self, queries: List[str], top_k: int = 20) -> dict:
        """多查询联合发散"""
        all_co = self.co_net.explore_multi(queries, top_k=top_k * 2)
        
        merged: Dict[str, float] = {}
        for term, strength in all_co:
            merged[term] = strength

        sorted_merged = sorted(merged.items(), key=lambda x: -x[1])[:top_k]
        seeds_str = " / ".join(queries)

        if sorted_merged:
            injection = (
                f"【语义桥】与「{seeds_str}」关联的概念：\n" +
                "\n".join(f"  • {t}（关联度 {s:.2f}）" for t, s in sorted_merged[:8])
            )
        else:
            injection = ""

        return {
            "seeds": queries,
            "co_occurrence": all_co[:top_k],
            "merged": sorted_merged,
            "injection": injection,
        }
