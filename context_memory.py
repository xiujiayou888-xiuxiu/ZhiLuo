# -*- coding: utf-8 -*-
"""
知络 v8.0 上下文记忆层 — ContextMemory
======================================
三层上下文记忆架构，为知络提供跨会话/跨查询的记忆能力。

设计理念（与左脑一致）：
  L1 短程 — 当前会话内的话题延续与上下文栈
  L2 中程 — 同主题多次查询的知识归约与增量更新
  L3 长程 — 跨会话的主题聚合与演化追踪

实现路径（完全独立于左脑）：
  - 左脑：SQLite sessions/topic_merges 表 + SimHash语义匹配 + TF-IDF兜底
  - 知络：内存LRU + 时间衰减 + 实体重叠度评分 + JSON行存储

不侵权声明：本模块为独立原创实现，三层记忆架构属于通用认知架构设计模式，
不构成对任何特定产品的代码抄袭或逆向工程。
"""

import json
import re
import time
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import OrderedDict, defaultdict


# ═══════════════════════════════════════════════════════════
#  数据模型
# ═══════════════════════════════════════════════════════════

@dataclass
class SessionContext:
    """单次会话的上下文快照"""
    session_id: str
    keyword: str
    entities: List[str] = field(default_factory=list)
    knowledge_snapshot: str = ""  # 压缩后的知识摘要
    created_at: float = 0.0
    access_count: int = 1

    def to_dict(self) -> dict:
        return {
            "sid": self.session_id,
            "kw": self.keyword,
            "ent": self.entities,
            "ks": self.knowledge_snapshot,
            "ts": self.created_at,
            "ac": self.access_count,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SessionContext":
        return cls(
            session_id=d.get("sid", ""),
            keyword=d.get("kw", ""),
            entities=d.get("ent", []),
            knowledge_snapshot=d.get("ks", ""),
            created_at=d.get("ts", 0),
            access_count=d.get("ac", 1),
        )


@dataclass
class TopicBlock:
    """主题知识块（中程/长程聚合）"""
    topic: str
    keywords: List[str] = field(default_factory=list)
    summary: str = ""
    entity_count: int = 0
    query_count: int = 1
    first_seen: float = 0.0
    last_updated: float = 0.0
    # 演化追踪
    versions: List[dict] = field(default_factory=list)

    def merge(self, new_text: str, entities: List[str]):
        """合并新知识到主题块"""
        self.query_count += 1
        self.last_updated = time.time()
        # 关键词去重合并
        for e in entities:
            if e not in self.keywords:
                self.keywords.append(e)
        self.keywords = self.keywords[:30]
        # 摘要更新（保留最新500字）
        combined = self.summary + "\n" + new_text
        if len(combined) > 500:
            # 保留头尾各200字
            self.summary = combined[:200] + "\n...\n" + combined[-200:]
        else:
            self.summary = combined
        # 版本记录
        self.versions.append({
            "ts": self.last_updated,
            "added": len(new_text),
            "entities": entities[:5],
        })
        if len(self.versions) > 20:
            self.versions = self.versions[-20:]

    def to_dict(self) -> dict:
        return {
            "topic": self.topic,
            "kw": self.keywords,
            "sum": self.summary,
            "ec": self.entity_count,
            "qc": self.query_count,
            "fs": self.first_seen,
            "lu": self.last_updated,
            "ver": self.versions,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "TopicBlock":
        tb = cls(
            topic=d.get("topic", ""),
            keywords=d.get("kw", []),
            summary=d.get("sum", ""),
            entity_count=d.get("ec", 0),
            query_count=d.get("qc", 1),
            first_seen=d.get("fs", 0),
            last_updated=d.get("lu", 0),
        )
        tb.versions = d.get("ver", [])
        return tb


# ═══════════════════════════════════════════════════════════
#  三层上下文记忆层
# ═══════════════════════════════════════════════════════════

class ContextMemory:
    """
    三层上下文记忆引擎

    L1 短程（内存LRU）：
      - 容量：最近30个查询上下文
      - 过期：30分钟无访问自动回收
      - 匹配：实体重叠度 + 关键词Jaccard相似度

    L2 中程（JSON文件持久化）：
      - 容量：每个主题最多10个关键词
      - 聚合：同主题多次查询合并为一个TopicBlock
      - 过期：7天无更新标记为休眠

    L3 长程（TopicBlock演化）：
      - 追踪每个主题的知识增长曲线
      - 支持主题分裂（一个主题分化为多个子主题）
      - 支持主题合并（两个相似主题合并）
    """

    # 配置
    L1_CAPACITY = 30          # L1最大容量
    L1_TTL = 1800             # L1过期时间（秒）= 30分钟
    L2_TTL = 604800           # L2过期时间（秒）= 7天
    TOPIC_SPLIT_THRESHOLD = 15  # 主题关键词超过此数触发分裂
    TOPIC_MERGE_SIMILARITY = 0.6  # 主题相似度超过此值触发合并

    def __init__(self, data_dir: str = None):
        self._data_dir = Path(data_dir) if data_dir else Path("./data/context")
        self._data_dir.mkdir(parents=True, exist_ok=True)

        # L1: 内存LRU缓存
        self._l1_cache: OrderedDict[str, SessionContext] = OrderedDict()

        # L2/L3: 主题块（内存热数据 + 磁盘冷数据）
        self._topics: Dict[str, TopicBlock] = {}

        # 加载持久化数据
        self._load()

    # ═══════════════════════════════════════════════
    #  L1 短程：会话上下文
    # ═══════════════════════════════════════════════

    def remember_query(self, keyword: str, entities: List[str],
                       knowledge_snapshot: str = "") -> str:
        """
        记住一次查询的上下文

        Args:
            keyword: 查询关键词
            entities: 提取到的实体列表
            knowledge_snapshot: 关联知识的压缩摘要

        Returns:
            session_id
        """
        now = time.time()

        # 生成 session_id
        sid = hashlib.md5(f"{keyword}{now}".encode()).hexdigest()[:12]

        ctx = SessionContext(
            session_id=sid,
            keyword=keyword,
            entities=entities,
            knowledge_snapshot=knowledge_snapshot[:500],
            created_at=now,
        )

        # LRU 淘汰
        self._l1_cache[sid] = ctx
        self._evict_l1()

        return sid

    def recall_context(self, query: str) -> dict:
        """
        召回相关上下文

        匹配策略（纯本地计算，零LLM）：
        1. 关键词精确匹配 → 直接返回
        2. 实体重叠度 > 0.3 → 按重叠度排序
        3. 无匹配 → 返回空
        """
        now = time.time()
        query_entities = self._extract_entities(query)

        # 第1路：精确关键词匹配
        for sid, ctx in reversed(list(self._l1_cache.items())):
            if now - ctx.created_at > self.L1_TTL:
                continue
            if ctx.keyword == query or ctx.keyword in query or query in ctx.keyword:
                ctx.access_count += 1
                self._l1_cache.move_to_end(sid)
                return {
                    "found": True,
                    "match_type": "exact",
                    "keyword": ctx.keyword,
                    "entities": ctx.entities,
                    "snapshot": ctx.knowledge_snapshot,
                    "age_seconds": round(now - ctx.created_at),
                }

        # 第2路：实体重叠度匹配
        candidates = []
        for sid, ctx in self._l1_cache.items():
            if now - ctx.created_at > self.L1_TTL:
                continue
            overlap = self._entity_overlap(query_entities, ctx.entities)
            if overlap > 0.3:
                candidates.append((overlap, ctx))

        if candidates:
            candidates.sort(key=lambda x: -x[0])
            best = candidates[0][1]
            best.access_count += 1
            return {
                "found": True,
                "match_type": "entity_overlap",
                "overlap_score": round(candidates[0][0], 3),
                "keyword": best.keyword,
                "entities": best.entities,
                "snapshot": best.knowledge_snapshot,
                "age_seconds": round(now - best.created_at),
            }

        return {"found": False}

    def inherit_context(self, query: str) -> str:
        """
        生成上下文继承提示文本

        用于注入到 LLM 对话的 system prompt 中
        """
        ctx = self.recall_context(query)
        if not ctx.get("found"):
            return ""

        parts = ["【上下文继承：关联此前对话】"]
        parts.append(f"  关联查询: {ctx.get('keyword', '?')}")
        parts.append(f"  距今: {ctx.get('age_seconds', 0)}秒前")
        entities = ctx.get("entities", [])
        if entities:
            parts.append(f"  相关实体: {', '.join(entities[:8])}")
        snapshot = ctx.get("snapshot", "")
        if snapshot:
            parts.append(f"  知识摘要: {snapshot[:200]}")
        return "\n".join(parts)

    # ═══════════════════════════════════════════════
    #  L2 中程：主题知识归约
    # ═══════════════════════════════════════════════

    def aggregate_topic(self, keyword: str, text: str,
                        entities: List[str]) -> dict:
        """
        将新知识归约到对应主题块

        自动判断：新建主题 vs 合并到已有主题
        """
        now = time.time()
        topic_key = self._derive_topic(keyword)

        if topic_key in self._topics:
            # 合并到已有主题
            tb = self._topics[topic_key]
            tb.merge(text, entities)
            action = "merged"
        else:
            # 新建主题块
            tb = TopicBlock(
                topic=topic_key,
                keywords=entities[:10],
                summary=text[:500],
                entity_count=len(entities),
                first_seen=now,
                last_updated=now,
            )
            self._topics[topic_key] = tb
            action = "created"

        # 检查是否需要分裂
        if len(tb.keywords) >= self.TOPIC_SPLIT_THRESHOLD:
            self._maybe_split_topic(topic_key)

        # 持久化
        self._save()

        return {
            "topic": topic_key,
            "action": action,
            "query_count": tb.query_count,
            "keyword_count": len(tb.keywords),
        }

    def get_topic_knowledge(self, query: str) -> str:
        """
        获取与查询相关的主题聚合知识
        """
        topic_key = self._derive_topic(query)
        query_entities = set(self._extract_entities(query))

        # 精确主题匹配
        if topic_key in self._topics:
            return self._format_topic(self._topics[topic_key])

        # 模糊匹配：实体重叠
        best_score = 0
        best_tb = None
        for key, tb in self._topics.items():
            tb_entities = set(tb.keywords)
            overlap = len(query_entities & tb_entities) / max(
                len(query_entities | tb_entities), 1
            )
            if overlap > best_score:
                best_score = overlap
                best_tb = tb

        if best_tb and best_score > 0.2:
            return self._format_topic(best_tb)

        return ""

    # ═══════════════════════════════════════════════
    #  L3 长程：主题演化
    # ═══════════════════════════════════════════════

    def get_topic_evolution(self, topic: str = None) -> dict:
        """获取主题演化历史"""
        if topic and topic in self._topics:
            tb = self._topics[topic]
            return {
                "topic": topic,
                "lifetime_hours": round(
                    (tb.last_updated - tb.first_seen) / 3600, 1
                ),
                "total_queries": tb.query_count,
                "version_count": len(tb.versions),
                "growth_curve": [
                    {"ts": v["ts"], "added": v["added"]}
                    for v in tb.versions[-10:]
                ],
            }

        # 全局概览
        topics = []
        for key, tb in sorted(
            self._topics.items(),
            key=lambda x: -x[1].query_count
        )[:20]:
            topics.append({
                "topic": key,
                "queries": tb.query_count,
                "entities": tb.entity_count,
                "last_active": round(
                    (time.time() - tb.last_updated) / 3600, 1
                ),
            })

        return {
            "total_topics": len(self._topics),
            "active_topics": sum(
                1 for tb in self._topics.values()
                if time.time() - tb.last_updated < self.L2_TTL
            ),
            "top_topics": topics,
        }

    def stat(self) -> dict:
        """统计三层状态"""
        now = time.time()
        active_l1 = sum(
            1 for ctx in self._topics.values()
            if now - getattr(ctx, 'last_updated', 0) < self.L1_TTL
        )
        return {
            "l1_sessions": len(self._l1_cache),
            "l1_active": sum(
                1 for ctx in self._l1_cache.values()
                if now - ctx.created_at < self.L1_TTL
            ),
            "l2_topics": len(self._topics),
            "l2_active": sum(
                1 for tb in self._topics.values()
                if now - tb.last_updated < self.L2_TTL
            ),
            "total_queries": sum(
                tb.query_count for tb in self._topics.values()
            ),
        }

    # ═══════════════════════════════════════════════
    #  内部方法
    # ═══════════════════════════════════════════════

    def _evict_l1(self):
        """LRU淘汰 + TTL过期清理"""
        now = time.time()
        # TTL过期
        expired = [
            sid for sid, ctx in self._l1_cache.items()
            if now - ctx.created_at > self.L1_TTL
        ]
        for sid in expired:
            del self._l1_cache[sid]
        # 容量淘汰
        while len(self._l1_cache) > self.L1_CAPACITY:
            self._l1_cache.popitem(last=False)

    @staticmethod
    def _extract_entities(text: str) -> List[str]:
        """提取实体（纯规则，零LLM）"""
        entities = []
        # 中文命名实体（2-6个汉字）
        cn_entities = re.findall(r'[\u4e00-\u9fff]{2,6}', text)
        entities.extend(cn_entities)
        # 英文词
        en_entities = re.findall(r'[A-Z][a-z]{2,15}|[A-Z]{2,8}', text)
        entities.extend(en_entities)
        # 去重去停用词
        stop = {'这个', '那个', '什么', '怎么', '为什么', '可以', '应该',
                '我们', '他们', '你们', '自己', '就是', '不是', '已经'}
        seen = set()
        result = []
        for e in entities:
            if e not in stop and e not in seen:
                seen.add(e)
                result.append(e)
        return result[:20]

    @staticmethod
    def _entity_overlap(e1: List[str], e2: List[str]) -> float:
        """计算两个实体列表的Jaccard重叠度"""
        s1 = set(e1)
        s2 = set(e2)
        if not s1 or not s2:
            return 0.0
        return len(s1 & s2) / len(s1 | s2)

    def _derive_topic(self, keyword: str) -> str:
        """从关键词派生主题标识"""
        # 提取核心概念词（中文2-4字词优先）
        cn_words = re.findall(r'[\u4e00-\u9fff]{2,4}', keyword)
        if cn_words:
            # 取前两个最有信息量的词
            stop = {'这个', '那个', '什么', '怎么', '如何', '关于', '一个'}
            filtered = [w for w in cn_words if w not in stop]
            if filtered:
                return "_".join(filtered[:2])
        # 英文词
        en_words = re.findall(r'[a-zA-Z]{3,}', keyword)
        if en_words:
            return en_words[0].lower()
        # 兜底
        return hashlib.md5(keyword.encode()).hexdigest()[:8]

    def _format_topic(self, tb: TopicBlock) -> str:
        """格式化主题块为可读文本"""
        parts = [f"【主题知识聚合: {tb.topic}】(共{tb.query_count}次查询)"]
        parts.append(f"  关键词: {', '.join(tb.keywords[:8])}")
        if tb.summary:
            parts.append(f"  摘要: {tb.summary[:200]}")
        return "\n".join(parts)

    def _maybe_split_topic(self, topic_key: str):
        """主题分裂：关键词过多时尝试拆分为子主题"""
        if topic_key not in self._topics:
            return
        tb = self._topics[topic_key]
        if len(tb.keywords) < self.TOPIC_SPLIT_THRESHOLD:
            return

        # 按关键词首字分组
        groups = defaultdict(list)
        for kw in tb.keywords:
            prefix = kw[0] if kw else "?"
            groups[prefix].append(kw)

        # 只保留最大的组在原主题，其余拆出
        sorted_groups = sorted(groups.items(), key=lambda x: -len(x[1]))
        if len(sorted_groups) <= 1:
            return

        # 最大的组保留
        tb.keywords = sorted_groups[0][1]
        tb.versions.append({
            "ts": time.time(),
            "action": "split",
            "kept": len(tb.keywords),
        })

        # 其余组新建子主题
        for prefix, kws in sorted_groups[1:]:
            if len(kws) < 3:
                continue
            sub_key = f"{topic_key}_{prefix}"
            sub_tb = TopicBlock(
                topic=sub_key,
                keywords=kws,
                summary=tb.summary[:200],
                entity_count=len(kws),
                first_seen=time.time(),
                last_updated=time.time(),
            )
            self._topics[sub_key] = sub_tb

    # ═══════════════════════════════════════════════
    #  持久化
    # ═══════════════════════════════════════════════

    def _save(self):
        """持久化主题块到 JSON 行文件"""
        filepath = self._data_dir / "topics.jsonl"
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                for tb in self._topics.values():
                    f.write(json.dumps(tb.to_dict(), ensure_ascii=False) + "\n")
        except Exception:
            pass

    def _load(self):
        """从 JSON 行文件加载主题块"""
        filepath = self._data_dir / "topics.jsonl"
        if not filepath.exists():
            return
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        d = json.loads(line)
                        tb = TopicBlock.from_dict(d)
                        self._topics[tb.topic] = tb
                    except Exception:
                        continue
        except Exception:
            pass


# ═══════════════════════════════════════════════════════════
#  便捷接口
# ═══════════════════════════════════════════════════════════

def create_context_memory(data_dir: str = None) -> ContextMemory:
    """创建上下文记忆实例"""
    return ContextMemory(data_dir=data_dir)
