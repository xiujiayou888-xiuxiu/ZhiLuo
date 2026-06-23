# -*- coding: utf-8 -*-
"""
知络 v8.0 知识模型 — 关系类型注册表 + 元数据结构
引擎不预设任何行业规则，所有关系类型由调用方注册
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List, Dict, Any
import json


@dataclass
class RelationType:
    """关系类型定义"""
    name: str                       # 关系名: part_of / affect / cause / substitute / synonym / related
    propagation: str                # 传播方向: forward / backward / bidirectional / none
    decay_factor: float = 0.5       # 默认衰减系数
    transitive: bool = False        # 是否可传递 (A→B, B→C ⇒ A→C)
    symmetric: bool = False         # 是否对称 (A→B ⇒ B→A)
    description: str = ""           # 说明

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "propagation": self.propagation,
            "decay_factor": self.decay_factor,
            "transitive": self.transitive,
            "symmetric": self.symmetric,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "RelationType":
        return cls(**d)


class RelationRegistry:
    """关系类型注册表 — 引擎不预设行业规则"""

    # PARA 知识分类法则（Tiago Forte）
    # P=Projects(有截止日的短期目标) A=Areas(需持续维护的长期领域)
    # R=Resources(未来有用的参考资源) A=Archives(不再活跃的归档内容)
    PARA_CATEGORIES = {
        "project": "P-项目：有明确目标和截止日的短期任务",
        "area": "A-领域：需持续关注和维护的长期责任",
        "resource": "R-资源：未来可能用得上的参考素材",
        "archive": "A-归档：已完成或过期的内容，保留备查",
    }

    def __init__(self):
        self._types: Dict[str, RelationType] = {}
        self._register_defaults()

    def _register_defaults(self):
        """注册最基础的关系类型，不含任何行业语义"""
        defaults = [
            RelationType("part_of", "backward", 0.8, True, False,
                         "A是B的组成部分 → B的成本变化影响A"),
            RelationType("affect", "forward", 0.7, True, False,
                         "A影响B → A的变化单向传播到B"),
            RelationType("cause", "forward", 0.8, True, False,
                         "A导致B → A的变化单向传播到B"),
            RelationType("substitute", "bidirectional", 0.6, True, True,
                         "A可替代B → 双向传播，替代互惠"),
            RelationType("synonym", "bidirectional", 1.0, True, True,
                         "A和B同义 → 等价传播"),
            RelationType("related", "forward", 0.3, False, False,
                         "A和B相关 → 弱关联，衰减传播"),
            RelationType("depend_on", "backward", 0.7, True, False,
                         "A依赖B → B的变化影响A"),
            RelationType("belongs_to", "none", 0.0, False, False,
                         "A属于B分类 → 不传播成本"),
        ]
        for rt in defaults:
            self.register(rt)

    def register(self, rt: RelationType):
        self._types[rt.name] = rt

    def get(self, name: str) -> Optional[RelationType]:
        return self._types.get(name)

    def propagation_direction(self, relation_name: str) -> str:
        """返回关系类型的传播方向"""
        rt = self._types.get(relation_name)
        return rt.propagation if rt else "forward"

    def decay(self, relation_name: str) -> float:
        """返回关系类型的衰减系数"""
        rt = self._types.get(relation_name)
        return rt.decay_factor if rt else 0.3

    def is_transitive(self, relation_name: str) -> bool:
        rt = self._types.get(relation_name)
        return rt.transitive if rt else False

    def list_all(self) -> List[Dict]:
        return [rt.to_dict() for rt in self._types.values()]

    def to_json(self) -> str:
        return json.dumps(self.list_all(), ensure_ascii=False, indent=2)

    def __repr__(self):
        return f"RelationRegistry({len(self._types)} types)"


# ── 实体/边元数据结构 ──

def now_iso() -> str:
    return datetime.now().strftime("%Y-%m-%dT%H:%M:%S")


@dataclass
class EntityMeta:
    """实体节点元数据"""
    name: str
    entity_type: str = "entity"
    confidence: float = 1.0
    source: str = "manual"          # manual / llm / auto_crawl / import
    created_at: str = field(default_factory=now_iso)
    tags: List[str] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "entity_type": self.entity_type,
            "confidence": self.confidence,
            "source": self.source,
            "created_at": self.created_at,
            "tags": self.tags,
            "meta": self.meta,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "EntityMeta":
        return cls(
            name=d.get("name", ""),
            entity_type=d.get("entity_type", "entity"),
            confidence=d.get("confidence", 1.0),
            source=d.get("source", "manual"),
            created_at=d.get("created_at", now_iso()),
            tags=d.get("tags", []),
            meta=d.get("meta", {}),
        )


@dataclass
class EdgeMeta:
    """关系边元数据"""
    relation: str = "related"
    weight: float = 1.0
    confidence: float = 0.5
    trigger: str = ""               # 触发事件: "涨价" / "断供"
    change_pct: float = 0.0         # 变化百分比
    created_at: str = field(default_factory=now_iso)
    source: str = "auto"            # manual / llm / auto
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "relation": self.relation,
            "weight": self.weight,
            "confidence": self.confidence,
            "trigger": self.trigger,
            "change_pct": self.change_pct,
            "created_at": self.created_at,
            "source": self.source,
            "meta": self.meta,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "EdgeMeta":
        return cls(
            relation=d.get("relation", "related"),
            weight=d.get("weight", 1.0),
            confidence=d.get("confidence", 0.5),
            trigger=d.get("trigger", ""),
            change_pct=d.get("change_pct", 0.0),
            created_at=d.get("created_at", now_iso()),
            source=d.get("source", "auto"),
            meta=d.get("meta", {}),
        )
