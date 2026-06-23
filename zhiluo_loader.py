# -*- coding: utf-8 -*-
"""知络 v8.3 — 加载入口（开源版，图推理增强 + BrainWrapper增量层）"""
import sys, os
SKILL_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SKILL_DIR)

# 直接 import .py 源码
import engine as _engine
import entity as _entity
import vector as _vector
import graph_engine as _graph_engine
import relation_extractor as _relation_extractor
import graph_wrapper as _graph_wrapper

ZhiLuo = _engine.ZhiLuo
MemoryStore = _engine.MemoryStore
SimHash = _engine.SimHash
MinHashLSH = _engine.MinHashLSH
SelfCheck = _engine.SelfCheck
Intent = _engine.Intent

for n in ["pagerank","find","mermaid_graph","consolidate","entangle",
          "auto_learn","context_inject","smart_tokenize","smart_extract_keywords",
          "EDGE_RELATED","EDGE_CAUSE","EDGE_PART_OF","EDGE_SYNONYM",
          "EDGE_REFINES","EDGE_CONTRADICTS","EDGE_TYPES",
          "SKILL_DIR","DATA_DIR","WS_DIR","DATA_FILE"]:
    if hasattr(_engine, n):
        globals()[n] = getattr(_engine, n)

# v8.3 图推理增强 + BrainWrapper增量层
ZhiLuoGraph = _graph_wrapper.ZhiLuoGraph
GraphEngine = _graph_engine.GraphEngine
extract_relations = _relation_extractor.extract_relations
extract_entities = _relation_extractor.extract_entities
extract_all = _relation_extractor.extract_all
INDUSTRY_ENTITIES = _relation_extractor.INDUSTRY_ENTITIES

# v8.3 BrainWrapper 增量模块
try:
    from brain_wrapper import BrainWrapper, create_brain
    from knowledge_schema import RelationRegistry, RelationType, EntityMeta, EdgeMeta
    from context_memory import ContextMemory, create_context_memory
    from genre_retrieval import GenreClassifier, GenreAwareRetriever, SkeletonExtractor
    from semantic_bridge import CoOccurrenceNetwork, SemanticBridge
    from fast_index import FastIndex
    from tools import to_mermaid, export_as, pagerank, backup, restore
    from generic_extractor import extract_entities as generic_extract_entities, extract_relations as generic_extract_relations
    _BRAIN_OK = True
except Exception:
    _BRAIN_OK = False
    BrainWrapper = None
    create_brain = None
