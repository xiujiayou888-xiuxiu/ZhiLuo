# -*- coding: utf-8 -*-
"""知络向量检索模块 — sqlite-vec + BGE 轻量 embedding"""
import numpy as np
import os, sqlite3, json, time, struct, threading
from pathlib import Path

_HAS_VEC = False
try:
    import sqlite_vec
    _HAS_VEC = True
except ImportError:
    pass

# embedding 模型（懒加载）
_EMBEDDER = None
_EMBEDDER_LOCK = threading.Lock()
_EMBED_DIM = 384

def _get_embedder():
    global _EMBEDDER
    if _EMBEDDER is not None:
        return _EMBEDDER
    with _EMBEDDER_LOCK:
        if _EMBEDDER is not None:
            return _EMBEDDER
        try:
            from sentence_transformers import SentenceTransformer
            _EMBEDDER = SentenceTransformer("all-MiniLM-L6-v2")
        except ImportError:
            _EMBEDDER = None
    return _EMBEDDER

def embed(text: str):
    """文本转向量，返回 bytes"""
    m = _get_embedder()
    if m is None:
        return None
    vec = m.encode(text, normalize_embeddings=True)
    return vec.astype(np.float32).tobytes()

def embed_batch(texts: list):
    """批量文本转向量"""
    m = _get_embedder()
    if m is None:
        return [None] * len(texts)
    vecs = m.encode(texts, normalize_embeddings=True)
    return [v.astype(np.float32).tobytes() for v in vecs]


class VectorStore:
    """SQLite + sqlite-vec 向量存储"""
    
    def __init__(self, db_path: str):
        self.db_path = str(db_path)
        self._conn = None
        self._init()
    
    def _get_conn(self):
        if self._conn is None:
            self._conn = sqlite3.connect(self.db_path)
            self._conn.execute("PRAGMA journal_mode=WAL")
            if _HAS_VEC:
                self._conn.enable_load_extension(True)
                sqlite_vec.load(self._conn)
                self._conn.enable_load_extension(False)
        return self._conn
    
    def _init(self):
        if not _HAS_VEC:
            return
        conn = self._get_conn()
        conn.execute("CREATE TABLE IF NOT EXISTS vec_texts (rowid INTEGER PRIMARY KEY, text TEXT, learn_at REAL)")
        conn.execute("CREATE VIRTUAL TABLE IF NOT EXISTS vec_items USING vec0(embedding float[%d])" % _EMBED_DIM)
        conn.commit()
    
    def add(self, node_id: int, text: str):
        """添加文本向量"""
        if not _HAS_VEC:
            return False
        vec_bytes = embed(text)
        if vec_bytes is None:
            return False
        conn = self._get_conn()
        try:
            conn.execute("INSERT OR REPLACE INTO vec_texts(rowid, text, learn_at) VALUES (?, ?, ?)",
                        (node_id, text, time.time()))
            conn.execute("INSERT OR REPLACE INTO vec_items(rowid, embedding) VALUES (?, ?)",
                        (node_id, vec_bytes))
            conn.commit()
            return True
        except Exception:
            return False
    
    def delete(self, node_id: int):
        """删除向量"""
        if not _HAS_VEC:
            return
        conn = self._get_conn()
        conn.execute("DELETE FROM vec_texts WHERE rowid=?", (node_id,))
        conn.execute("DELETE FROM vec_items WHERE rowid=?", (node_id,))
        conn.commit()
    
    def search(self, query: str, top_k: int = 10):
        """向量搜索，返回 [(node_id, text, distance), ...]"""
        if not _HAS_VEC:
            return []
        vec_bytes = embed(query)
        if vec_bytes is None:
            return []
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT rowid, distance FROM vec_items WHERE embedding MATCH ? ORDER BY distance LIMIT ?",
            (vec_bytes, top_k)
        ).fetchall()
        if not rows:
            return []
        results = []
        for rowid, dist in rows:
            t = conn.execute("SELECT text FROM vec_texts WHERE rowid=?", (rowid,)).fetchone()
            if t:
                results.append((rowid, t[0], dist))
        return results
    
    def rebuild(self, nodes: list):
        """从节点列表重建向量索引"""
        if not _HAS_VEC:
            return 0
        conn = self._get_conn()
        conn.execute("DELETE FROM vec_texts")
        conn.execute("DELETE FROM vec_items")
        conn.commit()
        
        texts = [n.get("text", "")[:500] for n in nodes if n.get("text")]
        node_ids = [n["id"] for n in nodes if n.get("text")]
        
        if not texts:
            return 0
        
        vecs = embed_batch(texts)
        count = 0
        for nid, text, vbytes in zip(node_ids, texts, vecs):
            if vbytes:
                conn.execute("INSERT INTO vec_texts(rowid, text, learn_at) VALUES (?, ?, ?)",
                           (nid, text, time.time()))
                conn.execute("INSERT INTO vec_items(rowid, embedding) VALUES (?, ?)",
                           (nid, vbytes))
                count += 1
        conn.commit()
        return count
    
    def stats(self):
        """状态"""
        if not _HAS_VEC:
            return {"enabled": False, "count": 0}
        conn = self._get_conn()
        count = conn.execute("SELECT COUNT(*) FROM vec_texts").fetchone()[0]
        return {"enabled": True, "count": count, "dim": _EMBED_DIM}
