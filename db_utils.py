# -*- coding: utf-8 -*-
"""
知络 v8.9.7 — 统一数据库工具 db_utils.py

提供所有模块共用的数据库连接管理，消除各处重复的 _get_conn/_close_conn 模式。
推荐所有新代码使用 get_db_cursor() / get_db_connection() 上下文管理器。

用法：
    from db_utils import get_db_cursor, get_global_db_path

    # Context manager 风格（推荐）:
    with get_db_cursor() as cur:
        cur.execute("SELECT ...")
        rows = cur.fetchall()
    
    # 需要原始连接时:
    with get_db_connection() as conn:
        conn.execute("INSERT ...")
        conn.commit()

    # 获取数据库路径:
    db_path = get_global_db_path()
"""

import hashlib
import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def get_global_db_path():
    """获取 global.db 的绝对路径（统一入口）。

    优先级：kb_config.get_global_db() → 默认本地路径
    """
    try:
        from kb_config import get_global_db
        return get_global_db()
    except Exception:
        return ROOT / "data" / "workspaces" / "global.db"


@contextmanager
def get_db_cursor(db_path=None, row_factory=True, store=None):
    """上下文管理器：自动管理数据库连接生命周期。

    参数:
        db_path:   数据库路径（默认自动发现 global.db）
        row_factory: 是否设置 sqlite3.Row 作为 row_factory
        store:     MemoryStore 对象（优先使用其内部连接）

    用法:
        with get_db_cursor() as cur:
            cur.execute("SELECT * FROM nodes WHERE category=?", ("AI",))
            for row in cur.fetchall():
                print(row["text"])
    """
    conn = None
    should_close = True

    # 优先使用 store 的内部连接
    if store is not None:
        try:
            conn = store._get_conn()
            should_close = False
        except Exception:
            pass

    if conn is None:
        if db_path is None:
            db_path = str(get_global_db_path())
        conn = sqlite3.connect(str(db_path), timeout=30)
        conn.execute("PRAGMA busy_timeout=30000")
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        if row_factory:
            conn.row_factory = sqlite3.Row

    try:
        yield conn.cursor() if hasattr(conn, 'cursor') else conn
    finally:
        if should_close and conn:
            try:
                conn.close()
            except Exception:
                pass


@contextmanager
def get_db_connection(db_path=None, store=None):
    """v8.9.7: 上下文管理器，返回原始 sqlite3.Connection（含自动WAL配置）。

    用于需要 conn.execute().fetchall() 或 conn.commit() 的场景。
    推荐优先使用 get_db_cursor()，仅在需要原始连接时使用此函数。
    
    用法:
        with get_db_connection() as conn:
            conn.execute("INSERT INTO nodes(...) VALUES(...)")
            conn.commit()
    """
    conn = None
    should_close = True
    if store is not None:
        try:
            conn = store._get_conn()
            should_close = False
        except Exception:
            pass
    if conn is None:
        if db_path is None:
            db_path = str(get_global_db_path())
        conn = sqlite3.connect(str(db_path), timeout=30)
        conn.execute("PRAGMA busy_timeout=30000")
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        if should_close and conn:
            try:
                conn.close()
            except Exception:
                pass


def _node_columns(conn):
    return {r[1] for r in conn.execute("PRAGMA table_info(nodes)").fetchall()}


def _stable_hash(text):
    return hashlib.sha256((text or "").strip().encode("utf-8")).hexdigest()


def safe_insert_node(
    text,
    workspace="global",
    category="未分类",
    source="unknown",
    tags=None,
    confidence=1.0,
    para=None,
    created_by="scheduler",
    visibility="team",
    trust_score=1.0,
    source_url="",
    status="active",
    edges_json=None,
    db_path=None,
    conn=None,
    merge_duplicate=True,
    extra=None,
):
    """修复: UNIQUE 冲突 / 日期: 2026-07-04

    Unified node insert helper. Never passes an explicit nodes.id; SQLite owns
    AUTOINCREMENT. If the same content already exists, return that row instead
    of creating a duplicate.
    """
    if not text or not str(text).strip():
        raise ValueError("safe_insert_node requires non-empty text")

    own_conn = conn is None
    if conn is None:
        path = db_path or get_global_db_path()
        conn = sqlite3.connect(str(path), timeout=30)
        conn.execute("PRAGMA busy_timeout=30000")
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.row_factory = sqlite3.Row

    try:
        cols = _node_columns(conn)
        now = datetime.now().isoformat()
        text = str(text).strip()
        content_hash = _stable_hash(text)

        if merge_duplicate:
            try:
                if "content_hash" in cols:
                    row = conn.execute(
                        "SELECT id FROM nodes WHERE content_hash=? LIMIT 1",
                        (content_hash,),
                    ).fetchone()
                    if row:
                        return {"id": int(row[0]), "inserted": False, "merged": True}
                row = conn.execute(
                    "SELECT id FROM nodes WHERE text=? LIMIT 1",
                    (text,),
                ).fetchone()
                if row:
                    return {"id": int(row[0]), "inserted": False, "merged": True}
            except sqlite3.Error:
                pass

        tags_value = tags if tags is not None else []
        if not isinstance(tags_value, str):
            tags_value = json.dumps(tags_value, ensure_ascii=False)
        edge_value = edges_json if edges_json is not None else []
        if not isinstance(edge_value, str):
            edge_value = json.dumps(edge_value, ensure_ascii=False)

        values = {
            "text": text,
            "workspace": workspace or "global",
            "category": category or "未分类",
            "simhash": 0,
            "access_count": 0,
            "tags": tags_value,
            "confidence": float(confidence if confidence is not None else 1.0),
            "created_at": now,
            "learned_at": now,
            "updated_at": now,
            "last_accessed_at": now,
            "session_id": "",
            "source": source or "unknown",
            "edges_json": edge_value,
            "para": para,
            "created_by": created_by or "scheduler",
            "visibility": visibility or "team",
            "trust_score": float(trust_score if trust_score is not None else 1.0),
            "source_url": source_url or "",
            "content_hash": content_hash,
            "fetched_at": now,
            "status": status or "active",
            "confirmed_by": "",
            "text_jieba": "",
        }
        if extra:
            values.update({k: v for k, v in extra.items() if k != "id"})

        insert_cols = [c for c in values.keys() if c in cols and c != "id"]
        placeholders = ",".join("?" for _ in insert_cols)
        sql = "INSERT INTO nodes(%s) VALUES(%s)" % (",".join(insert_cols), placeholders)
        params = [values[c] for c in insert_cols]

        try:
            cur = conn.execute(sql, params)
        except sqlite3.IntegrityError as exc:
            if "nodes.id" in str(exc):
                max_id = conn.execute("SELECT COALESCE(MAX(id),0) FROM nodes").fetchone()[0]
                try:
                    conn.execute(
                        "UPDATE sqlite_sequence SET seq=? WHERE name='nodes'",
                        (int(max_id),),
                    )
                except sqlite3.Error:
                    pass
                cur = conn.execute(sql, params)
            else:
                raise

        if own_conn:
            conn.commit()
        return {"id": int(cur.lastrowid), "inserted": True, "merged": False}
    finally:
        if own_conn:
            conn.close()
