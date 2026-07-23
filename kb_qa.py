# -*- coding: utf-8 -*-
"""
知络 v8.9.1 — 知识库问答引擎 kb_qa.py

基于本地知识库的语义问答，不依赖外部 LLM API。
使用 TF-IDF + 关键词匹配在知识库中检索相关节点，
返回最匹配的答案。

用法:
  python kb_qa.py "食材成本占比多少"
  python kb_qa.py --top 5 "差评怎么处理"
"""

import sqlite3
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
try:
    from kb_config import get_global_db
    GLOBAL_DB = get_global_db()
except ImportError:
    GLOBAL_DB = ROOT / "data" / "workspaces" / "global.db"


def _get_conn():
    con = sqlite3.connect(str(GLOBAL_DB), timeout=30)
    con.execute("PRAGMA busy_timeout=30000")
    return con


def _tokenize(text):
    """简单中文分词：按2-gram切分。"""
    text = re.sub(r"[^\u4e00-\u9fff\w]", " ", text.lower())
    tokens = []
    for i in range(len(text) - 1):
        tokens.append(text[i:i + 2])
    # 也加入单字
    tokens.extend(text)
    return tokens


def _jaccard_sim(tokens_a, tokens_b):
    """Jaccard 相似度。"""
    set_a = set(tokens_a)
    set_b = set(tokens_b)
    if not set_a or not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union


def search(query, top_k=5, min_score=0.05):
    """在知识库中搜索与 query 最相关的节点。
    
    返回: [(node_dict, score), ...] 按相关度降序
    """
    query_tokens = _tokenize(query)
    if not query_tokens:
        return []

    # 提取关键词（2字以上中文词）
    keywords = set()
    for i in range(len(query)):
        for j in range(3, 7):  # 3-6字
            if i + j <= len(query):
                w = query[i:i + j]
                if re.match(r"^[\u4e00-\u9fff]+$", w):
                    keywords.add(w)
    # 2字词
    for i in range(len(query) - 1):
        w = query[i:i + 2]
        if re.match(r"^[\u4e00-\u9fff]+$", w):
            keywords.add(w)

    conn = _get_conn()
    try:
        cur = conn.execute(
            "SELECT id, text, category, confidence, source, tags, created_at "
            "FROM nodes WHERE status != 'merged' ORDER BY confidence DESC"
        )
        nodes = [dict(zip([d[0] for d in cur.description], r)) for r in cur.fetchall()]
    finally:
        conn.close()

    results = []
    for node in nodes:
        text = node.get("text", "")
        text_tokens = _tokenize(text)

        # 基础 Jaccard 相似度
        jaccard = _jaccard_sim(query_tokens, text_tokens)

        # 关键词命中加分
        kw_score = 0
        for kw in keywords:
            if kw in text:
                kw_score += 0.15
            if kw in (node.get("tags") or ""):
                kw_score += 0.1

        # 综合得分
        score = jaccard * 0.6 + min(kw_score, 0.4) + node.get("confidence", 0.5) * 0.05

        if score >= min_score:
            results.append((node, score))

    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_k]


def answer(query, top_k=3):
    """基于知识库回答一个问题。
    
    返回: dict {answer, sources, confidence}
    """
    results = search(query, top_k=top_k)

    if not results:
        return {
            "answer": "知识库中暂无相关信息。",
            "sources": [],
            "confidence": 0.0,
        }

    # 最佳匹配
    best_node, best_score = results[0]

    if best_score >= 0.5:
        confidence = "高"
    elif best_score >= 0.3:
        confidence = "中"
    else:
        confidence = "低"

    return {
        "answer": best_node.get("text", ""),
        "sources": [
            {
                "id": str(n["id"])[:6],
                "text": n["text"][:100],
                "category": n.get("category", ""),
                "score": round(s, 3),
            }
            for n, s in results
        ],
        "confidence": confidence,
    }


def quick_lookup(keyword):
    """快速关键词查找（精确匹配 tags/category）。"""
    conn = _get_conn()
    try:
        cur = conn.execute(
            "SELECT id, text, category, tags FROM nodes "
            "WHERE (tags LIKE ? OR category LIKE ? OR text LIKE ?) "
            "AND status != 'merged' "
            "ORDER BY confidence DESC LIMIT 10",
            (f"%{keyword}%", f"%{keyword}%", f"%{keyword}%")
        )
        return [dict(zip([d[0] for d in cur.description], r)) for r in cur.fetchall()]
    finally:
        conn.close()


def stats():
    """知识库问题回答能力统计。"""
    conn = _get_conn()
    try:
        total = conn.execute("SELECT COUNT(*) FROM nodes WHERE status != 'merged'").fetchone()[0]
        categories = conn.execute(
            "SELECT category, COUNT(*) as cnt FROM nodes WHERE status != 'merged' "
            "GROUP BY category ORDER BY cnt DESC"
        ).fetchall()
        by_source = conn.execute(
            "SELECT source, COUNT(*) as cnt FROM nodes WHERE status != 'merged' "
            "GROUP BY source ORDER BY cnt DESC"
        ).fetchall()
        return {
            "total_nodes": total,
            "categories": dict(categories),
            "by_source": dict(by_source),
        }
    finally:
        conn.close()


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("知络 v8.9.1 知识库问答")
        print("用法: python kb_qa.py <问题>")
        print("      python kb_qa.py --top 5 <问题>")
        print("      python kb_qa.py --stats")
        sys.exit(0)

    if sys.argv[1] == "--stats":
        s = stats()
        print(f"\n📊 知识库统计:")
        print(f"  总节点: {s['total_nodes']}")
        print(f"  分类: {s['categories']}")
        print(f"  来源: {s['by_source']}")
    else:
        top_k = 3
        query_start = 1
        if sys.argv[1] == "--top":
            top_k = int(sys.argv[2])
            query_start = 3

        query = " ".join(sys.argv[query_start:])
        result = answer(query, top_k=top_k)

        print(f"\n❓ 问题: {query}")
        print(f"📊 置信度: {result['confidence']}")
        print(f"\n📝 答案: {result['answer']}")
        if result["sources"]:
            print(f"\n📚 相关来源 ({len(result['sources'])}):")
            for s in result["sources"]:
                print(f"  [{s['id']}] ({s['score']}) {s['text']}...")
