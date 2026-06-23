# -*- coding: utf-8 -*-
"""
知络 v5.3 向量模块
优化3: sqlite-vec 持久化向量 + Ollama 本地 embedding
无 sqlite-vec 时自动回退到 TF-IDF cosine
"""
import re, math, json, struct
from collections import Counter

# ====================== sqlite-vec 检测 ======================
try:
    import sqlite3
    _conn_test = sqlite3.connect(":memory:")
    _conn_test.enable_load_extension(True)
    try:
        _conn_test.load_extension("vec0")
        _HAS_VEC = True
    except Exception:
        _HAS_VEC = False
    _conn_test.close()
except Exception:
    _HAS_VEC = False

# ====================== Ollama embedding 检测 ======================
try:
    import urllib.request
    _HAS_OLLAMA = True
except Exception:
    _HAS_OLLAMA = False

OLLAMA_URL = "http://localhost:11434/api/embeddings"
OLLAMA_MODEL = "qwen3-embedding"

def _get_ollama_embedding(text):
    """通过 Ollama 本地服务获取 embedding"""
    if not _HAS_OLLAMA:
        return None
    try:
        data = json.dumps({"model": OLLAMA_MODEL, "prompt": text}).encode("utf-8")
        req = urllib.request.Request(OLLAMA_URL, data=data, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            result = json.loads(resp.read().decode("utf-8"))
            return result.get("embedding")
    except Exception:
        return None

def _vec_to_blob(vec):
    """向量转 SQLite BLOB"""
    if not vec:
        return None
    return struct.pack(f'{len(vec)}f', *vec)

def _blob_to_vec(blob):
    """SQLite BLOB 转向量"""
    if not blob:
        return None
    n = len(blob) // 4
    return list(struct.unpack(f'{n}f', blob))

# ====================== TF-IDF 回退方案 ======================
def _tokenize(text):
    try:
        from engine import smart_tokenize
        return smart_tokenize(text)
    except ImportError:
        result = []
        for t in re.findall(r'[\u4e00-\u9fff]+', text.lower()):
            result.extend(list(t))
        for t in re.findall(r'[a-zA-Z0-9_]+', text.lower()):
            if len(t) >= 1:
                result.append(t.lower())
        return [t for t in result if len(t) >= 1]

def _cosine_sim(a, b):
    dot = sum(ai * bi for ai, bi in zip(a, b))
    na = math.sqrt(sum(ai * ai for ai in a))
    nb = math.sqrt(sum(bi * bi for bi in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)

def _tfidf_search(query, texts, top_k=10):
    if not texts or len(texts) < 2:
        return [(i, 0.0) for i in range(len(texts))]
    corpus = [_tokenize(t) for t in texts]
    vocab = sorted(set(w for doc in corpus for w in doc))
    if not vocab:
        return [(i, 0.0) for i in range(len(texts))]
    word_index = {w: i for i, w in enumerate(vocab)}
    n_docs = len(corpus)
    idf = {}
    for w in vocab:
        df = sum(1 for doc in corpus if w in doc)
        idf[w] = math.log((n_docs + 1) / (df + 1)) + 1
    tfidf = []
    for doc in corpus:
        tf = Counter(doc)
        max_tf = max(tf.values()) if tf else 1
        vec = [0.0] * len(vocab)
        for w, cnt in tf.items():
            if w in word_index:
                vec[word_index[w]] = (cnt / max_tf) * idf[w]
        tfidf.append(vec)
    qt = _tokenize(query)
    qtf = Counter(qt)
    max_qtf = max(qtf.values()) if qtf else 1
    qv = [0.0] * len(vocab)
    for w, cnt in qtf.items():
        if w in word_index:
            qv[word_index[w]] = (cnt / max_qtf) * idf[w]
    scores = [(i, _cosine_sim(qv, vec)) for i, vec in enumerate(tfidf)]
    scores.sort(key=lambda x: -x[1])
    return scores[:top_k]

# ====================== 统一接口 ======================
def semantic_search(query, texts, top_k=10):
    """优先用 Ollama embedding，回退 TF-IDF"""
    emb = _get_ollama_embedding(query)
    if emb:
        scored = []
        for i, text in enumerate(texts):
            text_emb = _get_ollama_embedding(text)
            if text_emb:
                sim = _cosine_sim(emb, text_emb)
                scored.append((i, sim))
            else:
                scored.append((i, 0.0))
        scored.sort(key=lambda x: -x[1])
        return scored[:top_k]
    return _tfidf_search(query, texts, top_k)

def semantic_find(store, query, workspace=None, top_k=10):
    if workspace:
        nodes = store.valid(workspace)
    else:
        nodes = store.valid()
    if len(nodes) < 2:
        return []
    texts = [n.get('text', '') for n in nodes]
    scores = semantic_search(query, texts, top_k=min(top_k, len(nodes)))
    results = []
    seen = set()
    for idx, score in scores:
        if score > 0.01:
            nid = nodes[idx]['id']
            if nid not in seen:
                seen.add(nid)
                n = store.get(nid)
                if n:
                    n['_semantic'] = round(score, 4)
                    results.append(n)
    return results

def store_embedding(conn, nid, text):
    """将 embedding 持久化到 SQLite（需要 sqlite-vec）"""
    if not _HAS_VEC:
        return False
    emb = _get_ollama_embedding(text)
    if not emb:
        return False
    blob = _vec_to_blob(emb)
    try:
        conn.execute("INSERT OR REPLACE INTO vec_nodes (id, embedding) VALUES (?, ?)", (nid, blob))
        conn.commit()
        return True
    except Exception:
        return False

def vec_search(conn, query, top_k=10):
    """sqlite-vec KNN 向量搜索"""
    if not _HAS_VEC:
        return []
    emb = _get_ollama_embedding(query)
    if not emb:
        return []
    blob = _vec_to_blob(emb)
    try:
        cursor = conn.execute(
            "SELECT id, distance FROM vec_nodes WHERE embedding MATCH ? ORDER BY distance LIMIT ?",
            (blob, top_k)
        )
        return [(row[0], row[1]) for row in cursor.fetchall()]
    except Exception:
        return []
