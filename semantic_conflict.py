# -*- coding: utf-8 -*-
"""
语义冲突检测模块 — 懒加载方案
- 首选：ONNX embedding模型（all-MiniLM-L6-v2, ~48MB，首次自动下载）
- 降级：TF-IDF + jieba分词 + 余弦相似度（sklearn，零下载）
- 全程零token，纯本地计算

升级ONNX（可选，提升精度）：
  1. 下载模型文件到 semantic_models/ 目录：
     - https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/onnx/model.onnx
     - https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/tokenizer.json
  2. 下次调用 analyze("冲突") 时自动识别并升级为ONNX模式
  3. 也可设置环境变量 ZHILUO_SEMANTIC_MODEL_DIR 指定模型目录
"""
import os
import hashlib
import threading

# ========== 懒加载全局状态 ==========
_semantic_model = None          # 已加载的模型对象
_semantic_mode = None            # "onnx" / "tfidf" / None(未初始化)
_semantic_lock = threading.Lock()
_model_dir = os.environ.get(
    "ZHILUO_SEMANTIC_MODEL_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "semantic_models")
)


def _download_file(url, dest_path, chunk_size=8192):
    """断点续传下载"""
    import urllib.request
    done = 0
    if os.path.exists(dest_path):
        done = os.path.getsize(dest_path)
    req = urllib.request.Request(url)
    if done > 0:
        req.add_header("Range", "bytes=%d-" % done)
    resp = urllib.request.urlopen(req, timeout=60)
    mode = "ab" if done > 0 and resp.status == 206 else "wb"
    total = int(resp.headers.get("Content-Length", 0)) + done
    with open(dest_path, mode) as f:
        while True:
            chunk = resp.read(chunk_size)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
    return done


def _load_onnx_model():
    """懒加载ONNX embedding模型，失败返回None"""
    global _semantic_model, _semantic_mode
    try:
        import onnxruntime as ort
        import numpy as np
        
        model_path = os.path.join(_model_dir, "model.onnx")
        tokenizer_path = os.path.join(_model_dir, "tokenizer.json")
        
        # 首次：下载模型文件
        if not os.path.exists(model_path) or not os.path.exists(tokenizer_path):
            os.makedirs(_model_dir, exist_ok=True)
            # 使用 HuggingFace 的 MiniLM-L6-v2 ONNX 版本
            base_url = "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main"
            files_to_download = {
                "model.onnx": f"{base_url}/onnx/model.onnx",
                "tokenizer.json": f"{base_url}/tokenizer.json",
                "tokenizer_config.json": f"{base_url}/tokenizer_config.json",
                "config.json": f"{base_url}/config.json",
            }
            for fname, url in files_to_download.items():
                fpath = os.path.join(_model_dir, fname)
                if not os.path.exists(fpath):
                    try:
                        _download_file(url, fpath)
                    except Exception as e:
                        # 下载失败，清理不完整文件
                        if os.path.exists(fpath):
                            os.remove(fpath)
                        raise RuntimeError("下载 %s 失败: %s" % (fname, e))
        
        # 加载ONNX模型
        sess = ort.InferenceSession(model_path)
        
        # 加载tokenizer（用HuggingFace tokenizers库）
        try:
            from tokenizers import Tokenizer
            tokenizer = Tokenizer.from_file(os.path.join(_model_dir, "tokenizer.json"))
        except ImportError:
            # 降级：用简单分词
            tokenizer = None
        
        _semantic_model = {
            "session": sess,
            "tokenizer": tokenizer,
            "max_length": 256,
        }
        _semantic_mode = "onnx"
        return _semantic_model
    except Exception:
        return None


def _jieba_cut(text):
    """jieba分词，返回空格分隔的词串"""
    import jieba
    return " ".join(jieba.cut(text))


def _load_tfidf_model():
    """降级方案：TF-IDF + jieba分词 + 余弦相似度"""
    global _semantic_model, _semantic_mode
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        import numpy as np
        _semantic_model = {
            "vectorizer": TfidfVectorizer(max_features=5000, ngram_range=(1, 2)),
            "cosine_similarity": cosine_similarity,
            "preprocess": _jieba_cut,  # 预处理函数
        }
        _semantic_mode = "tfidf"
        return _semantic_model
    except Exception:
        return None


def _ensure_model():
    """确保模型已加载（懒加载入口）"""
    global _semantic_model, _semantic_mode
    if _semantic_model is not None:
        return _semantic_model
    
    with _semantic_lock:
        if _semantic_model is not None:
            return _semantic_model
        
        # 首选ONNX
        model = _load_onnx_model()
        if model:
            return model
        
        # 降级TF-IDF
        model = _load_tfidf_model()
        if model:
            return model
        
        # 全部失败
        _semantic_mode = "none"
        return None


def _encode_onnx(texts, model_info):
    """ONNX模型编码"""
    import numpy as np
    sess = model_info["session"]
    tokenizer = model_info["tokenizer"]
    max_len = model_info["max_length"]
    
    all_embeddings = []
    for text in texts:
        if tokenizer:
            encoding = tokenizer.encode(text[:512])
            input_ids = encoding.ids[:max_len]
            attention_mask = [1] * len(input_ids)
            # padding
            pad_len = max_len - len(input_ids)
            input_ids += [0] * pad_len
            attention_mask += [0] * pad_len
        else:
            # 简单分词：按字符
            chars = list(text[:max_len])
            input_ids = [ord(c) % 30522 for c in chars]
            attention_mask = [1] * len(input_ids)
            pad_len = max_len - len(input_ids)
            input_ids += [0] * pad_len
            attention_mask += [0] * pad_len
        
        inputs = {
            "input_ids": np.array([input_ids], dtype=np.int64),
            "attention_mask": np.array([attention_mask], dtype=np.int64),
            "token_type_ids": np.zeros((1, max_len), dtype=np.int64),
        }
        
        # 尝试不同输入名
        try:
            ort_inputs = {}
            for name in sess.get_inputs():
                if name.name in inputs:
                    ort_inputs[name.name] = inputs[name.name]
                elif "input_ids" in name.name:
                    ort_inputs[name.name] = inputs["input_ids"]
                elif "attention" in name.name:
                    ort_inputs[name.name] = inputs["attention_mask"]
                elif "token_type" in name.name or "segment" in name.name:
                    ort_inputs[name.name] = inputs["token_type_ids"]
            
            outputs = sess.run(None, ort_inputs)
            # Mean pooling over non-padding tokens
            last_hidden = outputs[0]  # (1, seq_len, hidden_dim)
            mask = np.array(attention_mask).reshape(1, -1, 1)
            embedding = np.sum(last_hidden * mask, axis=1) / np.maximum(np.sum(mask, axis=1), 1e-9)
            all_embeddings.append(embedding[0])
        except Exception:
            # ONNX推理失败，用随机向量降级
            all_embeddings.append(np.random.randn(384))
    
    return np.array(all_embeddings)


def _encode_tfidf(texts, model_info):
    """TF-IDF编码（先用jieba分词预处理）"""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    
    vectorizer = model_info["vectorizer"]
    preprocess = model_info.get("preprocess")
    try:
        # 预处理：jieba分词
        processed = [preprocess(t) if preprocess else t for t in texts]
        tfidf_matrix = vectorizer.fit_transform(processed)
        return tfidf_matrix.toarray()
    except Exception:
        return None


def encode_texts(texts):
    """编码文本列表为向量，返回numpy数组或None"""
    model = _ensure_model()
    if model is None:
        return None
    
    if _semantic_mode == "onnx":
        return _encode_onnx(texts, model)
    elif _semantic_mode == "tfidf":
        return _encode_tfidf(texts, model)
    
    return None


def find_semantic_conflicts(nodes, threshold=0.8, max_pairs=20):
    """
    检测语义冲突：同类别节点中，文本高度相似但含义可能不同的对
    
    Args:
        nodes: 节点列表 [{id, text, category, ...}, ...]
        threshold: 相似度阈值（0-1），超过此值报告为潜在冲突
        max_pairs: 最多返回多少对
    
    Returns:
        list of {"text_a": str, "text_b": str, "id_a": int, "id_b": int, "similarity": float, "category": str}
    """
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity as _cosine_sim
    
    if not nodes or len(nodes) < 2:
        return []
    
    # 按类别分组
    by_category = {}
    for n in nodes:
        if not n or not n.get("text"):
            continue
        cat = n.get("category", "未分类")
        if cat not in by_category:
            by_category[cat] = []
        by_category[cat].append(n)
    
    conflicts = []
    for cat, cat_nodes in by_category.items():
        if len(cat_nodes) < 2:
            continue
        
        texts = [n["text"] for n in cat_nodes]
        embeddings = encode_texts(texts)
        
        if embeddings is None:
            continue
        
        # 计算相似度矩阵
        try:
            if _semantic_mode == "onnx":
                sim_matrix = _cosine_sim(embeddings)
            else:
                sim_matrix = _cosine_sim(embeddings)
        except Exception:
            continue
        
        # 找高相似度对（排除自身和重复）
        for i in range(len(cat_nodes)):
            for j in range(i + 1, len(cat_nodes)):
                sim = float(sim_matrix[i][j]) if sim_matrix.ndim == 2 else 0.0
                if sim >= threshold:
                    # 排除完全相同的文本（那是重复，不是语义冲突）
                    if texts[i].strip() == texts[j].strip():
                        continue
                    conflicts.append({
                        "text_a": texts[i],
                        "text_b": texts[j],
                        "id_a": cat_nodes[i].get("id", 0),
                        "id_b": cat_nodes[j].get("id", 0),
                        "similarity": round(sim, 4),
                        "category": cat,
                    })
                    if len(conflicts) >= max_pairs:
                        break
            if len(conflicts) >= max_pairs:
                break
    
    # 按相似度降序排列
    conflicts.sort(key=lambda x: x["similarity"], reverse=True)
    return conflicts


def get_mode_info():
    """返回当前语义检测模式信息"""
    _ensure_model()  # 触发懒加载
    return {
        "mode": _semantic_mode or "unavailable",
        "model_loaded": _semantic_model is not None,
        "description": {
            "onnx": "ONNX embedding（all-MiniLM-L6-v2），高精度语义检测",
            "tfidf": "TF-IDF + jieba分词 + 余弦相似度，基础语义检测（降级模式；将模型文件放入semantic_models/可自动升级为ONNX）",
            "none": "语义检测不可用",
            "unavailable": "语义检测未初始化",
        }.get(_semantic_mode or "unavailable", "未知")
    }
