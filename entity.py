# -*- coding: utf-8 -*-
"""
知络 v5.3 实体抽取模块
增强：更多模式 + jieba 词性标注（可选）
"""
import re

PATTERNS = {
    "person": [
        r"(?<=负责人[是为])\w+",
        r"(?<=[由让找给请派安排委托指定])\w+(?=负责)",
        r"\w+(?=主管)",
        r"\w+(?=经理)",
        r"(?<=联系)\w+",
        r"(?<=[叫叫小老大小大])\w{2,3}(?=[说讲做干])",
        r"[\u4e00-\u9fff]{2,4}(?=老师|师傅|总|哥|姐)",
    ],
    "project": [
        r"项目[\w\u4e00-\u9fff()（）]+",
        r"[\w\u4e00-\u9fff()（）]+项目",
        r"[\w\u4e00-\u9fff]+系统",
        r"[\w\u4e00-\u9fff]+平台",
    ],
    "date": [
        r"\d+年\d+月\d+日",
        r"\d+月\d+日",
        r"\d{4}-\d{2}-\d{2}",
        r"\d+号",
        r"(下|本|这)?周[一二三四五六日天]",
        r"(明|后|大后)天",
    ],
    "amount": [
        r"\d+\.?\d*(?:万|亿)?元",
        r"\d+\.?\d*(?:万|亿)?块钱",
        r"\d+\.?\d*%?",
    ],
    "location": [
        r"[\u4e00-\u9fff]{2,6}(?=市|省|区|县|路|街|号)",
        r"(?<=在)[\u4e00-\u9fff]{2,6}",
    ],
    "url": [
        r"https?://[^\s]+",
        r"www\.[^\s]+",
    ],
    "email": [
        r"[\w.+-]+@[\w-]+\.[\w.-]+",
    ],
    "phone": [
        r"1[3-9]\d{9}",
        r"\d{3}-\d{8}",
        r"\d{4}-\d{7}",
    ],
}

def extract(text):
    ents = {}
    for etype, patterns in PATTERNS.items():
        for pat in patterns:
            for m in re.finditer(pat, text):
                val = m.group(0).strip()
                if len(val) >= 2:
                    ents[f"{etype}:{val}"] = {"type": etype, "name": val}
    # jieba 词性标注增强（可选）
    try:
        import jieba.posseg as pseg
        for word, flag in pseg.cut(text):
            if flag in ("nr", "ns", "nt", "nz") and len(word) >= 2:
                etype = {"nr": "person", "ns": "location", "nt": "org", "nz": "other"}.get(flag, "entity")
                key = f"{etype}:{word}"
                if key not in ents:
                    ents[key] = {"type": etype, "name": word}
    except ImportError:
        pass
    return ents

def find_in_nodes(store, query):
    results = []
    seen = set()
    for key, nids in getattr(store, "entity_idx", {}).items():
        if query in key or query in key.split(":", 1)[-1]:
            for nid in nids:
                if nid not in seen:
                    n = store.get(nid)
                    if n:
                        seen.add(nid)
                        results.append(n)
    if not results:
        from engine import find as _find
        results = _find(store, query)
    return results[:10]

def graph(store, name):
    results = []
    for key, nids in getattr(store, "entity_idx", {}).items():
        if name in key or name in key.split(":", 1)[-1]:
            for nid in nids:
                n = store.get(nid)
                if n:
                    results.append({"entity": key, "text": n["text"][:60], "category": n["category"]})
    return results[:20]
