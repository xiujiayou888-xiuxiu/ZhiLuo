# -*- coding: utf-8 -*-
"""自动浮现图谱 v1.0 — 从知识库中发现高频关联模式"""
from collections import Counter

def auto_emerge(store, workspace=None, top_k=10):
    """自动浮现图谱: 发现高频共现词对+提炼浮现概念"""
    from engine import SimHash
    nodes = store.valid(workspace)
    if not nodes:
        return {"clusters": [], "emergent": []}
    
    # 1. 词频统计
    wf, nw = Counter(), {}
    for n in nodes:
        valid = [t for t in SimHash.tokens(n.get("text", ""))
                 if len(t) > 1 and t not in SimHash._STOP and not t.isdigit()]
        valid = list(dict.fromkeys(valid))
        nw[n["id"]] = valid
        for w in valid:
            wf[w] += 1
    
    # 2. 共现分析
    co = Counter()
    for _, words in nw.items():
        if len(words) >= 2:
            for i in range(len(words)):
                for j in range(i+1, len(words)):
                    co[tuple(sorted([words[i], words[j]]))] += 1
    
    # 3. 强关联集群
    min_pair = max(2, len(nodes) * 0.1)
    sp = {k: v for k, v in co.items() if v >= min_pair}
    clusters, used = [], set()
    for pair, freq in sorted(sp.items(), key=lambda x: -x[1])[:top_k]:
        if pair[0] not in used or pair[1] not in used:
            clusters.append({"words": list(pair), "frequency": freq,
                            "strength": round(freq / len(nodes), 2)})
            used.add(pair[0]); used.add(pair[1])
    
    # 4. 浮现概念（连接度/出现比高的词）
    wd = Counter()
    for pair, freq in co.items():
        wd[pair[0]] += freq
        wd[pair[1]] += freq
    
    emergent = []
    for w, deg in wd.most_common(top_k * 2):
        freq = wf.get(w, 1)
        score = round(deg / freq, 2)
        if deg >= min_pair * 2 and score >= 1.0:
            emergent.append({"concept": w, "frequency": freq,
                            "connectivity": deg, "score": score})
    emergent.sort(key=lambda x: -x["score"])
    
    return {"clusters": clusters[:top_k], "emergent": emergent[:top_k]}
