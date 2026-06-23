# -*- coding: utf-8 -*-
"""深度推理引擎 v1.0 — 问题拆解 / 量子关联 / 神经扩散"""
import re
from collections import Counter

def deep_reason(store, question, workspace=None):
    """深度推理流水线: 拆解→检索→分析→打分→输出"""
    from engine import SimHash, find, diffuse, causal, conflicts
    
    result = {"question": question, "sub_questions": [], "evidence": [],
              "angles": [], "score": 0.0, "verdict": "", "structured": ""}
    
    # 1. 问题拆解
    tokens = SimHash.tokens(question)
    key_words = list(dict.fromkeys([t for t in tokens if len(t) > 1 and t not in SimHash._STOP]))
    causal_words = {"导致","影响","因为","所以","为什么","如何","怎么"}
    has_causal = bool(key_words & causal_words) if isinstance(key_words, set) else any(c in question for c in causal_words)
    
    sub_questions = [f"{question} 的关联知识检索"]
    if has_causal and len(key_words) >= 2:
        sub_questions.insert(0, f"{question} 的原因分析")
        sub_questions.insert(1, f"{question} 的影响分析")
    if len(key_words) >= 3:
        sub_questions.append(f"{question} 中 {'、'.join(key_words[:3])} 的关联分析")
    result["sub_questions"] = sub_questions
    
    # 2. 知识检索（双层: FTS5 + 扩散）
    all_evidence, seen = [], set()
    for kw in key_words:
        for n in find(store, kw, workspace):
            txt = n.get("text", "")
            if txt and txt not in seen:
                seen.add(txt)
                all_evidence.append({"id": n["id"], "text": txt,
                    "category": n.get("category","未分类"),
                    "confidence": n.get("confidence", 1.0)})
    for n in diffuse(store, question, workspace, hops=2):
        txt = n.get("text", "")
        if txt and txt not in seen:
            seen.add(txt)
            all_evidence.append({"id": n["id"], "text": txt,
                "category": n.get("category","未分类"),
                "confidence": n.get("confidence", 1.0) * n.get("_c", 0.5)})
    all_evidence.sort(key=lambda x: -x["confidence"])
    result["evidence"] = all_evidence[:20]
    
    # 3. 多角度分析
    angles, combined = [], " ".join(e["text"] for e in all_evidence[:10])
    
    # 事实分析
    a1, nums = [], re.findall(r"-?[0-9]+[.]?[0-9]*%?", question + " " + combined)
    up = sum(1 for w in ("增长","上升","提高","增加","涨","升") if w in combined)
    down = sum(1 for w in ("下降","降低","减少","跌","降") if w in combined)
    if nums: a1.append(f"涉及 {len(nums)} 个数字/百分比")
    if up or down:
        tr = "上升" if up > down else ("下降" if down > up else "平稳")
        a1.append(f"趋势: {tr} (上升{up}条, 下降{down}条)")
    a1.append(f"知识库命中 {len(all_evidence)} 条相关记录")
    if a1:
        angles.append({"name":"事实分析","points":a1,"confidence":min(0.9, 0.5+len(all_evidence)*0.05)})
    
    # 关联分析
    if len(key_words) >= 2:
        co = Counter()
        for e in all_evidence:
            m = [k for k in key_words if k in e["text"]]
            if len(m) >= 2:
                co[tuple(sorted(m[:2]))] += 1
        pts = []
        if co:
            bp = co.most_common(1)[0]
            pts.append(f"关键词 {bp[0][0]} 与 {bp[0][1]} 在 {bp[1]} 条记录中关联出现")
        pts.append(f"共 {len(all_evidence)} 条证据")
        angles.append({"name":"关联分析","points":pts,"confidence":0.5 if co else 0.3})
    
    # 因果/矛盾分析
    if has_causal:
        cc = causal(store, key_words[0] if key_words else question, workspace)
        pts = []
        if cc.get("chain"):
            pts.append(f"发现因果链: {len(cc['chain'])} 环")
            for i, c in enumerate(cc["chain"][:3]):
                pts.append(f"  环{i+1}: {c['text'][:60]}")
        pts.append("问题包含因果逻辑")
        if not cc.get("chain"):
            pts.append("知识库中未发现完整因果链")
        angles.append({"name":"因果分析","points":pts,"confidence":0.7 if cc.get("chain") else 0.3})
    else:
        cf = conflicts(store) if hasattr(store, 'conflicts') else {"conflicts": []}
        if cf.get("conflicts"):
            pts = [f"检测到 {cf['total']} 组冲突知识"]
            for c in cf["conflicts"][:3]:
                pts.append(f"  {c['node'][:30]} vs {c['vs'][:30]}")
            angles.append({"name":"矛盾分析","points":pts,"confidence":0.6})
    result["angles"] = angles
    
    # 4. 评估打分
    ac = sum(x["confidence"] for x in angles) / max(len(angles), 1) if angles else 0
    cv = min(1.0, len(all_evidence) / max(len(key_words), 1) * 0.2) if all_evidence else 0
    score = round(ac * 0.6 + cv * 0.4, 2)
    result["score"] = score
    result["verdict"] = "可信度较高" if score >= 0.7 else ("部分可信" if score >= 0.4 else "证据不足")
    
    # 5. 结构化输出
    lines = [f"## 深度推理报告: {question[:60]}",
             f"**可信度评分: {score:.2f}** | 结论: {result['verdict']}", ""]
    lines.append("### 1. 问题拆解")
    for i, sq in enumerate(sub_questions, 1):
        lines.append(f"  {i}. {sq}")
    lines.append(""); lines.append(f"### 2. 知识检索 ({len(all_evidence)}条)")
    for e in all_evidence[:5]:
        lines.append(f"  - [{e['category']}] {e['text'][:80]} (置信度:{e['confidence']:.2f})")
    if len(all_evidence) > 5:
        lines.append(f"  ... 还有 {len(all_evidence)-5} 条")
    lines.append(""); lines.append("### 3. 多角度分析")
    for x in angles:
        lines.append(f"**{x['name']}** (置信度: {x['confidence']:.2f}):")
        for p in x["points"][:5]:
            lines.append(f"  - {p}")
        lines.append("")
    lines.append("### 4. 综合评估")
    lines.append(f"  评分: {score:.2f} / 1.0")
    lines.append(f"  结论: {result['verdict']}")
    if score < 0.4:
        lines.append("  建议: 补充更多相关数据后重新分析")
    elif score < 0.7:
        lines.append("  建议: 部分结论需人工验证")
    else:
        lines.append("  建议: 结论可信，可作为决策参考")
    result["structured"] = "\n".join(lines)
    return result


def quantum_assoc(store, words, workspace=None, top_k=15):
    """量子级知识关联: 加权纠缠场 + 多跳叠加传播"""
    wl = words.strip().split() if isinstance(words, str) else words
    if not wl:
        return {"nodes": [], "edges": [], "associations": []}
    
    from engine import find, SimHash
    seeds = find(store, wl[0], workspace)
    if not seeds:
        return {"nodes": [], "edges": [], "associations": []}
    
    resonance, visited, decay = {}, set(), 0.6
    for n in seeds:
        nid = n["id"]
        resonance[nid] = resonance.get(nid, 0) + 1.0
        visited.add(nid)
    
    # 第一跳传播
    for nid in list(resonance.keys()):
        for e in store.adj.get(nid, []):
            tid = e["target"]
            w = resonance[nid] * decay * e.get("weight", 1.0)
            resonance[tid] = max(resonance.get(tid, 0), w)
            visited.add(tid)
    
    # 第二跳传播（叠加效应）
    for nid in list(resonance.keys()):
        if nid in {n["id"] for n in seeds}:
            continue
        for e in store.adj.get(nid, []):
            tid = e["target"]
            if tid in visited:
                continue
            w = resonance[nid] * decay * decay * e.get("weight", 1.0)
            resonance[tid] = resonance.get(tid, 0) + w * 0.3
            visited.add(tid)
    
    # 多词协同增强
    if len(wl) >= 2:
        for w2 in wl[1:]:
            nids2 = {n["id"] for n in find(store, w2, workspace)}
            for nid in nids2:
                if nid in resonance:
                    resonance[nid] *= 1.5
    
    # 排序输出
    anodes = []
    for nid, w in sorted(resonance.items(), key=lambda x: -x[1])[:top_k]:
        n = store.get(nid)
        if n:
            anodes.append({"id": nid, "text": n["text"][:60],
                          "category": n["category"], "weight": round(w, 4)})
    
    # 关联边
    idset = {a["id"] for a in anodes}
    edges = []
    for a in anodes[:10]:
        for e in store.adj.get(a["id"], [])[:5]:
            if e["target"] in idset:
                edges.append({"from": a["id"], "to": e["target"],
                             "type": e.get("type", "related")})
    return {"nodes": anodes, "edges": edges[:20], "associations": anodes}


def neural_diffuse(store, question, workspace=None, depth=3, threshold=0.15, top_k=20):
    """神经网络式扩散: 多跳衰减 + 阈值截断的语义激活传播"""
    from engine import SimHash, find
    kw = list(dict.fromkeys([t for t in SimHash.tokens(question)
                              if len(t) > 1 and t not in SimHash._STOP]))
    seeds = []
    for k in kw:
        seeds.extend(find(store, k, workspace))
    if not seeds:
        return {"activated": [], "propagation": [], "total_activated": 0}
    
    activation, queue = {}, []
    for n in seeds:
        activation[n["id"]] = 1.0
        queue.append((n["id"], 0, 1.0))
    
    processed, df = set(), 0.5
    while queue:
        nid, layer, strength = queue.pop(0)
        if nid in processed or layer >= depth:
            continue
        processed.add(nid)
        for e in store.adj.get(nid, []):
            tid, w = e["target"], e.get("weight", 1.0)
            v = strength * w * (df ** (layer + 1))
            if v >= threshold:
                activation[tid] = max(activation.get(tid, 0), v)
                queue.append((tid, layer + 1, v))
    
    activated = []
    for nid, v in sorted(activation.items(), key=lambda x: -x[1])[:top_k]:
        n = store.get(nid)
        if n:
            activated.append({"id": nid, "text": n["text"][:60],
                            "category": n["category"], "activation": round(v, 4)})
    
    prop = [f"节点 #{x['id']}: {x['text'][:40]} (激活值:{x['activation']:.3f})"
            for x in activated[:10]]
    return {"activated": activated, "propagation": prop, "total_activated": len(activated)}
