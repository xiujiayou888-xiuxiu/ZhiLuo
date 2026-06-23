# -*- coding: utf-8 -*-
"""用户画像蒸馏 v1.0 — 踩坑防重复 + 用户画像"""
from collections import Counter, defaultdict

def pitfall_tracker(store, workspace=None):
    """踩坑防重复: 从变更历史提取纠错模式，识别高频踩坑点"""
    from engine import SimHash
    nodes = store.valid(workspace)
    if not nodes:
        return {"pitfalls": [], "patterns": []}
    
    logs = store.change_history()
    cc = Counter()
    for log in logs:
        txt = str(log)
        if "correct" in txt or "纠正" in txt:
            for t in SimHash.tokens(txt):
                if len(t) > 1:
                    cc[t] += 1
    
    pp = []
    for word, count in cc.most_common(15):
        if count >= 2:
            level = "高危" if count >= 3 else "关注"
            pp.append({"keyword": word, "correct_count": count, "level": level})
    
    return {"pitfalls": pp, "patterns": pp}


def user_profile(store, workspace=None):
    """用户画像蒸馏: 自动观察分析用户行为特征，零配置"""
    from engine import SimHash
    nodes = store.valid(workspace)
    if not nodes:
        return {"profile": {}}
    
    p = {}
    wf = Counter()
    for n in nodes:
        for t in SimHash.tokens(n.get("text", "")):
            if len(t) > 1 and t not in SimHash._STOP:
                wf[t] += 1
    
    # 1. 高频词
    p["高频词"] = [w for w, c in wf.most_common(15)]
    
    # 2. 领域分布
    df = Counter(n.get("category", "未分类") for n in nodes)
    total = len(nodes)
    p["领域分布"] = {k: {"count": v, "ratio": round(v/total*100, 1)}
                     for k, v in df.most_common()}
    
    # 3. 学习风格
    lc = sum(1 for n in nodes if n.get("source") == "learn")
    qc = sum(1 for n in nodes if n.get("last_accessed_at") is not None)
    if lc > qc * 1.5:
        p["学习风格"] = "输入型(偏好主动学习)"
    elif qc > lc * 1.5:
        p["学习风格"] = "检索型(偏好查询已有知识)"
    else:
        p["学习风格"] = "平衡型"
    p["知识总量"] = total
    
    # 4. 活跃时段
    hf = defaultdict(int)
    for n in nodes:
        t = n.get("learned_at", "")
        if len(t) >= 13:
            try:
                hf[int(t[11:13])] += 1
            except ValueError:
                pass
    if hf:
        ph = max(hf, key=hf.get)
        period = "凌晨" if ph < 6 else ("上午" if ph < 12 else ("下午" if ph < 18 else "晚上"))
        p["活跃时段"] = f"{period}({ph}时)"
    
    # 5. 知识深度（关联复杂度）
    ec = sum(len(n.get("edges", [])) for n in nodes)
    p["知识深度"] = min(10, round(ec / max(total, 1) * 2, 1))
    
    # 6. 格式偏好
    al = sum(len(n.get("text", "")) for n in nodes) / max(total, 1)
    p["格式偏好"] = "详细记录型" if al > 100 else ("标准记录型" if al > 30 else "简洁笔记型")
    
    # 7. 情感语气
    pos = sum(1 for w, _ in wf.most_common(30) if w in "好优秀成功增长提高喜欢推荐")
    neg = sum(1 for w, _ in wf.most_common(30) if w in "差失败下降问题错误不好难")
    if pos > neg:
        p["情感语气"] = "偏积极"
    elif neg > pos:
        p["情感语气"] = "偏消极/问题导向"
    else:
        p["情感语气"] = "中性理性"
    
    return {"profile": p}
