# -*- coding: utf-8 -*-
"""
知络关系提取器 v7.0 — 全行业通用
核心：LLM 提取（主力）+ 规则提取（保底）+ 降噪管道
"""
import re, json, hashlib, time
try:
    import jieba; import jieba.posseg as pseg
    _HAS_JIEBA = True
except ImportError:
    _HAS_JIEBA = False

_VERB_CHARS = set("涨跌降需要包含由使用提供供应生产制造影响导致引起波及替代代替换成因为由于所以是的有的和与及了着过到给被把让使叫为以对从向在上下里外中内间")

# ── 动词 → 关系映射（全行业，无实体依赖）──
_REL_VERBS = {
    "需要": "part_of", "包含": "part_of", "由": "part_of", "由": "part_of",
    "用": "part_of", "使用": "part_of", "组成": "part_of",
    "提供": "related", "供应": "related", "生产": "related",
    "制造": "related", "采购": "related",
    "影响": "affect", "导致": "affect", "引起": "affect",
    "波及": "affect", "传导": "affect",
    "替代": "substitute", "代替": "substitute", "换成": "substitute",
    "替换": "substitute",
    "因为": "cause", "由于": "cause",
    "连锁反应":"affect",
    "提振":"affect",
    "挂钩":"related",
}

INDUSTRY_ENTITIES = {
    "餐饮": ["牛肉","猪肉","鸡肉","羊肉","鱼","虾","鸡蛋","番茄","土豆",
             "白菜","豆腐","面条","米饭","食用油","盐","糖","酱油",
             "牛肉面","番茄炒蛋","鱼香肉丝","宫保鸡丁","麻婆豆腐",
             "供应商","厨房","菜单","菜品","食材","调料","成本","售价",
             "牛腩","牛腱","五花肉","里脊","排骨","鸡腿","鸡胸","草鱼","鲈鱼","基围虾",
             "西兰花","生菜","菠菜","青椒","红椒","大蒜","生姜","葱","香菜",
             "早餐","午餐","晚餐","外卖","堂食","翻台率","客单价","毛利率"],
    "制造": ["钢材","铝材","铝锭","塑料","橡胶","玻璃","芯片","电池","电机",
             "汽车","手机","电脑","零件","部件","组件","工序","BOM",
             "生产线","工厂","报价","成本","良率","供应商",
             "原材料","半成品","成品","模具","夹具","铝","钢"],
    "建筑": ["水泥","钢筋","砂石","木材","涂料","管材","电缆",
             "地基","墙体","楼板","屋顶","装修","工期","人工",
             "预算","工程","施工","监理","材料","竣工","招标","投标"],
    "零售": ["SKU","品类","库存","进货","售价","成本","毛利",
             "促销","折扣","供应链","仓库","物流","配送","周转率","坪效"],
    "通用": ["成本","价格","利润","收入","支出","预算","报价",
             "供应商","客户","订单","合同","项目","时间","地点",
             "销量","增长率","现金流","毛利率","净利率","ROI",
             "运费","原料","材料","人工费","设备","损耗"],
}

_ALL_ENTITIES = set()
for _ents in INDUSTRY_ENTITIES.values():
    _ALL_ENTITIES.update(_ents)

_NOUN_SUFFIXES = {"成本","价格","利润","费用","材料","零件","产品",
                  "菜品","食材","报价","预算","工期","人工","库存","管理费","上升","下降"}

# ── 置信度评估 ──
SOURCE_WEIGHTS = {"百科":0.95,"标准":0.95,"招投标":0.85,"财报":0.90,
                  "词典":0.80,"jieba":0.60,"正则":0.40,"采集":0.50,"用户":0.90}
RELATION_BASE_CONFIDENCE = {"part_of":0.70,"cause":0.65,"affect":0.60,"related":0.50,
                            "synonym":0.75,"substitute":0.70}
DEFAULT_CONFIDENCE_THRESHOLD = 0.35
MIN_ENTITY_FREQ = 1  # 采集用

_entity_freq = {}
_jieba_initialized = False

def _init_jieba():
    global _jieba_initialized
    if _jieba_initialized or not _HAS_JIEBA: return
    for ent in _ALL_ENTITIES: jieba.add_word(ent)
    _jieba_initialized = True

def _has_verb_char(name):
    return any(c in _VERB_CHARS for c in name)

def _estimate_confidence(source_type, rtype, freq=1, has_cross_verify=False):
    base = SOURCE_WEIGHTS.get(source_type, 0.5)
    rel_base = RELATION_BASE_CONFIDENCE.get(rtype, 0.5)
    conf = (base + rel_base) / 2
    freq_bonus = min(0.2, freq * 0.05)
    conf += freq_bonus
    if has_cross_verify: conf += 0.15
    return min(max(conf, 0.0), 1.0)

def _update_freq(relations, source="采集"):
    for rel in relations:
        for name in [rel["source"], rel["target"]]:
            if name not in _entity_freq:
                _entity_freq[name] = {"count": 0, "sources": set()}
            _entity_freq[name]["count"] += 1
            _entity_freq[name]["sources"].add(source)

def add_industry_entities(industry, entities):
    if industry not in INDUSTRY_ENTITIES: INDUSTRY_ENTITIES[industry] = []
    existing = set(INDUSTRY_ENTITIES[industry])
    new = [e for e in entities if e not in existing]
    INDUSTRY_ENTITIES[industry].extend(new)
    _ALL_ENTITIES.update(new)
    if _HAS_JIEBA:
        for e in new: jieba.add_word(e)
    return len(new)

def get_entity_freq():
    return {k:v["count"] for k,v in _entity_freq.items()}

def reset_freq():
    _entity_freq.clear()

# ═══════════════════════════════════════════════
#  LLM 提取（主力）
# ═══════════════════════════════════════════════

def extract_with_llm(text, llm_func=None):
    """LLM 提取实体关系（主力，精度 80%+）
    当 llm_func 为 None 时回退到规则提取"""
    if llm_func is not None:
        prompt = f'''从以下文本中提取实体和关系。实体是名词性概念（人/物/组织/概念），关系是它们之间的业务逻辑联系。

返回 JSON (不要加 markdown 代码块)：
{{"relations": [{{"source":"实体A","target":"实体B","relation":"part_of|affect|cause|related|synonym|substitute","weight":0.0-1.0,"confidence":0.0-1.0}}]}}

关系类型:
- part_of: A是B的组成部分/A需要B
- affect: A影响B（单向影响）
- cause: A导致B（因果关系）
- related: A和B相关（双向弱关联）
- synonym: A和B是同义词/等价
- substitute: A可替代B（不同但功能相似）

文本: {text}'''
        try:
            result = llm_func(prompt)
            if isinstance(result, str):
                # 去掉 markdown 代码块
                result = result.replace("```json","").replace("```","").strip()
                s, e = result.find("{"), result.rfind("}")+1
                if s >= 0 and e > s:
                    data = json.loads(result[s:e])
                    rels = data.get("relations", data if isinstance(data, list) else [])
                    return rels if isinstance(rels, list) else []
            return result if isinstance(result, list) else []
        except:
            pass
    # 回退到规则提取
    return extract_relations(text)

# ═══════════════════════════════════════════════
#  规则提取（保底）
# ═══════════════════════════════════════════════

def extract_entities(text):
    entities, seen = [], set()
    # 1. 词典精确匹配
    for entity in sorted(_ALL_ENTITIES, key=len, reverse=True):
        if entity in text and entity not in seen:
            seen.add(entity)
            entities.append({"name": entity, "type": _guess_type(entity)})
    # 2. jieba 补充
    if _HAS_JIEBA:
        _init_jieba()
        for word, flag in pseg.cut(text):
            w = word.strip()
            if len(w) >= 2 and w not in seen:
                if flag in ("n","nr","ns","nt","nz","vn","ng"):
                    seen.add(w)
                    entities.append({"name": w, "type": _guess_type(w)})
                elif any(w.endswith(suf) for suf in _NOUN_SUFFIXES):
                    seen.add(w)
                    entities.append({"name": w, "type": _guess_type(w)})
    # 3. 正则辅助
    for m in re.finditer(r"[A-Za-z\u4e00-\u9fff]{2,8}[A-Z]?\d*", text):
        w = m.group()
        if _has_verb_char(w) and not any(w.endswith(suf) for suf in _NOUN_SUFFIXES): continue
        if len(w) >= 2 and w not in seen and not w.isdigit():
            seen.add(w)
            entities.append({"name": w, "type": "entity"})
    return entities

def _guess_type(name):
    if name in INDUSTRY_ENTITIES.get("餐饮", []):
        return "dish" if name in ("牛肉面","番茄炒蛋","鱼香肉丝","宫保鸡丁","麻婆豆腐","牛腩面","牛腱面") else "ingredient"
    if name in INDUSTRY_ENTITIES.get("制造", []):
        return "product" if name in ("汽车","手机","电脑") else "part"
    if name in INDUSTRY_ENTITIES.get("建筑", []): return "material"
    if name in INDUSTRY_ENTITIES.get("零售", []): return "sku"
    if name in ["供应商","客户","甲方","乙方"]: return "actor"
    if any(kw in name for kw in ["成本","价格","利润","报价","预算","毛利","净利率","ROI"]): return "metric"
    return "entity"

def _split_entities(text, keyword):
    idx = text.find(keyword)
    if idx < 0: return None, None
    left, right = text[:idx], text[idx+len(keyword):]
    return _nearest_entity(left, from_right=True), _nearest_entity(right, from_right=False)

def _nearest_entity(text, from_right=False):
    """找最近的实体（词典优先，jieba兜底）"""
    if not text: return None
    found = [(i, e) for e in sorted(_ALL_ENTITIES, key=len, reverse=True) for i in [text.find(e)] if i >= 0]
    if found:
        found.sort(key=lambda x: x[0], reverse=from_right)
        return found[0][1]
    if _HAS_JIEBA:
        _init_jieba()
        words = list(pseg.cut(text))
        if from_right: words = list(reversed(words))
        for word, flag in words:
            w = word.strip()
            if len(w) >= 2 and flag in ("n","nr","ns","nt","nz","vn","ng") and not _has_verb_char(w):
                return w
    return None

def extract_relations(text):
    relations, seen = [], set()
    _EVENT_MARKERS = {"价格波动","成本上升","成本下降","价格下降"}

    def _add(source, target, rtype, weight=0.5):
        if not source or not target or source == target: return
        if source in _EVENT_MARKERS or target in _EVENT_MARKERS: return
        key = (source, target, rtype)
        if key in seen: return
        seen.add(key)
        relations.append({"source": source, "target": target, "relation": rtype, "weight": weight})

    last_left_ent = None

    for verb, rtype in sorted(_REL_VERBS.items(), key=lambda x: -len(x[0])):
        if verb not in text: continue
        idx_v = 0
        while idx_v < len(text):
            idx = text.find(verb, idx_v)
            if idx < 0: break
            prefix_before = text[max(0,idx-8):idx]
            if verb in ("制造","生产") and any(kw in prefix_before for kw in ["导致","影响","引起","使得","使"]):
                idx_v = idx + 1
                continue
            left_text, right_text = text[:idx], text[idx+len(verb):]
            ye_pos = left_text.rfind("也")
            if ye_pos >= 0:
                after_ye = left_text[ye_pos+1:].strip()
                import re as _re
                if _re.match(r"^(可以|能|会|要|就可以|就能|就能够)?$", after_ye):
                    left_ent = last_left_ent
                else:
                    left_ent = _nearest_entity(left_text, from_right=True)
                    if left_ent: last_left_ent = left_ent
            else:
                left_ent = _nearest_entity(left_text, from_right=True)
                if left_ent: last_left_ent = left_ent
            if not left_ent:
                idx_v = idx + 1
                continue
            right_parts = [p.strip() for p in __import__("re").split(r"[和与、,，]", right_text) if p.strip()]
            for part in right_parts:
                if not part: continue
                ent = _nearest_entity(part)
                if ent: _add(left_ent, ent, rtype, _estimate_weight(rtype))
            idx_v = idx + 1

    cost_m = __import__("re").search(r"([\u4e00-\u9fffA-Za-z]{2,10})\s*(?:\u4ef7\u683c)?(?:\u6da8\u4ef7?|\u8dcc|\u964d\u4ef7?)[\u4e86\u4ef7]?\s*(\d+\.?\d*)\s*%?", text)
    if cost_m:
        entity = cost_m.group(1)
        while entity and len(entity) > 1 and entity[-1] in _VERB_CHARS:
            entity = entity[:-1]
        if entity and len(entity) >= 1 and entity not in _EVENT_MARKERS:
            after_text = text[cost_m.end():]
            target_ent = _nearest_entity(after_text)
            if target_ent and target_ent not in _EVENT_MARKERS:
                pct = float(cost_m.group(2))/100.0 if float(cost_m.group(2)) > 1 else float(cost_m.group(2))
                _add(entity, target_ent, "affect", pct)

    affect_m = __import__("re").search(r"(?:\u5f71\u54cd|\u6ce2\u53ca|\u4f20\u5bfc\u5230?|\u5bfc\u81f4|\u62d6\u7d2f|\u5229\u597d|\u8fde\u9501\u53cd\u5e94|\u632f\u6296)\s*(.+)", text)
    if affect_m:
        affected_text = affect_m.group(1)
        cause_text = text[:affect_m.start()].strip()
        cause_ent = _nearest_entity(cause_text.replace("导致","").replace("影响","").strip(), from_right=True)
        if not cause_ent: cause_ent = last_left_ent
        if cause_ent and cause_ent not in _EVENT_MARKERS:
            for ent in sorted(_ALL_ENTITIES, key=len, reverse=True):
                if ent in affected_text and ent not in _EVENT_MARKERS:
                    _add(cause_ent, ent, "affect", 0.7)

    return relations

def _estimate_weight(rtype):
    return {"part_of": 0.5, "cause": 0.9, "affect": 0.7, "related": 0.4, "synonym": 1.0, "substitute": 0.85}.get(rtype, 0.4)

def extract_all(text):
    return {"entities": extract_entities(text), "relations": extract_relations(text)}

# ── 降噪管道 ──
def confidence_filter(relations, source_type="采集", threshold=DEFAULT_CONFIDENCE_THRESHOLD, freq_map=None):
    if not relations:
        return [], {"low_confidence": 0, "low_freq": 0}
    filtered, dropped = [], {"low_confidence": 0, "low_freq": 0}
    for rel in relations:
        freq = 1
        if freq_map:
            freq = max(freq_map.get(rel["source"], 0), freq_map.get(rel["target"], 0))
            if freq < MIN_ENTITY_FREQ: dropped["low_freq"] += 1; continue
        conf = _estimate_confidence(source_type, rel.get("relation","related"), freq)
        rel["confidence"] = round(conf, 3)
        if conf >= threshold: filtered.append(rel)
        else: dropped["low_confidence"] += 1
    return filtered, dropped

def extract_filtered(text, source_type="采集", threshold=DEFAULT_CONFIDENCE_THRESHOLD):
    raw = extract_all(text)
    _update_freq(raw["relations"], source_type)
    filtered, dropped = confidence_filter(raw["relations"], source_type=source_type, threshold=threshold,
                                          freq_map={k:v["count"] for k,v in _entity_freq.items()})
    return {"entities": raw["entities"], "relations": filtered, "dropped": dropped,
            "raw_count": len(raw["relations"]), "filtered_count": len(filtered)}

__all__ = [
    "extract_entities", "extract_relations", "extract_all", "extract_with_llm",
    "extract_filtered", "confidence_filter", "add_industry_entities",
    "INDUSTRY_ENTITIES", "get_entity_freq", "reset_freq",
    "DEFAULT_CONFIDENCE_THRESHOLD",
]
