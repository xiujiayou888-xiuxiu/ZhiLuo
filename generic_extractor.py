# -*- coding: utf-8 -*-
"""
知络 v8.0 通用提取器 — 不绑任何行业词典
纯基于 jieba 词性标注 + LLM，任何文本都能自动提取实体和关系
"""
import re, json
from datetime import datetime

try:
    import jieba
    import jieba.posseg as pseg
    _HAS_JIEBA = True
except ImportError:
    _HAS_JIEBA = False

# 动词 → 关系类型映射（纯语言层面，不绑行业）
_VERB_REL_MAP = {
    "需要": "depend_on", "依赖": "depend_on", "取决于": "depend_on",
    "包含": "part_of", "组成": "part_of", "由": "part_of",
    "属于": "belongs_to", "是": "belongs_to",
    "影响": "affect", "波及": "affect", "传导": "affect",
    "导致": "cause", "引起": "cause", "触发": "cause", "造成": "cause",
    "替代": "substitute", "代替": "substitute", "换成": "substitute", "替换": "substitute",
    "等于": "synonym", "也叫": "synonym", "即": "synonym",
    "关联": "related", "相关": "related", "涉及": "related",
}

# 事件/状态词（不入实体节点，转为边的属性）
_EVENT_WORDS = {
    "涨价", "降价", "上涨", "下跌", "波动", "断供", "缺货",
    "增长", "下降", "提升", "降低", "优化", "恶化",
    "延期", "提前", "完成", "失败", "崩溃", "宕机",
    "价格波动", "成本上升", "成本下降", "性能下降", "性能提升",
}

# 停用词（不提取为实体）
_STOP_ENTITIES = {
    "可以", "可能", "应该", "需要", "必须", "已经", "正在",
    "目前", "现在", "之前", "之后", "以后", "以上", "以下",
    "这个", "那个", "这些", "那些", "一个", "一种", "一些",
    "所有", "全部", "部分", "其中", "另外", "其他",
    "比如", "例如", "包括", "等等", "方面", "情况",
}


def _is_valid_entity(word):
    """判断是否为有效实体（名词性、非停用、非事件）"""
    if len(word) < 2:
        return False
    if word in _STOP_ENTITIES:
        return False
    if word in _EVENT_WORDS:
        return False
    if word.isdigit():
        return False
    if re.match(r'^\d+\.?\d*%?$', word):
        return False
    return True


def _is_event(word):
    """判断是否为事件/状态变化"""
    return word in _EVENT_WORDS or any(
        kw in word for kw in ["价格波动", "成本上升", "成本下降",
                              "性能下降", "性能提升", "响应时间增加"]
    )


def extract_entities(text):
    """
    通用实体提取 — 不依赖任何行业词典
    纯 jieba 词性标注，所有名词性词汇都是候选实体
    """
    entities = []
    seen = set()

    if _HAS_JIEBA:
        for word, flag in pseg.cut(text):
            w = word.strip()
            if w in seen:
                continue
            # 名词性：n/nr/ns/nt/nz/vn/ng + 长度>=2
            if flag in ("n", "nr", "ns", "nt", "nz", "vn", "ng", "eng"):
                if _is_valid_entity(w):
                    seen.add(w)
                    etype = _guess_type(w, flag)
                    entities.append({"name": w, "type": etype, "pos": flag})
            # 英文/数字混合词
            elif flag == "eng" and len(w) >= 2:
                if _is_valid_entity(w):
                    seen.add(w)
                    entities.append({"name": w, "type": "entity", "pos": flag})

    # jieba 不可用时：正则兜底
    if not entities:
        for m in re.finditer(r'[\u4e00-\u9fffA-Za-z][\u4e00-\u9fffA-Za-z0-9_]{1,15}', text):
            w = m.group()
            if w in seen:
                continue
            if _is_valid_entity(w):
                seen.add(w)
                entities.append({"name": w, "type": "entity", "pos": ""})

    return entities


def _guess_type(word, pos_flag=""):
    """根据词性猜测实体类型（通用，不绑行业）"""
    if pos_flag == "nr":
        return "person"
    if pos_flag == "ns":
        return "location"
    if pos_flag == "nt":
        return "organization"
    if pos_flag == "t" or re.match(r'\d', word):
        return "time"
    if any(kw in word for kw in ["项目", "系统", "平台", "模块", "组件", "服务"]):
        return "project"
    if any(kw in word for kw in ["成本", "价格", "预算", "利润", "收入", "费用"]):
        return "metric"
    if any(kw in word for kw in ["用户", "客户", "团队", "部门", "负责人"]):
        return "actor"
    return "entity"


def extract_relations(text):
    """
    通用关系提取 — 纯基于动词匹配 + jieba 定位前后实体
    不依赖行业词典
    """
    relations = []
    seen = set()

    def _add(src, tgt, rtype, weight=0.5):
        if not src or not tgt or src == tgt:
            return
        if _is_event(src) or _is_event(tgt):
            return
        key = (src, tgt, rtype)
        if key in seen:
            return
        seen.add(key)
        relations.append({
            "source": src, "target": tgt,
            "relation": rtype, "weight": weight,
            "confidence": 0.6,
        })

    # 1. 动词匹配建边
    for verb, rtype in _VERB_REL_MAP.items():
        if verb not in text:
            continue
        idx = text.find(verb)
        left_text = text[:idx]
        right_text = text[idx + len(verb):]

        left_ent = _nearest_noun(left_text, from_right=True)
        if left_ent:
            # 支持多实体：右边可能用逗号/和/与分隔
            right_parts = re.split(r'[和与、,，]', right_text)
            for part in right_parts:
                ent = _nearest_noun(part.strip())
                if ent:
                    _add(left_ent, ent, rtype, 0.6)

    # 2. 数字变化检测（"增加了30%""下降了20%"→触发事件）
    change_m = re.search(
        r'([\u4e00-\u9fffA-Za-z0-9]{1,15})\s*(?:增加|减少|提升|降低|上升|下降|增长)[了到]?\s*(\d+\.?\d*)\s*%?',
        text
    )
    if change_m:
        subject = change_m.group(1).strip()
        pct = float(change_m.group(2))
        if "降" in text or "减少" in text:
            pct = -pct
        # 找到受影响的目标
        right = text[change_m.end():]
        target = _nearest_noun(right) if right else None
        if target and not _is_event(subject):
            _add(subject, target, "affect", abs(pct) / 100.0)

    # 3. 日期/截止信息
    date_m = re.search(r'(?:截止|到期|交付|完成)[日期时间]?\s*(?:是|在|为)?\s*(.+?)(?:[，。\s]|$)', text)
    if date_m:
        date_text = date_m.group(1).strip()
        if len(date_text) >= 2:
            # 往前找关联实体
            before = text[:date_m.start()]
            ent = _nearest_noun(before, from_right=True)
            if ent:
                _add(ent, date_text, "depend_on", 0.5)

    return relations


def _nearest_noun(text, from_right=False):
    """找最近的实体（jieba 词性标注，纯语言层面）"""
    if not text or not text.strip():
        return None

    if _HAS_JIEBA:
        words = list(pseg.cut(text))
        if from_right:
            words = list(reversed(words))
        for word, flag in words:
            w = word.strip()
            if _is_valid_entity(w) and flag in ("n", "nr", "ns", "nt", "nz", "vn", "ng", "eng"):
                return w

    # 兜底：找最后/第一个连续中英文字符
    matches = re.findall(r'[\u4e00-\u9fffA-Za-z][\u4e00-\u9fffA-Za-z0-9_]{1,10}', text)
    if matches:
        return matches[-1] if from_right else matches[0]

    return None


def extract_all(text):
    """提取全部实体和关系"""
    return {
        "entities": extract_entities(text),
        "relations": extract_relations(text),
    }


def extract_with_llm(text, llm_func=None):
    """
    LLM 提取（主力，精度最高）
    不依赖任何行业提示词，纯通用
    """
    if llm_func is None:
        return extract_relations(text)

    prompt = f'''从以下文本中提取实体和关系。实体是名词性概念，关系是它们之间的业务逻辑联系。

返回 JSON（不要 markdown 代码块）：
{{"relations": [{{"source":"实体A","target":"实体B","relation":"part_of|depend_on|affect|cause|related|synonym|substitute|belongs_to","weight":0.0-1.0,"confidence":0.0-1.0}}]}}

文本: {text}'''

    try:
        result = llm_func(prompt)
        if isinstance(result, str):
            result = result.replace("```json", "").replace("```", "").strip()
            s, e = result.find("{"), result.rfind("}") + 1
            if s >= 0 and e > s:
                data = json.loads(result[s:e])
                rels = data.get("relations", data if isinstance(data, list) else [])
                return rels if isinstance(rels, list) else []
        return result if isinstance(result, list) else []
    except Exception:
        pass

    return extract_relations(text)
