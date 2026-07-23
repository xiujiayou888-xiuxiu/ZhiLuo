# -*- coding: utf-8 -*-
"""
知络 输入去风格化预处理 — 角色混淆防御
来源：Prompt Injection as Role Confusion (2026)
原理：LLM靠文本风格判断角色归属，去风格化可将注入攻击成功率从61%降到10%
设计：纯增量模块，不影响原有learn/query流程
"""
import re


# 伪系统指令的关键词模式（不区分大小写）
_SYSTEM_PATTERNS = [
    # 仿system prompt风格
    re.compile(r'(?i)\b(policy\s+(states?|says?|requires?|allows?))\b'),
    re.compile(r'(?i)\b(rule\s*\d*\s*[:：])'),
    re.compile(r'(?i)(system\s+(instruction|prompt|message|note))\s*[:：]'),
    re.compile(r'(?i)\b(you\s+are\s+(now|acting\s+as))\b'),
    re.compile(r'(?i)\b(ignore|disregard)\b.{0,20}\b(instructions?|rules?|prompts?|directives?)\b'),
    re.compile(r'(?i)\b(disregard)\b.{0,20}\b(training|instructions?|rules?)\b'),
    re.compile(r'(?i)\b(override|bypass|disable)\b.{0,20}\b(safety|security|content|filter)\b'),
    # 仿thinking标签风格
    re.compile(r'(?i)\b(the\s+user\s+requests?|user\s+is\s+asking)\b.*\b(allowed|permitted|approved)\b'),
    # 中文伪系统指令模式
    re.compile(r'(忽略|无视|忘记|跳过).{0,6}(指令|规则|设定|提示|限制|要求|约束)'),
    re.compile(r'(你现在|从现在起|从现在开始|请扮演|假装你是|角色扮演|扮演).{0,10}(AI|助手|模型|角色|身份|不限)'),
    re.compile(r'(系统提示|系统指令|系统设定|系统消息|系统规则)\s*[:：]'),
    re.compile(r'(规则|法则)\s*\d*\s*[:：]'),
    re.compile(r'(覆盖|解除|关闭|绕过|跳过|突破).{0,4}(安全|内容|审核|过滤|限制|约束)'),
    re.compile(r'(不要遵守|不用遵守|不必遵守|无需遵守).{0,4}(指令|规则|设定|限制|约束)'),
    re.compile(r'(DAN模式|越狱模式|开发者模式|无限制模式|黑客模式|不设限模式)'),
]

# 去除的标签模式
_TAG_PATTERNS = [
    re.compile(r'</?(system|think|assistant|internal|scratchpad)[^>]*>', re.IGNORECASE),
]

# 可疑程度分级
_RISK_THRESHOLD_LOW = 1    # 0-1: 低风险，正常通过
_RISK_THRESHOLD_HIGH = 3  # 3+: 高风险，需要警告


def sanitize_input(text: str) -> dict:
    """
    去风格化预处理：检测并标记可能的注入攻击
    
    Returns:
        {
            "text": 清洗后的文本,
            "original": 原始文本,
            "risk_score": 风险分数(0-5+),
            "risk_level": "low"/"medium"/"high",
            "flags": [检测到的可疑模式列表],
            "cleaned": 是否进行了清洗
        }
    """
    if not text or not isinstance(text, str):
        return {"text": text or "", "original": text or "", "risk_score": 0,
                "risk_level": "low", "flags": [], "cleaned": False}
    
    original = text
    flags = []
    risk_score = 0
    cleaned = False
    result = text
    
    # 1. 检测并移除标签
    for pattern in _TAG_PATTERNS:
        matches = pattern.findall(result)
        if matches:
            flags.append("伪标签: %s" % ", ".join(set(matches)))
            risk_score += 2
            result = pattern.sub('', result)
            cleaned = True
    
    # 2. 检测伪系统指令模式（高危+2分，普通+1分）
    # 高危pattern：中文全部+系统提示/忽略指令/角色扮演/DAN模式
    _HIGH_RISK_KEYWORDS = ('忽略', '无视', '忘记', '跳过', '不要遵守', 'DAN模式', '越狱',
                           '系统提示', '系统指令', '系统设定', '系统消息', '系统规则',
                           '请扮演', '假装你是', '角色扮演', '绕过', '突破',
                           'ignore', 'disregard', 'system', 'override', 'bypass')
    for pattern in _SYSTEM_PATTERNS:
        matches = pattern.findall(result)
        if matches:
            flags.append("伪指令: %s" % pattern.pattern[:40])
            is_high_risk = any(kw in pattern.pattern.lower() for kw in _HIGH_RISK_KEYWORDS)
            risk_score += 2 if is_high_risk else 1
    
    # 3. 去风格化：将可疑文本改写为更中性的形式
    # 如果检测到伪系统指令，将冒号后的内容加上引号（使其看起来像引用而非指令）
    if risk_score >= 2:
        # 将 "Policy states: xxx" 这类改为 "（引用内容）xxx"
        result = re.sub(
            r'(?i)(policy\s+(states?|says?|requires?|allows?))\s*[:：]\s*',
            r'【系统引用检测】',
            result
        )
        result = re.sub(
            r'(?i)(rule\s*\d*)\s*[:：]\s*',
            r'【规则引用检测】',
            result
        )
        cleaned = True
    
    # 4. 判断风险等级
    if risk_score <= _RISK_THRESHOLD_LOW:
        risk_level = "low"
    elif risk_score <= _RISK_THRESHOLD_HIGH:
        risk_level = "medium"
    else:
        risk_level = "high"
    
    return {
        "text": result,
        "original": original,
        "risk_score": risk_score,
        "risk_level": risk_level,
        "flags": flags,
        "cleaned": cleaned,
    }


def is_suspicious(text: str) -> bool:
    """快速判断输入是否可疑（供learn()调用）"""
    result = sanitize_input(text)
    return result["risk_score"] >= 2


def get_sanitized_text(text: str) -> str:
    """获取清洗后的文本（供learn()调用，静默模式）"""
    result = sanitize_input(text)
    return result["text"]
