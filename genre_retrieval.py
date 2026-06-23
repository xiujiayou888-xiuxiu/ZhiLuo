# -*- coding: utf-8 -*-
"""
知络 v8.0 体裁感知检索模块 — GenreRetrieval
============================================
设计理念：不同体裁的文本，检索策略应该不同。
- 流程类文本 → 按步骤顺序召回，权重向步骤词倾斜
- 论证类文本 → 按因果链召回，权重向连接词倾斜
- 数据类文本 → 按指标召回，权重向数字/百分比倾斜
- 对话类文本 → 按口语关键词召回，降低精确匹配要求
- 定义类文本 → 精确匹配优先，权重向"是指""即"附近倾斜

实现方式：纯规则驱动，零LLM调用，零token消耗。
与左脑的设计理念一致（体裁感知路由），但实现路径完全不同：
- 左脑用正则模式匹配 + 骨架提取
- 知络用特征向量打分 + 策略权重映射

不侵权声明：本模块为独立原创实现，基于通用NLP体裁分类方法论设计。
"""

import re
import math
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field


# ═══════════════════════════════════════════════════════════
#  体裁定义
# ═══════════════════════════════════════════════════════════

@dataclass
class GenreProfile:
    """体裁画像 — 定义一种体裁的检索偏好"""
    name: str
    # 关键词权重倍增器（这类词在检索时加权）
    boost_keywords: List[str] = field(default_factory=list)
    # 权重衰减词（这类词在检索时降权）
    dampen_keywords: List[str] = field(default_factory=list)
    # 最小匹配阈值（低于此分数直接跳过）
    min_score_threshold: float = 0.15
    # 优先匹配模式：exact=精确, fuzzy=模糊, keyword=关键词
    match_mode: str = "fuzzy"
    # 是否启用子串匹配（对话类通常需要）
    substring_match: bool = False
    # 数字敏感度：0=忽略, 1=正常, 2=高度敏感
    number_sensitivity: int = 1


# 体裁特征定义（基于通用NLP方法论，非任何产品的特征库）
GENRES = {
    "process": GenreProfile(
        name="流程/步骤",
        boost_keywords=["步骤", "流程", "方法", "操作", "配置", "安装", "部署",
                       "先", "然后", "接着", "最后", "第一步", "第二步"],
        dampen_keywords=["的", "了", "是", "在", "有"],
        min_score_threshold=0.12,
        match_mode="keyword",
        substring_match=True,
    ),
    "argument": GenreProfile(
        name="论证/分析",
        boost_keywords=["因为", "所以", "因此", "导致", "原因", "影响",
                       "如果", "那么", "不仅", "而且", "但是", "然而"],
        dampen_keywords=["嗯", "哦", "哈哈"],
        min_score_threshold=0.10,
        match_mode="fuzzy",
    ),
    "definition": GenreProfile(
        name="定义/说明",
        boost_keywords=["是指", "指的是", "即", "意为", "定义", "概念",
                       "包括", "包含", "分为", "分类", "属于"],
        dampen_keywords=[],
        min_score_threshold=0.20,
        match_mode="exact",
    ),
    "data_summary": GenreProfile(
        name="数据/统计",
        boost_keywords=["增长", "下降", "提升", "减少", "占比", "达到",
                       "超过", "低于", "同比", "环比", "平均"],
        dampen_keywords=[],
        min_score_threshold=0.15,
        match_mode="fuzzy",
        number_sensitivity=2,
    ),
    "dialogue": GenreProfile(
        name="对话/交流",
        boost_keywords=["你", "我", "他", "她", "我们", "他们",
                       "好的", "是的", "对吧", "嗯", "哦"],
        dampen_keywords=[],
        min_score_threshold=0.05,
        match_mode="fuzzy",
        substring_match=True,
        number_sensitivity=0,
    ),
    "essay": GenreProfile(
        name="叙述/随笔",
        boost_keywords=["记得", "有一次", "那时候", "当时", "后来",
                       "终于", "从此", "感动", "难忘"],
        dampen_keywords=[],
        min_score_threshold=0.10,
        match_mode="fuzzy",
        substring_match=True,
    ),
    "knowledge": GenreProfile(
        name="知识/通用",
        boost_keywords=["知识", "概念", "原理", "机制", "作用", "功能"],
        dampen_keywords=[],
        min_score_threshold=0.12,
        match_mode="keyword",
    ),

    # ═══════════════════════════════════════════
    #  PARA 知识分类法则
    #  P=Projects  A=Areas  R=Resources  A=Archives
    #  参见：Tiago Forte《Building a Second Brain》
    # ═══════════════════════════════════════════
    "project": GenreProfile(
        name="P-项目",
        boost_keywords=["项目", "计划", "目标", "里程碑", "截止", "交付",
                       "进度", "任务", "待办", "排期", "冲刺", "迭代",
                       "验收", "上线", "发布"],
        dampen_keywords=[],
        min_score_threshold=0.10,
        match_mode="keyword",
        substring_match=True,
    ),
    "area": GenreProfile(
        name="A-领域",
        boost_keywords=["领域", "持续", "维护", "健康", "财务", "成长",
                       "技能", "习惯", "长期", "责任", "标准", "底线",
                       "每年", "每月", "定期"],
        dampen_keywords=[],
        min_score_threshold=0.10,
        match_mode="fuzzy",
    ),
    "resource": GenreProfile(
        name="R-资源",
        boost_keywords=["参考", "工具", "模板", "教程", "资料", "素材",
                       "链接", "文档", "灵感", "收藏", "备查", "引用",
                       "方法库", "知识库"],
        dampen_keywords=[],
        min_score_threshold=0.10,
        match_mode="keyword",
        substring_match=True,
    ),
    "archive": GenreProfile(
        name="A-归档",
        boost_keywords=["完成", "过期", "旧版", "历史", "废弃", "不再",
                       "已结束", "已取消", "归档", "备份", "存档",
                       "v1", "v2", "旧方案"],
        dampen_keywords=[],
        min_score_threshold=0.15,
        match_mode="exact",
    ),
}


# ═══════════════════════════════════════════════════════════
#  体裁分类器
# ═══════════════════════════════════════════════════════════

class GenreClassifier:
    """
    基于特征向量打分的体裁分类器

    与左脑的 GenreClassifier 设计理念一致（体裁感知），但实现路径不同：
    - 左脑用正则模式列表逐条匹配计数
    - 知络用特征向量（词汇密度、句式结构、标点分布、数字密度等）打分
    """

    @staticmethod
    def _lexical_density(text: str) -> float:
        """实词密度（名词+动词+形容词占比）"""
        if not text:
            return 0.0
        # 实词：长度≥2的中文词 + 英文词
        real_words = len(re.findall(r'[\u4e00-\u9fff]{2,}|[a-zA-Z]{3,}', text))
        total = len(text)
        return min(1.0, real_words / max(total * 0.1, 1))

    @staticmethod
    def _number_density(text: str) -> float:
        """数字密度"""
        if not text:
            return 0.0
        nums = len(re.findall(r'\d+\.?\d*%?', text))
        return min(1.0, nums / max(len(text) * 0.02, 1))

    @staticmethod
    def _sentence_complexity(text: str) -> float:
        """句式复杂度（平均句长/标点密度）"""
        if not text:
            return 0.0
        sentences = re.split(r'[。！？\n]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        if not sentences:
            return 0.0
        avg_len = sum(len(s) for s in sentences) / len(sentences)
        return min(1.0, avg_len / 80)

    @staticmethod
    def _dialogue_score(text: str) -> float:
        """对话特征分"""
        score = 0.0
        # 引号密度
        quotes = len(re.findall(r'["""''「」『』]', text))
        score += min(0.3, quotes * 0.05)
        # 人称代词密度
        pronouns = len(re.findall(r'[你我他她它我们你们他们她们]', text))
        score += min(0.4, pronouns * 0.02)
        # 短句比例
        sentences = re.split(r'[。！？\n]+', text)
        short = sum(1 for s in sentences if len(s.strip()) < 15)
        if sentences:
            score += min(0.3, short / len(sentences) * 0.5)
        return score

    @staticmethod
    def _connective_density(text: str) -> float:
        """连接词密度（论证类特征）"""
        connectives = re.findall(
            r'(因为|所以|因此|从而|导致|由于|如果|那么|'
            r'不仅|而且|但是|然而|虽然|尽管|总之|综上所述)',
            text
        )
        return min(1.0, len(connectives) * 0.05)

    # 关键词快速路由：高置信特征词直接判定体裁
    # 每条 pattern 组独立匹配，1 命中即触发（避免短文本漏判）
    _QUICK_ROUTES = [
        # (genre, patterns, min_hits, base_conf)
        ("code", [
            r'\b(def |class |import |from |return |print|lambda |async |await )',
            r'[{};]\s*$',
            r'(function|const |let |var |=>|\.then\(|\.catch\()',
        ], 1, 0.80),
        ("process", [
            r'第[一二三四五六七八九十\d]+[步个环节阶段]',
            r'(先|首先|第一步|然后|接着|再|最后|其次|随后)',
            r'(步骤|流程|方法|操作|指南|教程|配置|安装|部署)',
        ], 1, 0.70),
        ("argument", [
            r'(因为|所以|因此|从而|导致|由于|原因在于)',
            r'(如果.{0,8}就|只有.{0,8}才|不仅.{0,8}而且)',
            r'(但是|然而|不过|虽然|尽管|总之|综上所述)',
        ], 1, 0.72),
        ("definition", [
            r'(是指|指的是|即|意为|所谓|定义|概念)',
            r'(分为|包括|包含|由.{0,8}组成)',
        ], 1, 0.75),
        ("data_summary", [
            r'\d{2,}\s*[万亿千百%倍个次人元件条只]',
            r'(增长|下降|提升|减少|占比|达到|超过|低于|同比|环比)',
        ], 2, 0.78),  # 至少2命中：数字+趋势词同时出现才算数据类
        ("dialogue", [
            r'[「『""][^「『""]{2,40}[」』""]',
            r'^(你|我|他|她|它|我们|你们)[\u4e00-\u9fff]',
            r'(你好|哈哈|嗯|好的|是的|对啊|哦|好吧|谢谢|早安|晚安|没事|加油)',
        ], 1, 0.68),
        ("essay", [
            r'(我记得|有一次|那时候|小时候|第一次|感动|难忘)',
            r'(开始|后来|最后|终于|从此)',
        ], 1, 0.70),
        ("knowledge", [
            r'(知识|原理|机制|架构|体系|框架|模型|算法)',
            r'(核心|关键|基础|底层|本质)',
        ], 1, 0.65),
        # PARA 快速路由
        ("project", [
            r'(项目|计划|目标|里程碑|截止|交付|排期|冲刺|迭代)',
            r'(进度|待办|验收|上线|发布|倒计时|deadline)',
        ], 1, 0.78),
        ("area", [
            r'(领域|持续|维护|长期责任|每年|每月|定期|标准)',
            r'(健康|财务|成长|技能|习惯|底线|OKR)',
        ], 1, 0.72),
        ("resource", [
            r'(参考|工具|模板|教程|资料|素材|链接|文档)',
            r'(灵感|收藏|备查|引用|方法库|知识库|checklist)',
        ], 1, 0.72),
        ("archive", [
            r'(已完成|已过期|已废弃|已结束|已取消|不再使用|不再维护)',
            r'(归档|存档|旧版|旧方案|v\d+替换|历史版本|备份存档)',
        ], 1, 0.85),
    ]

    @classmethod
    def _quick_classify(cls, text: str) -> Optional[Tuple[str, float]]:
        """快速路由：高置信关键词直接判定体裁"""
        best_genre = None
        best_conf = 0
        for genre, patterns, min_hits, base_conf in cls._QUICK_ROUTES:
            hits = 0
            for pat in patterns:
                if re.search(pat, text):
                    hits += 1
            if hits >= min_hits:
                conf = min(0.90, base_conf + hits * 0.08)
                if conf > best_conf:
                    best_conf = conf
                    best_genre = genre
        if best_genre:
            return (best_genre, best_conf)
        return None

    @classmethod
    def classify(cls, text: str) -> Tuple[str, float]:
        """
        分类文本体裁

        策略：先快速路由（高置信关键词）→ 不行再走特征向量打分

        Returns:
            (genre_key, confidence) — 体裁键 + 置信度(0~1)
        """
        if not text or len(text) < 8:
            return ("dialogue", 0.5)

        # 第一层：快速路由
        quick = cls._quick_classify(text)
        if quick:
            return quick

        # 第二层：特征向量
        features = {
            "lexical": cls._lexical_density(text),
            "numbers": cls._number_density(text),
            "complexity": cls._sentence_complexity(text),
            "dialogue": cls._dialogue_score(text),
            "connective": cls._connective_density(text),
        }

        scores = {}

        scores["data_summary"] = (
            features["numbers"] * 0.50 +
            features["lexical"] * 0.25 +
            (1 - features["dialogue"]) * 0.25
        )

        scores["process"] = (
            features["lexical"] * 0.30 +
            features["complexity"] * 0.25 +
            features["connective"] * 0.25 +
            (1 - features["numbers"]) * 0.20
        )

        scores["argument"] = (
            features["connective"] * 0.45 +
            features["complexity"] * 0.30 +
            features["lexical"] * 0.25
        )

        scores["dialogue"] = (
            features["dialogue"] * 0.60 +
            (1 - features["complexity"]) * 0.25 +
            (1 - features["lexical"]) * 0.15
        )

        scores["definition"] = (
            features["lexical"] * 0.40 +
            (1 - features["connective"]) * 0.30 +
            (1 - features["complexity"]) * 0.20 +
            (1 - features["dialogue"]) * 0.10
        )

        scores["essay"] = (
            features["dialogue"] * 0.30 +
            features["complexity"] * 0.35 +
            features["lexical"] * 0.35
        )

        scores["knowledge"] = (
            features["lexical"] * 0.40 +
            features["complexity"] * 0.35 +
            (1 - features["dialogue"]) * 0.25
        )

        primary = max(scores, key=scores.get)
        confidence = scores[primary]

        if confidence < 0.15:
            return ("knowledge", 0.3)

        return (primary, round(confidence, 3))

    @classmethod
    def get_profile(cls, genre: str) -> GenreProfile:
        """获取体裁画像"""
        return GENRES.get(genre, GENRES["knowledge"])


# ═══════════════════════════════════════════════════════════
#  体裁感知检索器
# ═══════════════════════════════════════════════════════════

class GenreAwareRetriever:
    """
    体裁感知检索器 — 根据输入文本的体裁调整检索策略

    核心逻辑：
    1. 先对 query 做体裁分类
    2. 根据体裁画像调整检索参数（匹配模式、权重偏置、阈值）
    3. 使用调整后的参数执行检索
    """

    def __init__(self, fast_index=None):
        """
        Args:
            fast_index: FastIndex 实例（知络的微秒级检索索引）
        """
        self.idx = fast_index

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        genre_hint: Optional[str] = None,
    ) -> dict:
        """
        体裁感知检索

        Args:
            query: 查询文本
            top_k: 返回数量
            genre_hint: 手动指定体裁（可选，不传则自动分类）

        Returns:
            {
                "genre": str,           # 检测到的体裁
                "confidence": float,    # 体裁置信度
                "strategy": dict,       # 使用的检索策略
                "results": [...]        # 检索结果
            }
        """
        # 1. 体裁分类
        if genre_hint and genre_hint in GENRES:
            genre = genre_hint
            confidence = 0.8
        else:
            genre, confidence = GenreClassifier.classify(query)

        profile = GENRES.get(genre, GENRES["knowledge"])

        # 2. 确定检索策略
        strategy = {
            "match_mode": profile.match_mode,
            "min_score": profile.min_score_threshold,
            "number_boost": profile.number_sensitivity,
            "substring": profile.substring_match,
        }

        # 3. 执行检索（根据 match_mode 路由到 FastIndex 的不同模式）
        # FastIndex 支持的 mode: exact / keyword / fts / auto
        # 体裁 match_mode → FastIndex mode 映射
        _MODE_MAP = {"exact": "exact", "fuzzy": "keyword", "keyword": "keyword"}
        idx_mode = _MODE_MAP.get(profile.match_mode, "keyword")

        if self.idx is None:
            return {
                "genre": genre,
                "confidence": confidence,
                "strategy": strategy,
                "results": [],
                "warning": "FastIndex 未注入，跳过检索"
            }

        results = self.idx.search(
            query,
            mode=idx_mode,
            top_k=top_k * 2  # 先多取，后面再做体裁加权
        )

        # 4. 体裁加权：根据关键词偏置调整分数
        weighted = self._apply_genre_weights(
            results.get("results", []),
            profile,
            top_k
        )

        return {
            "genre": genre,
            "confidence": confidence,
            "strategy": strategy,
            "results": weighted,
        }

    def _apply_genre_weights(
        self,
        results: List,
        profile: GenreProfile,
        top_k: int,
    ) -> List[dict]:
        """对检索结果按体裁偏置调整权重"""
        weighted = []
        for item in results:
            # FastIndex 返回 (node_id, score) 元组，统一转为 dict
            if isinstance(item, (list, tuple)):
                nid, score = item[0], item[1] if len(item) > 1 else 0.5
                entry = {"node_id": nid, "name": str(nid), "score": score}
            else:
                nid = item.get("node_id") or item.get("id")
                score = item.get("score", 0.5)
                entry = dict(item)

            name = entry.get("name", "") or entry.get("text", "")

            # 关键词加权
            for kw in profile.boost_keywords:
                if kw in name:
                    score *= 1.3
                    break

            # 衰减词降权
            for kw in profile.dampen_keywords:
                if kw in name:
                    score *= 0.7
                    break

            # 数字敏感度调整
            if profile.number_sensitivity >= 2:
                num_count = len(re.findall(r'\d+', name))
                if num_count > 0:
                    score *= 1.0 + num_count * 0.1
            elif profile.number_sensitivity == 0:
                # 对话类：数字不重要
                if re.search(r'\d{3,}', name):
                    score *= 0.5

            # 阈值过滤
            if score >= profile.min_score_threshold:
                entry["adjusted_score"] = round(score, 4)
                weighted.append(entry)

        # 按调整后分数排序
        weighted.sort(key=lambda x: x.get("adjusted_score", 0), reverse=True)
        return weighted[:top_k]


# ═══════════════════════════════════════════════════════════
#  骨架提取器（独立于左脑的实现）
# ═══════════════════════════════════════════════════════════

class SkeletonExtractor:
    """
    文本骨架提取器 — 从长文本中提取结构化关键点

    设计理念与左脑的 extract_skeleton 一致，但实现方式不同：
    - 左脑用正则逐体裁匹配 + 手动拼接
    - 知络用统一的标题/列表/段首提取 + 体裁后处理

    适用场景：learn 时自动提取摘要、search 时辅助结果排序
    """

    @staticmethod
    def extract(text: str, genre: Optional[str] = None) -> dict:
        """
        提取文本骨架

        Returns:
            {
                "type": str,       # 体裁类型
                "headings": [...],  # 标题/主题词
                "key_points": [...],# 关键点
                "numbers": [...],   # 关键数字
                "summary": str,     # 一句话摘要
            }
        """
        if not text:
            return {"type": "unknown", "headings": [], "key_points": [],
                    "numbers": [], "summary": ""}

        # 体裁分类
        if genre is None:
            genre, _ = GenreClassifier.classify(text)

        # 提取标题
        headings = []
        for line in text.split('\n'):
            line = line.strip()
            # Markdown 标题
            if re.match(r'^#{1,4}\s', line):
                headings.append(re.sub(r'^#+\s*', '', line)[:60])
            # 数字编号标题
            elif re.match(r'^[\d一二三四五六七八九十]+[、.．）\)]\s*\S', line):
                headings.append(re.sub(r'^[\d一二三四五六七八九十]+[、.．）\)]\s*', '', line)[:60])

        # 提取关键点（每个自然段的首句）
        key_points = []
        paragraphs = re.split(r'\n\s*\n', text)
        for para in paragraphs[:20]:
            para = para.strip()
            if len(para) > 20:
                # 取前两句
                sentences = re.split(r'[。！？；]', para, maxsplit=2)
                first = sentences[0].strip()
                if len(first) > 8:
                    key_points.append(first[:120])

        # 提取关键数字
        numbers = []
        num_matches = re.findall(r'(\d+\.?\d*\s*[万亿千百%倍个次人元件条只]+)', text)
        seen = set()
        for n in num_matches:
            n = n.strip()
            if n not in seen and len(n) < 30:
                seen.add(n)
                numbers.append(n)

        # 一句话摘要（取前100字核心内容）
        clean = re.sub(r'\s+', ' ', text).strip()
        summary = clean[:150]
        if len(clean) > 150:
            summary += "..."

        return {
            "type": genre,
            "headings": headings[:10],
            "key_points": key_points[:8],
            "numbers": numbers[:10],
            "summary": summary,
        }


# ═══════════════════════════════════════════════════════════
#  便捷接口
# ═══════════════════════════════════════════════════════════

def classify_genre(text: str) -> Tuple[str, float]:
    """便捷接口：分类文本体裁"""
    return GenreClassifier.classify(text)


def genre_retrieve(query: str, fast_index=None, top_k: int = 10) -> dict:
    """便捷接口：体裁感知检索"""
    retriever = GenreAwareRetriever(fast_index)
    return retriever.retrieve(query, top_k=top_k)


def extract_skeleton(text: str) -> dict:
    """便捷接口：提取文本骨架"""
    return SkeletonExtractor.extract(text)
