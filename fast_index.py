# -*- coding: utf-8 -*-
"""
知络 v8.0 微秒级检索层 — FastIndex
===================================
三级索引架构 + SimHash去重 + MinHashLSH，零LLM调用，微秒级响应。

移植自 v6.0 开源版检索骨架，适配 v8.0 BrainEngine。

索引层级:
  Hash索引 (O(1), 纳秒级):  entity_name → node_id 精确映射
  Keyword索引 (微秒级):     jieba分词 token → [node_ids]
  FTS5全文索引 (微秒级):    SQLite内存数据库+FTS5 trigram分词

去重管线:
  SimHash → 桶查找 → 汉明距离≤2 → 合并
  MinHashLSH → 候选集 → Jaccard>0.4 → 合并

零Token设计: 所有检索操作纯本地计算，不调LLM。
"""

import re
import json
import os
import hashlib
import sqlite3
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Set

# ── jieba: 可选依赖 ──
try:
    import jieba
    import jieba.analyse
    _HAS_JIEBA = True
except ImportError:
    _HAS_JIEBA = False


# ═══════════════════════════════════════════════════════════
#  分词工具
# ═══════════════════════════════════════════════════════════

_STOP_WORDS = set("的了是在我有和就不人都一个上也这到说他她你们这那之")

# ═══════════════════════════════════════════════════════════
#  零成本文本归一化（纯本地，不调LLM）
# ═══════════════════════════════════════════════════════════

# 中文数字 → 阿拉伯数字
_CN_NUM = {
    "零": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4,
    "五": 5, "六": 6, "七": 7, "八": 8, "九": 9,
    "十": 10, "百": 100, "千": 1000, "万": 10000, "亿": 100000000,
}
_CN_PCT_WORDS = {"百分之", "百分之"}


def _cn_num_to_arabic(text: str) -> str:
    """中文数字 → 阿拉伯数字（零token纯规则）"""
    # "百分之二十" → "20%"
    text = re.sub(r'百分之([一二三四五六七八九]+)十([一二三四五六七八九]?)',
                  lambda m: f"{_CN_NUM[m.group(1)] * 10 + _CN_NUM.get(m.group(2), 0)}%", text)
    text = re.sub(r'百分之([一二三四五六七八九]+)',
                  lambda m: f"{_CN_NUM[m.group(1)]}%", text)
    # "十" 单独
    text = re.sub(r'百分之十(?!\d)', "10%", text)
    return text


def _normalize_text(text: str) -> str:
    """
    零成本文本归一化（纯规则，不调LLM）
    - 中文数字 → 阿拉伯数字
    - 常见同义动词归一
    - 去语气虚词
    """
    if not text:
        return ""

    # 1. 中文数字 → 阿拉伯
    text = _cn_num_to_arabic(text)

    # 2. 同义动词归一（只做高频稳定映射）
    _SYN_MAP = [
        (r'价格上涨', '涨价'),
        (r'价格下跌', '降价'),
        (r'成本上升', '成本上涨'),
        (r'成本下降', '成本降低'),
        (r'跟着涨', '涨价'),
        (r'跟着降', '降价'),
    ]
    for pat, rep in _SYN_MAP:
        text = re.sub(pat, rep, text)

    # 3. 去语气虚词
    _FILLER = r'(?:了|的|地|得|着|过|吧|呢|吗|啊|嘛|哦|嗯|哈|啦|呀)'
    text = re.sub(_FILLER, '', text)

    # 4. 统一标点
    text = text.replace('，', ',').replace('。', '.').replace('、', ',')

    # 5. 去连接词（SimHash对连接词太敏感，去掉后核心词更聚焦）
    _CONNECTIVES = r'(?:会导致|会|导致|使得|跟着|造成|引起|可能|将会|一定会|可以使|能|应该|需要|必须)'
    text = re.sub(_CONNECTIVES, '', text)

    return text


def smart_tokenize(text: str) -> List[str]:
    """智能分词: 归一化 → jieba优先 → 回退正则"""
    text = _normalize_text(text.lower())


    if _HAS_JIEBA:
        words = list(jieba.cut(text))
        return [w.strip() for w in words
                if w.strip() and w.strip() not in _STOP_WORDS]
    else:
        out = []
        for t in re.findall(r'[\w\u4e00-\u9fff]+', text):
            if re.match(r'^[\u4e00-\u9fff]+$', t):
                out.extend(list(t))  # 中文逐字
            else:
                out.append(t)
        return [t for t in out if t not in _STOP_WORDS]


def smart_extract_keywords(text: str, top_k: int = 10) -> List[str]:
    """提取关键词"""
    if _HAS_JIEBA:
        return jieba.analyse.extract_tags(text, topK=top_k)
    else:
        toks = smart_tokenize(text)
        freq = Counter(toks)
        return [w for w, _ in freq.most_common(top_k)]


# ═══════════════════════════════════════════════════════════
#  SimHash — 64位去重签名
# ═══════════════════════════════════════════════════════════

class SimHash:
    """64位 SimHash，用于快速近似去重"""

    @staticmethod
    def tokens(text: str) -> List[str]:
        return smart_tokenize(text)

    @staticmethod
    def hash(text: str) -> int:
        """计算64位SimHash"""
        toks = SimHash.tokens(text)
        if not toks:
            return 0
        v = [0] * 64
        for tok in toks:
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16) & 0x7FFFFFFFFFFFFFFF
            for i in range(64):
                if h >> i & 1:
                    v[i] += 1
                else:
                    v[i] -= 1
        out = 0
        for i in range(64):
            if v[i] > 0:
                out |= 1 << i
        return out & 0x7FFFFFFFFFFFFFFF

    @staticmethod
    def hamming(a: int, b: int) -> int:
        """汉明距离"""
        return bin(a ^ b).count("1")

    @staticmethod
    def bucket(sh: int) -> int:
        """取低4位作为桶号 (0-15)"""
        return sh & 15


# ═══════════════════════════════════════════════════════════
#  MinHashLSH — 集合相似度索引
# ═══════════════════════════════════════════════════════════

class MinHashLSH:
    """MinHash + LSH 用于集合相似度候选召回"""

    _NUM_PERM = 128
    _BANDS = 32
    _ROWS = 4

    def __init__(self):
        self._mersenne = (1 << 61) - 1
        self._a = [random.randint(1, self._mersenne) for _ in range(self._NUM_PERM)]
        self._b = [random.randint(0, self._mersenne) for _ in range(self._NUM_PERM)]
        self._bands: Dict[int, dict] = defaultdict(dict)

    def _signature(self, tokens: List[str]) -> List[int]:
        """计算MinHash签名"""
        if not tokens:
            return [0] * self._NUM_PERM
        shingles = set(tokens)
        sig = [self._mersenne] * self._NUM_PERM
        for s in shingles:
            h = int(hashlib.md5(s.encode()).hexdigest(), 16) % self._mersenne
            for i in range(self._NUM_PERM):
                ph = (self._a[i] * h + self._b[i]) % self._mersenne
                if ph < sig[i]:
                    sig[i] = ph
        return sig

    def _band_keys(self, sig: List[int]) -> List[Tuple[int, int]]:
        """计算LSH分桶键"""
        keys = []
        for b in range(self._BANDS):
            start = b * self._ROWS
            chunk = tuple(sig[start:start + self._ROWS])
            keys.append((b, hash(chunk)))
        return keys

    def insert(self, node_id: int, text: str):
        """插入节点到LSH索引"""
        tokens = SimHash.tokens(text)
        sig = self._signature(tokens)
        for b, h in self._band_keys(sig):
            self._bands[b].setdefault(h, []).append(node_id)

    def query(self, text: str) -> List[int]:
        """查询候选相似节点"""
        tokens = SimHash.tokens(text)
        sig = self._signature(tokens)
        candidates: Set[int] = set()
        for b, h in self._band_keys(sig):
            for nid in self._bands.get(b, {}).get(h, []):
                candidates.add(nid)
        return list(candidates)

    @staticmethod
    def jaccard(text_a: str, text_b: str) -> float:
        """计算Jaccard相似度"""
        sa = set(SimHash.tokens(text_a))
        sb = set(SimHash.tokens(text_b))
        if not sa and not sb:
            return 1.0
        return len(sa & sb) / max(len(sa | sb), 1)

    def stats(self) -> dict:
        total = sum(len(v) for b in self._bands.values() for v in b.values())
        return {"bands": len(self._bands), "entries": total}


# ═══════════════════════════════════════════════════════════
#  FastIndex — 三级索引 + 去重
# ═══════════════════════════════════════════════════════════

class FastIndex:
    """
    微秒级检索索引

    三级索引:
      - hash_idx:   entity_name → node_id  (O(1) 纳秒级)
      - kw_idx:     token → [node_ids]     (微秒级)
      - fts5:       SQLite内存FTS5          (微秒级)

    去重:
      - sh_buckets: SimHash桶 → [node_ids]
      - lsh:        MinHashLSH
      - node_texts: node_id → text 缓存
    """

    def __init__(self, workspace: str = "global", data_dir: Optional[str] = None):
        self.workspace = workspace

        # ── 数据目录 ──
        if data_dir:
            self._data_dir = Path(data_dir)
        else:
            self._data_dir = Path(__file__).resolve().parent / "data" / "fast_index"
        self._data_dir.mkdir(parents=True, exist_ok=True)

        # ── 三级索引 ──
        self.hash_idx: Dict[str, int] = {}        # name → node_id
        self.kw_idx: Dict[str, List[int]] = defaultdict(list)  # token → [node_ids]
        self.node_texts: Dict[int, str] = {}       # node_id → text
        self.node_names: Dict[int, str] = {}       # node_id → name
        self.node_types: Dict[int, str] = {}       # node_id → entity_type

        # ── SimHash 去重 ──
        self.sh_buckets: Dict[int, List[int]] = defaultdict(list)  # bucket → [node_ids]
        self.node_simhash: Dict[int, int] = {}     # node_id → simhash

        # ── MinHashLSH ──
        self.lsh = MinHashLSH()

        # ── FTS5 (内存数据库) ──
        self._conn = sqlite3.connect(":memory:", check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_fts()

        # ── 统计 ──
        self._add_count = 0
        self._remove_count = 0

    def _init_fts(self):
        """初始化FTS5虚拟表"""
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS idx_nodes (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                text TEXT NOT NULL,
                entity_type TEXT DEFAULT 'entity'
            );
            CREATE VIRTUAL TABLE IF NOT EXISTS idx_nodes_fts USING fts5(
                name, text, entity_type,
                content='idx_nodes', content_rowid='id',
                tokenize='trigram'
            );
            CREATE TRIGGER IF NOT EXISTS idx_nodes_ai AFTER INSERT ON idx_nodes BEGIN
                INSERT INTO idx_nodes_fts(rowid, name, text, entity_type)
                VALUES (new.id, new.name, new.text, new.entity_type);
            END;
            CREATE TRIGGER IF NOT EXISTS idx_nodes_ad AFTER DELETE ON idx_nodes BEGIN
                INSERT INTO idx_nodes_fts(idx_nodes_fts, rowid, name, text, entity_type)
                VALUES ('delete', old.id, old.name, old.text, old.entity_type);
            END;
            CREATE TRIGGER IF NOT EXISTS idx_nodes_au AFTER UPDATE ON idx_nodes BEGIN
                INSERT INTO idx_nodes_fts(idx_nodes_fts, rowid, name, text, entity_type)
                VALUES ('delete', old.id, old.name, old.text, old.entity_type);
                INSERT INTO idx_nodes_fts(rowid, name, text, entity_type)
                VALUES (new.id, new.name, new.text, new.entity_type);
            END;
        """)
        self._conn.commit()

    # ── CRUD ────────────────────────────────────────────────

    def add(self, node_id: int, name: str, text: str, entity_type: str = "entity"):
        """
        注册节点到所有索引: hash / keyword / FTS5 / SimHash / LSH

        Args:
            node_id:    GraphEngine中的节点ID
            name:       实体名称
            text:       节点文本内容
            entity_type: 实体类型
        """
        # 先移除旧索引（如果存在）
        self.remove(node_id)

        # ── Hash索引 ──
        self.hash_idx[name] = node_id
        self.node_names[node_id] = name
        self.node_texts[node_id] = text
        self.node_types[node_id] = entity_type

        # ── Keyword索引 ──
        tokens = SimHash.tokens(text)
        for tok in tokens:
            self.kw_idx[tok].append(node_id)
        # 同时索引名称
        name_tokens = SimHash.tokens(name)
        for tok in name_tokens:
            if tok not in tokens:
                self.kw_idx[tok].append(node_id)

        # ── FTS5 ──
        self._conn.execute(
            "INSERT INTO idx_nodes (id, name, text, entity_type) VALUES (?, ?, ?, ?)",
            (node_id, name, text, entity_type)
        )
        self._conn.commit()

        # ── SimHash去重 ──
        sh = SimHash.hash(text)
        bk = SimHash.bucket(sh)
        self.node_simhash[node_id] = sh
        self.sh_buckets[bk].append(node_id)

        # ── MinHashLSH ──
        self.lsh.insert(node_id, text)

        self._add_count += 1

    def remove(self, node_id: int):
        """从所有索引中移除节点"""
        name = self.node_names.pop(node_id, None)
        text = self.node_texts.pop(node_id, None)
        self.node_types.pop(node_id, None)

        # Hash索引
        if name:
            self.hash_idx.pop(name, None)

        # Keyword索引
        if text:
            tokens = SimHash.tokens(text)
            for tok in tokens:
                try:
                    self.kw_idx[tok].remove(node_id)
                except ValueError:
                    pass
        if name:
            name_tokens = SimHash.tokens(name)
            for tok in name_tokens:
                try:
                    self.kw_idx[tok].remove(node_id)
                except ValueError:
                    pass

        # FTS5
        self._conn.execute("DELETE FROM idx_nodes WHERE id=?", (node_id,))
        self._conn.commit()

        # SimHash
        sh = self.node_simhash.pop(node_id, None)
        if sh is not None:
            bk = SimHash.bucket(sh)
            try:
                self.sh_buckets[bk].remove(node_id)
            except ValueError:
                pass

        # MinHashLSH: 不逐出（LSH允许少量残留，不影响正确性）

        self._remove_count += 1

    def update(self, node_id: int, name: str, text: str):
        """更新索引（先remove再add）"""
        entity_type = self.node_types.get(node_id, "entity")
        self.remove(node_id)
        self.add(node_id, name, text, entity_type)

    # ── 检索 ────────────────────────────────────────────────

    def lookup_exact(self, name: str) -> Optional[int]:
        """
        Hash O(1) 精确查找 → 返回 node_id 或 None
        纳秒级响应
        """
        return self.hash_idx.get(name)

    def lookup_keyword(self, query: str, top_k: int = 20) -> List[Tuple[int, float]]:
        """
        Keyword索引查找 → 返回 [(node_id, score), ...]
        对query分词，查kw_idx，按命中次数排序
        微秒级响应
        """
        if not query.strip():
            return []

        tokens = SimHash.tokens(query)
        if not tokens:
            return []

        # 统计每个node_id的命中次数
        hits: Dict[int, int] = defaultdict(int)
        for tok in tokens:
            for nid in self.kw_idx.get(tok, []):
                hits[nid] += 1

        if not hits:
            return []

        # 按命中次数降序，按命中率算分
        max_hits = max(hits.values())
        scored = [
            (nid, round(hits[nid] / max_hits, 4))
            for nid in hits
        ]
        scored.sort(key=lambda x: -x[1])
        return scored[:top_k]

    def search_fts(self, query: str, top_k: int = 20) -> List[Tuple[int, float]]:
        """
        FTS5全文搜索 → 返回 [(node_id, rank), ...]
        短词(≤2字)走LIKE模糊匹配，长词走FTS5 MATCH
        微秒级响应
        """
        if not query.strip():
            return []

        q = query.strip()

        # 短词走LIKE
        if len(q) <= 2:
            pattern = f"%{q}%"
            cursor = self._conn.execute(
                "SELECT id, -1.0 as rank FROM idx_nodes WHERE name LIKE ? OR text LIKE ? "
                "ORDER BY id LIMIT ?",
                (pattern, pattern, top_k)
            )
        else:
            try:
                cursor = self._conn.execute(
                    "SELECT n.id, rank FROM idx_nodes_fts f "
                    "JOIN idx_nodes n ON n.id = f.rowid "
                    "WHERE idx_nodes_fts MATCH ? "
                    "ORDER BY rank LIMIT ?",
                    (q, top_k)
                )
            except sqlite3.OperationalError:
                # FTS5 MATCH 语法错误时回退LIKE
                pattern = f"%{q}%"
                cursor = self._conn.execute(
                    "SELECT id, -1.0 as rank FROM idx_nodes "
                    "WHERE name LIKE ? OR text LIKE ? "
                    "ORDER BY id LIMIT ?",
                    (pattern, pattern, top_k)
                )

        results = []
        for row in cursor.fetchall():
            rank = row["rank"]
            results.append((row["id"], round(float(rank) if rank else 0.0, 4)))

        return results

    def find_duplicate(self, text: str) -> Optional[int]:
        """
        SimHash去重 + MinHashLSH兜底

        流程:
          1. 计算新文本SimHash
          2. 查对应桶内节点
          3. 汉明距离 ≤ 2 → 返回node_id（重复）
          4. SimHash没命中 → MinHashLSH候选
          5. Jaccard > 0.5 → 返回node_id（相似）
          6. 否则返回None（全新）
        """
        if not text.strip():
            return None

        # ── Step 1-3: SimHash去重 ──
        sh = SimHash.hash(text)
        bk = SimHash.bucket(sh)

        for nid in self.sh_buckets.get(bk, []):
            existing_sh = self.node_simhash.get(nid, 0)
            if existing_sh and SimHash.hamming(sh, existing_sh) <= 3:
                return nid

        # ── Step 4-5: MinHashLSH兜底 ──
        candidates = self.lsh.query(text)
        for nid in candidates:
            existing_text = self.node_texts.get(nid, "")
            if existing_text:
                jac = MinHashLSH.jaccard(text, existing_text)
                if jac > 0.4:
                    return nid

        return None

    def search(self, query: str, mode: str = "auto",
               top_k: int = 20) -> Dict[str, any]:
        """
        统一检索入口

        mode:
          "exact"   → 只走Hash（纳秒）
          "keyword" → 只走Keyword索引（微秒）
          "fts"     → 只走FTS5（微秒）
          "auto"    → Hash → Keyword → FTS5 逐级回退

        返回:
          {
            "mode": "exact"|"keyword"|"fts",
            "results": [(node_id, score), ...],
            "total": int,
            "query_time_us": float,  # 微秒
          }
        """
        import time
        t0 = time.perf_counter()

        if mode == "exact":
            nid = self.lookup_exact(query)
            results = [(nid, 1.0)] if nid is not None else []
            used_mode = "exact"

        elif mode == "keyword":
            results = self.lookup_keyword(query, top_k)
            used_mode = "keyword"

        elif mode == "fts":
            results = self.search_fts(query, top_k)
            used_mode = "fts"

        elif mode == "auto":
            # 逐级回退: Hash → Keyword → FTS5
            nid = self.lookup_exact(query)
            if nid is not None:
                results = [(nid, 1.0)]
                used_mode = "exact"
            else:
                results = self.lookup_keyword(query, top_k)
                if results:
                    used_mode = "keyword"
                else:
                    results = self.search_fts(query, top_k)
                    used_mode = "fts"
        else:
            raise ValueError(f"未知搜索模式: {mode}")

        elapsed_us = (time.perf_counter() - t0) * 1_000_000

        return {
            "mode": used_mode,
            "results": results,
            "total": len(results),
            "query_time_us": round(elapsed_us, 2),
        }

    # ── 持久化 ──────────────────────────────────────────────

    def save_to_file(self, filepath: Optional[str] = None):
        """
        持久化FTS5数据库到文件

        Args:
            filepath: 目标文件路径，默认 data_dir/workspace_fts.db
        """
        if filepath is None:
            filepath = str(self._data_dir / f"{self.workspace}_fts.db")

        # 将内存数据库dump到文件
        dst_conn = sqlite3.connect(filepath)
        dst_conn.row_factory = sqlite3.Row

        # 备份
        self._conn.backup(dst_conn)

        # 同时保存索引元数据到JSON
        meta_path = Path(filepath).with_suffix(".idx.json")
        meta = {
            "workspace": self.workspace,
            "hash_idx": self.hash_idx,
            "kw_idx": {k: list(v) for k, v in self.kw_idx.items()},
            "node_simhash": self.node_simhash,
            "sh_buckets": {str(k): v for k, v in self.sh_buckets.items()},
            "node_names": self.node_names,
            "node_texts": self.node_texts,
            "node_types": self.node_types,
            "add_count": self._add_count,
            "remove_count": self._remove_count,
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        dst_conn.close()

    def load_from_file(self, filepath: Optional[str] = None):
        """
        从文件加载FTS5数据库

        Args:
            filepath: 源文件路径
        """
        if filepath is None:
            filepath = str(self._data_dir / f"{self.workspace}_fts.db")

        meta_path = Path(filepath).with_suffix(".idx.json")

        # 加载FTS5数据库
        if os.path.exists(filepath):
            src_conn = sqlite3.connect(filepath)
            src_conn.backup(self._conn)
            src_conn.close()

        # 加载索引元数据
        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)

            self.hash_idx = meta.get("hash_idx", {})
            self.kw_idx = defaultdict(list, {
                k: v for k, v in meta.get("kw_idx", {}).items()
            })
            self.node_simhash = {int(k): v for k, v in meta.get("node_simhash", {}).items()}
            self.sh_buckets = defaultdict(list, {
                int(k): v for k, v in meta.get("sh_buckets", {}).items()
            })
            self.node_names = {int(k): v for k, v in meta.get("node_names", {}).items()}
            self.node_texts = {int(k): v for k, v in meta.get("node_texts", {}).items()}
            self.node_types = {int(k): v for k, v in meta.get("node_types", {}).items()}
            self._add_count = meta.get("add_count", 0)
            self._remove_count = meta.get("remove_count", 0)

            # 重建MinHashLSH
            self.lsh = MinHashLSH()
            for nid, text in self.node_texts.items():
                self.lsh.insert(nid, text)

    # ── 重建 ────────────────────────────────────────────────

    def rebuild(self, nodes: List[dict]):
        """
        从节点列表完全重建索引

        Args:
            nodes: [{"id": int, "name": str, "text": str, "entity_type": str}, ...]
        """
        # 清空
        self.hash_idx.clear()
        self.kw_idx.clear()
        self.node_texts.clear()
        self.node_names.clear()
        self.node_types.clear()
        self.sh_buckets.clear()
        self.node_simhash.clear()
        self.lsh = MinHashLSH()
        self._add_count = 0
        self._remove_count = 0

        # 重建FTS5
        self._conn.execute("DELETE FROM idx_nodes")
        self._conn.commit()

        # 重新添加
        for node in nodes:
            self.add(
                node_id=node["id"],
                name=node.get("name", ""),
                text=node.get("text", ""),
                entity_type=node.get("entity_type", "entity"),
            )

    # ── 统计 ────────────────────────────────────────────────

    def stats(self) -> dict:
        """返回索引统计信息"""
        kw_total_entries = sum(len(v) for v in self.kw_idx.values())
        sh_total = sum(len(v) for v in self.sh_buckets.values())

        # FTS5统计
        cursor = self._conn.execute("SELECT COUNT(*) as c FROM idx_nodes")
        fts_count = cursor.fetchone()["c"]

        return {
            "workspace": self.workspace,
            "hash_entries": len(self.hash_idx),
            "kw_tokens": len(self.kw_idx),
            "kw_entries": kw_total_entries,
            "fts_nodes": fts_count,
            "simhash_entries": len(self.node_simhash),
            "simhash_buckets": len(self.sh_buckets),
            "simhash_bucket_entries": sh_total,
            "lsh": self.lsh.stats(),
            "add_count": self._add_count,
            "remove_count": self._remove_count,
        }


# ── 便捷入口 ──

def create_index(workspace: str = "global", data_dir: Optional[str] = None) -> FastIndex:
    """创建FastIndex实例"""
    return FastIndex(workspace=workspace, data_dir=data_dir)
