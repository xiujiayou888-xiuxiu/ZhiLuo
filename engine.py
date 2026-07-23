# -*- coding: utf-8 -*-
"""
知络 v8.9.1 — 统一版记忆引擎（开源版）
优化：jieba分词 + MinHashLSH去重 + LLM记忆合并 + sqlite-vec向量 + Mermaid可视化 + 返回截断
架构：内存 hash 索引（O(1)）+ SQLite+FTS5 持久化 + 向量持久化
"""
import re, json, math, os, sys, hashlib, sqlite3, threading, random, uuid
from datetime import datetime, timedelta
from pathlib import Path
from collections import Counter, defaultdict, deque, OrderedDict
from security_utils import SecurityError, safe_workspace_db_path, validate_workspace_name
from security_utils import ZhiLuoError, EngineError, ValidationError  # v8.9.7

# ====================== jieba 分词（优化1）======================
try:
    import jieba
    import jieba.analyse
    try:
        import logging as _logging
        jieba.setLogLevel(_logging.ERROR)
    except Exception:
        import logging as _logging; _logging.getLogger("zhiluo.engine").warning("engine.py: swallowed exception", exc_info=True)
    _HAS_JIEBA = True
except ImportError:
    _HAS_JIEBA = False

_TOKEN_TEXT_MAX_CHARS = int(os.environ.get("ZHILUO_TOKEN_TEXT_MAX_CHARS", "12000") or "12000")
_TOKEN_MAX_TOKENS = int(os.environ.get("ZHILUO_TOKEN_MAX_TOKENS", "1500") or "1500")
_LSH_MAX_TOKENS = int(os.environ.get("ZHILUO_LSH_MAX_TOKENS", "256") or "256")
_TOKEN_CACHE_SIZE = int(os.environ.get("ZHILUO_TOKEN_CACHE_SIZE", "256") or "256")
_TOKEN_CACHE = OrderedDict()
_TOKEN_CACHE_LOCK = threading.Lock()


def _clip_for_tokenize(text, max_chars=None):
    """Use representative slices for token indexes; keep full text in SQLite/FTS."""
    text = str(text or "")
    limit = max(1000, int(max_chars or _TOKEN_TEXT_MAX_CHARS))
    if len(text) <= limit:
        return text
    head = max(1, int(limit * 0.60))
    mid = max(1, int(limit * 0.20))
    tail = max(1, limit - head - mid)
    start_mid = max(0, (len(text) - mid) // 2)
    return text[:head] + "\n" + text[start_mid:start_mid + mid] + "\n" + text[-tail:]


def _token_cache_key(text, max_chars=None, max_tokens=None):
    clipped = _clip_for_tokenize(text, max_chars=max_chars)
    raw = "%s:%s:%s" % (max_chars or _TOKEN_TEXT_MAX_CHARS, max_tokens or _TOKEN_MAX_TOKENS, clipped)
    return hashlib.sha1(raw.encode("utf-8", errors="ignore")).hexdigest(), clipped


def smart_tokenize(text, max_chars=None, max_tokens=None):
    """jieba 分词，无 jieba 时回退到单字切分"""
    cache_key, clipped = _token_cache_key(text, max_chars=max_chars, max_tokens=max_tokens)
    with _TOKEN_CACHE_LOCK:
        cached = _TOKEN_CACHE.get(cache_key)
        if cached is not None:
            _TOKEN_CACHE.move_to_end(cache_key)
            return list(cached)
    text = clipped.lower()
    if _HAS_JIEBA:
        words = list(jieba.cut(text))
        result = [w.strip() for w in words if w.strip() and w.strip() not in SimHash._STOP]
    else:
        out = []
        for t in re.findall(r"[\w\u4e00-\u9fff]+", text):
            if re.match(r"^[\u4e00-\u9fff]+$", t):
                out.extend(list(t))
            else:
                out.append(t)
        result = [t for t in out if t not in SimHash._STOP]
    result = result[:max(10, int(max_tokens or _TOKEN_MAX_TOKENS))]
    with _TOKEN_CACHE_LOCK:
        _TOKEN_CACHE[cache_key] = tuple(result)
        while len(_TOKEN_CACHE) > _TOKEN_CACHE_SIZE:
            _TOKEN_CACHE.popitem(last=False)
    return list(result)

def smart_extract_keywords(text, top_k=10):
    """jieba 关键词抽取，无 jieba 时回退到 TF-IDF"""
    if _HAS_JIEBA:
        return jieba.analyse.extract_tags(text, topK=top_k)
    else:
        toks = smart_tokenize(text)
        freq = Counter(toks)
        return [w for w, _ in freq.most_common(top_k)]


def fast_index_tokenize(text, max_chars=None, max_tokens=None):
    """Fast fallback for loading old rows without text_jieba cache."""
    text = _clip_for_tokenize(text, max_chars=max_chars).lower()
    out = []
    for token in re.findall(r"[a-z0-9_]{2,}|[\u4e00-\u9fff]{2,6}", text):
        if token and token not in SimHash._STOP:
            out.append(token)
        if len(out) >= max(10, int(max_tokens or _TOKEN_MAX_TOKENS)):
            break
    return out

# ====================== SimHash（优化1: jieba分词）======================
class SimHash:
    _STOP = set("的了是在我有和就不人都一个上也这到说他她你们这那之")

    @staticmethod
    def tokens(text, max_chars=None, max_tokens=None):
        return smart_tokenize(text, max_chars=max_chars, max_tokens=max_tokens)

    @staticmethod
    def hash(text):
        return SimHash.hash_tokens(SimHash.tokens(text))

    @staticmethod
    def hash_tokens(toks):
        if not toks:
            return 0
        v = [0] * 64
        for tok in toks:
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16) & 0x7FFFFFFFFFFFFFFF
            for i in range(64):
                if (h >> i) & 1:
                    v[i] += 1
                else:
                    v[i] -= 1
        out = 0
        for i in range(64):
            if v[i] > 0:
                out |= 1 << i
        return out & 0x7FFFFFFFFFFFFFFF

    @staticmethod
    def hamming(a, b):
        return bin(a ^ b).count("1")

    @staticmethod
    def bucket(sh):
        return sh & 0xF

# ====================== MinHash + LSH（优化4: 近似去重）======================
class MinHashLSH:
    """MinHash + LSH 近似去重，补充 SimHash 的漏判"""
    _NUM_PERM = 128
    _BANDS = 32
    _ROWS = 4

    def __init__(self):
        self._mersenne = (1 << 61) - 1
        self._a = [random.randint(1, self._mersenne) for _ in range(self._NUM_PERM)]
        self._b = [random.randint(0, self._mersenne) for _ in range(self._NUM_PERM)]
        self._bands = defaultdict(dict)

    def _signature(self, tokens):
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

    def _band_keys(self, sig):
        keys = []
        for b in range(self._BANDS):
            start = b * self._ROWS
            chunk = tuple(sig[start:start + self._ROWS])
            keys.append((b, hash(chunk)))
        return keys

    def _bounded_tokens(self, tokens):
        tokens = list(tokens or [])
        limit = max(32, int(_LSH_MAX_TOKENS))
        if len(tokens) <= limit:
            return tokens
        head = int(limit * 0.60)
        mid = int(limit * 0.20)
        tail = max(1, limit - head - mid)
        start_mid = max(0, (len(tokens) - mid) // 2)
        return tokens[:head] + tokens[start_mid:start_mid + mid] + tokens[-tail:]

    def insert_tokens(self, nid, tokens):
        sig = self._signature(self._bounded_tokens(tokens))
        for b, h in self._band_keys(sig):
            self._bands[b].setdefault(h, []).append(nid)
        return sig

    def insert(self, nid, text):
        return self.insert_tokens(nid, SimHash.tokens(text))

    def query_tokens(self, tokens):
        """返回可能相似的节点 ID 列表"""
        sig = self._signature(self._bounded_tokens(tokens))
        candidates = set()
        for b, h in self._band_keys(sig):
            for nid in self._bands.get(b, {}).get(h, []):
                candidates.add(nid)
        return list(candidates)

    def query(self, text):
        """返回可能相似的节点 ID 列表"""
        return self.query_tokens(SimHash.tokens(text))

    def jaccard(self, text_a, text_b):
        sa = set(SimHash.tokens(text_a))
        sb = set(SimHash.tokens(text_b))
        if not sa and not sb:
            return 1.0
        return len(sa & sb) / max(len(sa | sb), 1)

# ====================== SQLite + 内存双引擎存储 ======================
SKILL_DIR = Path(__file__).resolve().parent
try:
    from kb_config import get_data_dir, get_kb_path
    DATA_DIR = Path(get_data_dir())
    WS_DIR = Path(get_kb_path())
except Exception:
    DATA_DIR = SKILL_DIR / "data"
    WS_DIR = DATA_DIR / "workspaces"
DATA_DIR.mkdir(parents=True, exist_ok=True)
WS_DIR.mkdir(parents=True, exist_ok=True)
DATA_FILE = DATA_DIR / "memory.db"
EDGE_RELATED, EDGE_CAUSE = "related", "cause"
EDGE_PART_OF, EDGE_SYNONYM = "part_of", "synonym"
EDGE_REFINES, EDGE_CONTRADICTS = "refines", "contradicts"
EDGE_TYPES = (EDGE_RELATED, EDGE_CAUSE, EDGE_PART_OF, EDGE_SYNONYM, EDGE_REFINES, EDGE_CONTRADICTS)

class MemoryStore:
    """SQLite+FTS5 持久化 + 内存 hash 索引 O(1) 读取 + 向量持久化"""

    _local = threading.local()

    # v8.9.7: 允许通过 update() 修改的列名白名单（防 SQL 注入）
    _ALLOWED_COLUMNS = {
        "text", "category", "tags", "confidence", "trust_score",
        "access_count", "last_accessed_at", "updated_at", "source",
        "status", "visibility", "para", "edges_json", "simhash",
        "created_by", "confirmed_by", "source_url", "content_hash",
        "text_jieba", "workspace", "learned_at", "created_at",
        "fetched_at", "session_id",
    }
    # v8.9.7: 允许 ALTER TABLE 的列声明白名单（从 _ensure_* 硬编码列表中提取）
    _KNOWN_COLUMN_DECLS = {
        "para": "TEXT DEFAULT ''",
        "created_by": "TEXT DEFAULT 'default'",
        "visibility": "TEXT DEFAULT 'team'",
        "trust_score": "REAL DEFAULT 1.0",
        "source_url": "TEXT DEFAULT ''",
        "content_hash": "TEXT DEFAULT ''",
        "fetched_at": "TEXT DEFAULT ''",
        "status": "TEXT DEFAULT 'active'",
        "confirmed_by": "TEXT DEFAULT ''",
        "text_jieba": "TEXT DEFAULT ''",
    }

    def __init__(self, path=None, workspace="global"):
        workspace = validate_workspace_name(workspace)
        self.path = Path(path) if path else safe_workspace_db_path(WS_DIR, workspace)
        if path:
            self.path = Path(path).resolve()
        self.ws = workspace
        self.nodes = []
        self.adj = {}
        self.hash_idx = {}
        self.kw_idx = {}
        self.sh_buckets = {}
        self.node_pos = {}
        self.lsh = MinHashLSH()
        self._next_id = 1
        self._write_lock = threading.RLock()  # v8.9.7: 保护写操作线程安全
        self._init_db()
        self.load()

    def _get_conn(self):
        if not hasattr(MemoryStore._local, "conn") or MemoryStore._local.conn is None:
            MemoryStore._local.conn = sqlite3.connect(str(self.path), timeout=30, check_same_thread=False)
            MemoryStore._local.conn.execute("PRAGMA journal_mode=WAL")
            MemoryStore._local.conn.execute("PRAGMA busy_timeout=30000")
            MemoryStore._local.conn.execute("PRAGMA foreign_keys=ON")
            MemoryStore._local.conn.execute("PRAGMA synchronous=NORMAL")
            MemoryStore._local.conn.execute("PRAGMA cache_size=-65536")
            MemoryStore._local.conn.row_factory = sqlite3.Row
        else:
            try:
                cur = MemoryStore._local.conn.execute("PRAGMA database_list")
                rows = cur.fetchall()
                if rows and rows[0][2]:
                    curr_path = str(Path(rows[0][2]).resolve())
                    my_path = str(self.path.resolve())
                    if curr_path != my_path:
                        MemoryStore._local.conn.close()
                        MemoryStore._local.conn = sqlite3.connect(str(self.path), timeout=30, check_same_thread=False)
                        MemoryStore._local.conn.execute("PRAGMA journal_mode=WAL")
                        MemoryStore._local.conn.execute("PRAGMA busy_timeout=30000")
                        MemoryStore._local.conn.execute("PRAGMA foreign_keys=ON")
                        MemoryStore._local.conn.execute("PRAGMA synchronous=NORMAL")
                        MemoryStore._local.conn.row_factory = sqlite3.Row
            except Exception:
                import logging as _logging
                _logging.getLogger("zhiluo.engine").warning(
                    "MemoryStore._get_conn: path validation failed, reusing existing connection", exc_info=True)
        return MemoryStore._local.conn

    def _close_conn(self):
        if hasattr(MemoryStore._local, "conn") and MemoryStore._local.conn is not None:
            MemoryStore._local.conn.close()
            MemoryStore._local.conn = None

    def _init_db(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        conn = self._get_conn()
        conn.executescript('''
            CREATE TABLE IF NOT EXISTS nodes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text TEXT NOT NULL,
                workspace TEXT NOT NULL DEFAULT 'global',
                category TEXT DEFAULT '未分类',
                simhash INTEGER DEFAULT 0,
                minhash_sig TEXT DEFAULT '[]',
                access_count INTEGER DEFAULT 0,
                tags TEXT DEFAULT '[]',
                confidence REAL DEFAULT 1.0,
                created_at TEXT,
                learned_at TEXT,
                updated_at TEXT,
                last_accessed_at TEXT,
                session_id TEXT,
                source TEXT DEFAULT 'user',
                edges_json TEXT DEFAULT '[]',
                embedding BLOB,
                para TEXT DEFAULT '',
                created_by TEXT DEFAULT 'default',
                visibility TEXT DEFAULT 'team',
                trust_score REAL DEFAULT 1.0,
                source_url TEXT DEFAULT '',
                content_hash TEXT DEFAULT '',
                fetched_at TEXT DEFAULT '',
                status TEXT DEFAULT 'active',
                confirmed_by TEXT DEFAULT '',
                text_jieba TEXT DEFAULT ''
            );
            CREATE VIRTUAL TABLE IF NOT EXISTS nodes_fts USING fts5(
                text, category, tags,
                content='nodes', content_rowid='id',
                tokenize='trigram'
            );
            CREATE TRIGGER IF NOT EXISTS nodes_ai AFTER INSERT ON nodes BEGIN
                INSERT INTO nodes_fts(rowid, text, category, tags)
                VALUES (new.id, new.text, new.category, new.tags);
            END;
            CREATE TRIGGER IF NOT EXISTS nodes_ad AFTER DELETE ON nodes BEGIN
                INSERT INTO nodes_fts(nodes_fts, rowid, text, category, tags)
                VALUES ('delete', old.id, old.text, old.category, old.tags);
            END;
            CREATE TRIGGER IF NOT EXISTS nodes_au AFTER UPDATE ON nodes BEGIN
                INSERT INTO nodes_fts(nodes_fts, rowid, text, category, tags)
                VALUES ('delete', old.id, old.text, old.category, old.tags);
                INSERT INTO nodes_fts(rowid, text, category, tags)
                VALUES (new.id, new.text, new.category, new.tags);
            END;
        ''')
        try:
            conn.execute("CREATE TABLE IF NOT EXISTS vec_nodes (id INTEGER PRIMARY KEY, embedding BLOB);")
        except Exception:
            import logging as _logging; _logging.getLogger("zhiluo.engine").warning("engine.py: swallowed exception", exc_info=True)
        self._ensure_v89_columns(conn)
        # === 超级版新增: pending待确认队列 + 变更历史 ===
        try:
            conn.executescript('''
                CREATE TABLE IF NOT EXISTS pending (
                    id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    category TEXT,
                    workspace TEXT DEFAULT 'global',
                    source TEXT DEFAULT 'auto',
                    created_at TEXT,
                    source_url TEXT DEFAULT '',
                    content_hash TEXT DEFAULT '',
                    source_hash TEXT DEFAULT ''
                );
                CREATE TABLE IF NOT EXISTS change_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    node_id INTEGER,
                    action TEXT,
                    old_text TEXT,
                    new_text TEXT,
                    timestamp TEXT
                );
            ''')
        except Exception:
            import logging as _logging; _logging.getLogger("zhiluo.engine").warning("engine.py: swallowed exception", exc_info=True)
        try:
            self._ensure_pending_columns(conn)
        except Exception:
            import logging as _logging; _logging.getLogger("zhiluo.engine").warning("engine.py: swallowed exception", exc_info=True)
        # 常用查询索引
        try:
            conn.execute("CREATE INDEX IF NOT EXISTS idx_nodes_cat ON nodes(category)")
        except Exception:
            import logging as _logging; _logging.getLogger("zhiluo.engine").warning("engine.py: swallowed exception", exc_info=True)
        try:
            conn.execute("CREATE INDEX IF NOT EXISTS idx_nodes_ws ON nodes(workspace)")
        except Exception:
            import logging as _logging; _logging.getLogger("zhiluo.engine").warning("engine.py: swallowed exception", exc_info=True)
        try:
            conn.execute("CREATE INDEX IF NOT EXISTS idx_nodes_conf ON nodes(confidence)")
        except Exception:
            import logging as _logging; _logging.getLogger("zhiluo.engine").warning("engine.py: swallowed exception", exc_info=True)
        conn.commit()

    def _ensure_v89_columns(self, conn):
        columns = [
            ("para", "TEXT DEFAULT ''"),
            ("created_by", "TEXT DEFAULT 'default'"),
            ("visibility", "TEXT DEFAULT 'team'"),
            ("trust_score", "REAL DEFAULT 1.0"),
            ("source_url", "TEXT DEFAULT ''"),
            ("content_hash", "TEXT DEFAULT ''"),
            ("fetched_at", "TEXT DEFAULT ''"),
            ("status", "TEXT DEFAULT 'active'"),
            ("confirmed_by", "TEXT DEFAULT ''"),
            ("text_jieba", "TEXT DEFAULT ''"),
        ]
        existing = {r[1] for r in conn.execute("PRAGMA table_info(nodes)").fetchall()}
        for name, decl in columns:
            if name not in existing:
                # v8.9.7: 列名和声明白名单校验
                if name not in self._KNOWN_COLUMN_DECLS or self._KNOWN_COLUMN_DECLS.get(name) != decl:
                    import logging as _logging
                    _logging.getLogger("zhiluo.engine").warning(
                        "Skipping unknown column: %s %s", name, decl)
                    continue
                conn.execute("ALTER TABLE nodes ADD COLUMN %s %s" % (name, decl))

    def _ensure_pending_columns(self, conn):
        columns = [
            ("source_url", "TEXT DEFAULT ''"),
            ("content_hash", "TEXT DEFAULT ''"),
            ("source_hash", "TEXT DEFAULT ''"),
        ]
        existing = {r[1] for r in conn.execute("PRAGMA table_info(pending)").fetchall()}
        for name, decl in columns:
            if name not in existing:
                # v8.9.7: 列名和声明白名单校验
                if name not in self._KNOWN_COLUMN_DECLS and name not in {"source_hash"}:
                    import logging as _logging
                    _logging.getLogger("zhiluo.engine").warning(
                        "Skipping unknown pending column: %s %s", name, decl)
                    continue
                conn.execute("ALTER TABLE pending ADD COLUMN %s %s" % (name, decl))
        conn.execute("CREATE INDEX IF NOT EXISTS idx_pending_source_url ON pending(source_url)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_pending_content_hash ON pending(content_hash)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_pending_source_hash ON pending(source_hash)")

    def _cached_tokens(self, node):
        cached = str(node.get("text_jieba") or "").strip()
        # Very old rows may contain full-document token dumps; splitting those
        # recreates the load spike we are avoiding. Retokenize a bounded slice.
        if cached and len(cached) <= _TOKEN_TEXT_MAX_CHARS * 4:
            toks = [t for t in cached.split() if t and t not in SimHash._STOP]
            if toks:
                return toks[:_TOKEN_MAX_TOKENS]
        return fast_index_tokenize(node.get("text", ""))

    def _index_node_tokens(self, nid, tokens):
        for kw in set(tokens):
            self.kw_idx.setdefault(kw, []).append(nid)
        self.lsh.insert_tokens(nid, tokens)

    def _remove_node_tokens(self, nid, tokens):
        for kw in set(tokens):
            try:
                self.kw_idx.get(kw, []).remove(nid)
            except ValueError:
                pass

    def load(self):
        self.nodes = []
        self.adj = {}
        self.hash_idx = {}
        self.kw_idx = {}
        self.sh_buckets = {}
        self.node_pos = {}
        self.lsh = MinHashLSH()
        self._next_id = 1
        if not self.path.exists():
            self._migrate_from_json()
            if not self.path.exists():
                return
        conn = self._get_conn()
        try:
            cursor = conn.execute("SELECT * FROM nodes ORDER BY id")
            load_batch = max(100, int(os.environ.get("ZHILUO_LOAD_BATCH", "2000") or "2000"))
            while True:
                rows = cursor.fetchmany(load_batch)
                if not rows:
                    break
                for row in rows:
                    node = dict(row)
                    try:
                        edges = json.loads(node.pop("edges_json", "[]"))
                    except (json.JSONDecodeError, TypeError):
                        edges = []
                    # 兼容旧格式：过滤掉非 dict 的边条目
                    edges = [e for e in (edges if isinstance(edges, list) else []) if isinstance(e, dict)]
                    node["edges"] = edges
                    try:
                        tags = json.loads(node.get("tags", "[]"))
                    except (json.JSONDecodeError, TypeError):
                        tags = []
                    node["tags"] = tags if isinstance(tags, list) else []
                    node.pop("minhash_sig", None)
                    node.pop("embedding", None)
                    self.nodes.append(node)
                    nid = node["id"]
                    self.node_pos[nid] = len(self.nodes) - 1
                    self.adj[nid] = []
                    sh = node.get("simhash", 0)
                    if sh:
                        self.hash_idx[sh] = nid
                    bk = SimHash.bucket(sh)
                    self.sh_buckets.setdefault(bk, []).append(nid)
                    index_tokens = self._cached_tokens(node)
                    self._index_node_tokens(nid, index_tokens)
                    for e in edges:
                        if isinstance(e, str):
                            # 旧格式兼容：字符串边转 dict
                            try:
                                e = json.loads(e)
                            except (json.JSONDecodeError, TypeError):
                                e = {"target": None, "type": EDGE_RELATED, "weight": 1.0}
                        if not isinstance(e, dict):
                            continue
                        self.adj[nid].append(e)
                        target = e.get("target")
                        if target is not None:
                            self.adj.setdefault(target, []).append(
                                {"target": nid, "type": e.get("type", EDGE_RELATED), "weight": e.get("weight", 1.0)}
                            )
            self._next_id = (max((n["id"] for n in self.nodes), default=0) + 1)
        except sqlite3.OperationalError:
            self._migrate_from_json()

    def _migrate_from_json(self):
        json_path = self.path.with_suffix(".json")
        if json_path.exists():
            try:
                data = json.loads(json_path.read_text(encoding="utf-8"))
                legacy_nodes = data.get("nodes", [])
                if legacy_nodes:
                    conn = self._get_conn()
                    conn.executescript("DELETE FROM nodes; DELETE FROM nodes_fts;")
                    for n in legacy_nodes:
                        nid = n.get("id")
                        if nid is None:
                            continue
                        edges = n.get("edges", [])
                        tags = n.get("tags", [])
                        conn.execute(
                            """INSERT OR IGNORE INTO nodes (id, text, workspace, category, simhash,
                               access_count, tags, confidence, created_at, learned_at,
                               updated_at, last_accessed_at, session_id, source, edges_json)
                               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                            (
                                nid,
                                n.get("text", ""),
                                n.get("workspace", "global"),
                                n.get("category", "未分类"),
                                n.get("simhash", 0),
                                n.get("access_count", 0),
                                json.dumps(tags, ensure_ascii=False),
                                n.get("confidence", 1.0),
                                n.get("created_at"),
                                n.get("learned_at"),
                                n.get("updated_at"),
                                n.get("last_accessed_at"),
                                n.get("session_id"),
                                n.get("source", "user"),
                                json.dumps(edges, ensure_ascii=False),
                            ),
                        )
                    conn.commit()
                    json_path.rename(json_path.with_suffix(".json.migrated"))
            except Exception:
                import logging as _logging; _logging.getLogger("zhiluo.engine").warning("engine.py: swallowed exception", exc_info=True)
        self._init_db()

    def save(self):
        conn = self._get_conn()
        conn.commit()

    def add(self, text, workspace=None, category=None, session_id=None, source="user", confidence=1.0):
        now = datetime.now().isoformat()
        workspace = workspace or self.ws
        if not category:
            category = self._cat(text)

        index_tokens = SimHash.tokens(text)
        text_jieba = " ".join(index_tokens) if _HAS_JIEBA else ""
        sh = SimHash.hash_tokens(index_tokens)
        bk = SimHash.bucket(sh)

        with self._write_lock:
            conn = self._get_conn()

            # Let SQLite allocate ids. In a shared multi-Agent DB, self._next_id can
            # become stale while other Agents or background jobs insert rows.
            try:
                cur = conn.execute(
                    """INSERT INTO nodes (text, workspace, category, simhash,
                       access_count, tags, confidence, created_at, learned_at,
                       updated_at, last_accessed_at, session_id, source, edges_json, text_jieba)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        text, workspace, category, sh,
                        0, "[]", confidence, now, now,
                        None, None, session_id, source, "[]", text_jieba,
                    ),
                )
            except sqlite3.OperationalError:
                cur = conn.execute(
                    """INSERT INTO nodes (text, workspace, category, simhash,
                       access_count, tags, confidence, created_at, learned_at,
                       updated_at, last_accessed_at, session_id, source, edges_json)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        text, workspace, category, sh,
                        0, "[]", confidence, now, now,
                        None, None, session_id, source, "[]",
                    ),
                )
            conn.commit()

            nid = int(cur.lastrowid)
            self._next_id = max(self._next_id, nid + 1)
            node = {
                "id": nid, "text": text, "workspace": workspace, "category": category, "simhash": sh,
                "access_count": 0, "tags": [], "confidence": confidence,
                "created_at": now, "learned_at": now, "updated_at": None, "last_accessed_at": None,
                "session_id": session_id, "source": source, "edges": [],
            }
            self.nodes.append(node)
            self.node_pos[nid] = len(self.nodes) - 1
            self.hash_idx[sh] = nid
            self.adj[nid] = []
            self.sh_buckets.setdefault(bk, []).append(nid)
            self._index_node_tokens(nid, index_tokens)
        try:
            self._auto_link(nid, text, workspace, category)
        except Exception:
            import logging as _logging
            _logging.getLogger("zhiluo.engine").warning(
                "_auto_link failed for nid=%s", nid, exc_info=True)
        return nid
    def _auto_link(self, nid, text, workspace, category, max_links=5, min_sim=0.35):
        """Create bidirectional related edges to similar existing knowledge."""
        conn = self._get_conn()
        rows = conn.execute(
            """SELECT id, text FROM nodes
               WHERE workspace=? AND category=? AND id != ?
               ORDER BY id DESC LIMIT 100""",
            (workspace, category, nid),
        ).fetchall()
        if not rows:
            return
        try:
            from semantic_base import semantic_find_similar
            candidates = [(r["id"], r["text"]) for r in rows[:50]]
            matches = semantic_find_similar(text, candidates, top_k=max_links)
        except Exception:
            import logging as _logging
            _logging.getLogger("zhiluo.engine").warning(
                "_auto_link semantic_find_similar failed", exc_info=True)
            return
        for match_id, _match_text, match_score in matches or []:
            if match_score < min_sim:
                continue
            self.add_edge(nid, match_id, "related", weight=match_score)
            self.add_edge(match_id, nid, "related", weight=match_score)

    def get(self, nid):
        i = self.node_pos.get(nid)
        return self.nodes[i] if i is not None else None

    def delete(self, nid):
        with self._write_lock:
            i = self.node_pos.get(nid)
            if i is None:
                return False
            # v8.9.7: 先删 DB，成功后再清内存，防止 DB 删除失败导致状态不一致
            conn = self._get_conn()
            conn.execute("DELETE FROM nodes WHERE id=?", (nid,))
            conn.commit()
            n = self.nodes[i]
            sh = n.get("simhash", 0)
            self.hash_idx.pop(sh, None)
            try:
                self.sh_buckets.get(SimHash.bucket(sh), []).remove(nid)
            except ValueError:
                pass
            self._remove_node_tokens(nid, self._cached_tokens(n))
            self.adj.pop(nid, None)
            for e in n.get("edges", []):
                target = e.get("target")
                if target in self.adj:
                    self.adj[target] = [e2 for e2 in self.adj[target] if e2.get("target") != nid]
            self.nodes[i] = None
            self.node_pos.pop(nid, None)
            return True

    def touch(self, nid):
        n = self.get(nid)
        if n:
            now = datetime.now().isoformat()
            n["access_count"] = n.get("access_count", 0) + 1
            n["last_accessed_at"] = now
            conn = self._get_conn()
            conn.execute("UPDATE nodes SET access_count=?, last_accessed_at=? WHERE id=?",
                         (n["access_count"], now, nid))
            conn.commit()

    def decay_confidence(self, workspace=None, stale_days=30, decay_rate=0.9, dry_run=False):
        """P1.2: 降低长期未访问知识的信任度（trust_score 优先，回退 confidence）。
        
        越久未访问→衰减越大；高访问量→抵抗衰减（被反复验证的知识不应轻易降权）。
        """
        ws = workspace or self.ws
        stale_since = (datetime.now() - timedelta(days=stale_days)).isoformat()
        conn = self._get_conn()
        # v8.9.7: BEGIN IMMEDIATE 确保衰减原子化 — 崩溃不产生部分衰减
        if not dry_run:
            conn.execute("BEGIN IMMEDIATE")
        try:
            rows = conn.execute(
                """SELECT id, trust_score, confidence, access_count, last_accessed_at
                   FROM nodes WHERE workspace=?
                   AND (last_accessed_at IS NULL OR last_accessed_at < ?)
                   AND COALESCE(trust_score, confidence, 1.0) > 0.1""",
                (ws, stale_since),
            ).fetchall()
            affected = 0
            for r in rows:
                trust = float(r["trust_score"] or r["confidence"] or 1.0)
                access = int(r["access_count"] or 0)
                # 高访问量减缓衰减：access>=10 时衰减率减半
                effective_rate = decay_rate + (1.0 - decay_rate) * min(1.0, access / 20.0) * 0.5
                new_trust = round(max(0.1, trust * effective_rate), 2)
                if abs(new_trust - trust) < 0.01:
                    continue
                if not dry_run:
                    conn.execute(
                        "UPDATE nodes SET trust_score=?, confidence=?, updated_at=? WHERE id=?",
                        (new_trust, new_trust, datetime.now().isoformat(), r["id"]),
                    )
                    i = self.node_pos.get(r["id"])
                    if i is not None and self.nodes[i] is not None:
                        self.nodes[i]["trust_score"] = new_trust
                        self.nodes[i]["confidence"] = new_trust
                affected += 1
            if not dry_run:
                conn.commit()
        except Exception:
            if not dry_run:
                conn.rollback()
            raise
        return {"affected": affected, "stale_days": stale_days, "decay_rate": decay_rate}

    def _cat(self, text):
        try:
            from knowledge_maintenance import _load_rules
            rules = _load_rules(self.path)
        except Exception:
            rules = [
                ("技术", ["python", "java", "代码", "bug", "接口", "api", "服务器", "数据库", "前端", "后端", "框架"]),
                ("项目", ["项目", "截止", "里程碑", "需求", "排期", "负责人", "上线", "迭代", "版本"]),
                ("人物", ["老板", "经理", "同事", "小王", "小李", "团队", "客户", "领导", "总监"]),
                ("财务", ["元", "万", "预算", "收入", "支出", "利润", "成本", "报销", "发票"]),
                ("会议", ["会议", "开会", "纪要", "讨论", "决议", "参会", "议程"]),
                ("学习", ["学习", "教程", "课程", "笔记", "知识点", "总结", "教程", "文档"]),
                ("生活", ["健身", "做饭", "菜谱", "运动", "饮食", "休息", "旅行"]),
            ]
        low = text.lower()
        best = "_待分类_"
        bs = 0
        for cat, kws in rules:
            s = sum(1 for kw in kws if kw in low)
            if s > bs:
                best = cat
                bs = s
        return best

    def stats(self):
        v = [n for n in self.nodes if n]
        edge_count = sum(len(n.get("edges", [])) for n in v)
        try:
            from knowledge_maintenance import edge_stats
            edge_count = edge_stats(self.path).get("directed_edges", edge_count)
        except Exception:
            import logging as _logging; _logging.getLogger("zhiluo.engine").warning("engine.py: swallowed exception", exc_info=True)
        return {
            "total": len(v),
            "workspaces": dict(Counter(n["workspace"] for n in v)),
            "categories": dict(Counter(n["category"] for n in v)),
            "edges": edge_count,
        }

    def valid(self, workspace=None):
        r = [n for n in self.nodes if n]
        if workspace:
            r = [n for n in r if n["workspace"] == workspace]
        return r

    def switch_ws(self, workspace):
        workspace = validate_workspace_name(workspace)
        # v8.9.7: 加写锁防止并发读看到空 nodes
        with self._write_lock:
            self.save()
            self._close_conn()
            self.ws = workspace
            self.path = safe_workspace_db_path(WS_DIR, workspace)
            self.nodes = []
            self._init_db()
            self.load()
        idx = set()
        idx_file = DATA_DIR / ".ws_index"
        if idx_file.exists():
            try:
                idx = set(json.loads(idx_file.read_text(encoding="utf-8")))
            except Exception:
                import logging as _logging; _logging.getLogger("zhiluo.engine").warning("engine.py: swallowed exception", exc_info=True)
        idx.add(workspace)
        idx_file.write_text(json.dumps(list(idx)), encoding="utf-8")

    def list_workspaces(self):
        idx_file = DATA_DIR / ".ws_index"
        if not idx_file.exists():
            return ["global"]
        try:
            names = json.loads(idx_file.read_text(encoding="utf-8"))
            return [validate_workspace_name(n) for n in names]
        except Exception:
            return ["global"]

    def fts_search(self, query, workspace=None, top_k=20):
        if not query.strip():
            return []
        conn = self._get_conn()
        q = query.strip()
        
        # v8.9: jieba分词查询词，用于FTS5
        if _HAS_JIEBA and len(q) > 2:
            q_jieba = " ".join(smart_tokenize(q))
        else:
            q_jieba = q
        
        if len(q) <= 2:
            pattern = "%" + q + "%"
            try:
                # v8.9: 同时搜索text和category字段
                if workspace:
                    cursor = conn.execute(
                        "SELECT * FROM nodes WHERE (text LIKE ? OR category LIKE ?) AND workspace=? ORDER BY id LIMIT ?",
                        (pattern, pattern, workspace, top_k),
                    )
                else:
                    cursor = conn.execute(
                        "SELECT * FROM nodes WHERE (text LIKE ? OR category LIKE ?) ORDER BY id LIMIT ?",
                        (pattern, pattern, top_k),
                    )
                results = []
                for row in cursor.fetchall():
                    node = dict(row)
                    try:
                        node["edges"] = json.loads(node.pop("edges_json", "[]"))
                    except (json.JSONDecodeError, TypeError):
                        node["edges"] = []
                    try:
                        node["tags"] = json.loads(node.get("tags", "[]"))
                    except (json.JSONDecodeError, TypeError):
                        node["tags"] = []
                    node.pop("minhash_sig", None)
                    node.pop("embedding", None)
                    node["_fts_rank"] = -1.0
                    results.append(node)
                return results
            except Exception:
                return []
        try:
            # v8.9: 尝试用jieba分词查询FTS，失败回退LIKE
            if workspace:
                cursor = conn.execute(
                    """SELECT n.*, rank FROM nodes_fts f
                       JOIN nodes n ON n.id = f.rowid
                       WHERE nodes_fts MATCH ? AND n.workspace = ?
                       ORDER BY rank
                       LIMIT ?""",
                    (q_jieba, workspace, top_k),
                )
            else:
                cursor = conn.execute(
                    """SELECT n.*, rank FROM nodes_fts f
                       JOIN nodes n ON n.id = f.rowid
                       WHERE nodes_fts MATCH ?
                       ORDER BY rank
                       LIMIT ?""",
                    (q_jieba, top_k),
                )
            results = []
            for row in cursor.fetchall():
                node = dict(row)
                rank = node.pop("rank", 0.0) or 0.0
                try:
                    node["edges"] = json.loads(node.pop("edges_json", "[]"))
                except (json.JSONDecodeError, TypeError):
                    node["edges"] = []
                try:
                    node["tags"] = json.loads(node.get("tags", "[]"))
                except (json.JSONDecodeError, TypeError):
                    node["tags"] = []
                node.pop("minhash_sig", None)
                node.pop("embedding", None)
                node["_fts_rank"] = round(float(rank), 4)
                results.append(node)
            return results
        except sqlite3.OperationalError:
            return []

    def add_edge(self, sid, tid, etype=EDGE_RELATED, weight=1.0):
        with self._write_lock:
            src = self.get(sid)
            tgt = self.get(tid)
            if not src or not tgt:
                return False
            edge = {"target": tid, "type": etype, "weight": weight}
            for e in src.get("edges", []):
                if e.get("target") == tid and e.get("type") == etype:
                    return False
            src["edges"].append(edge)
            self.adj.setdefault(sid, []).append(edge)
            self.adj.setdefault(tid, []).append(
                {"target": sid, "type": etype, "weight": weight}
            )
            conn = self._get_conn()
            conn.execute("UPDATE nodes SET edges_json=? WHERE id=?",
                         (json.dumps(src["edges"], ensure_ascii=False), sid))
            conn.commit()
            return True

    def remove_edge(self, sid, tid, etype=EDGE_RELATED):
        with self._write_lock:
            src = self.get(sid)
            if not src:
                return False
            src["edges"] = [e for e in src.get("edges", [])
                            if not (e.get("target") == tid and e.get("type") == etype)]
            if tid in self.adj:
                self.adj[tid] = [e for e in self.adj[tid]
                                 if not (e.get("target") == sid and e.get("type") == etype)]
            conn = self._get_conn()
            conn.execute("UPDATE nodes SET edges_json=? WHERE id=?",
                         (json.dumps(src["edges"], ensure_ascii=False), sid))
            conn.commit()
            return True

    def update(self, nid, **kwargs):
        n = self.get(nid)
        if not n:
            return False
        with self._write_lock:
            conn = self._get_conn()
            set_clauses = []
            params = []
            for key, val in kwargs.items():
                # v8.9.7: 列名白名单校验，防 SQL 注入
                if key not in self._ALLOWED_COLUMNS:
                    import logging as _logging
                    _logging.getLogger("zhiluo.engine").warning(
                        "update() rejected column: %s", key)
                    raise ValidationError(f"Invalid column for update: {key}")
                if key in ("tags",):
                    val = json.dumps(val, ensure_ascii=False)
                n[key] = val
                set_clauses.append(f"{key}=?")
                params.append(val)
            if not set_clauses:
                return False
            params.append(nid)
            conn.execute(f"UPDATE nodes SET {', '.join(set_clauses)} WHERE id=?", params)
            conn.commit()
            return True

    # === 超级版新增: pending待确认队列 (来自左脑v3.0) ===
    def add_pending(self, text, category=None, source="auto"):
        """添加待确认知识"""
        pid = uuid.uuid4().hex[:8]
        conn = self._get_conn()
        conn.execute(
            "INSERT INTO pending (id,content,category,workspace,source,created_at) VALUES (?,?,?,?,?,?)",
            (pid, text.strip(), category, self.ws, source, datetime.now().isoformat())
        )
        conn.commit()
        return pid

    def pending_list(self):
        """列出待确认知识"""
        conn = self._get_conn()
        rows = conn.execute("SELECT * FROM pending WHERE workspace=? ORDER BY created_at DESC", (self.ws,)).fetchall()
        return [dict(r) for r in rows]

    def pending_confirm(self, pid, promote=True):
        """确认待确认知识，升级为正式知识"""
        conn = self._get_conn()
        row = conn.execute("SELECT * FROM pending WHERE id=?", (pid,)).fetchone()
        if not row:
            return None
        row = dict(row)
        # v8.9.7: 先执行 add()，成功后再删除 pending 行，防止 add 失败导致数据丢失
        if promote:
            nid = self.add(
                row["content"],
                self.ws,
                category=row.get("category") or None,
                source=row.get("source") or "auto",
                session_id="pending-" + pid,
            )
            try:
                source_url = row.get("source_url") or ""
                if not source_url and str(row.get("source") or "").lower().startswith(("http://", "https://")):
                    source_url = row.get("source") or ""
                content_hash = row.get("content_hash") or ""
                if not content_hash:
                    normalized = re.sub(r"https?://\S+", "", row.get("content") or "", flags=re.I)
                    normalized = re.sub(r"\b\d{4}[-/年]\d{1,2}[-/月]\d{1,2}日?\b", "", normalized)
                    normalized = re.sub(r"\b\d{1,2}:\d{2}(:\d{2})?\b", "", normalized)
                    normalized = re.sub(r"\s+", " ", normalized).strip().lower()
                    content_hash = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]
                updates = []
                params = []
                cols = {r[1] for r in conn.execute("PRAGMA table_info(nodes)").fetchall()}
                if source_url and "source_url" in cols:
                    updates.append("source_url=?")
                    params.append(source_url)
                if content_hash and "content_hash" in cols:
                    updates.append("content_hash=?")
                    params.append(content_hash)
                if updates:
                    updates.append("updated_at=?")
                    params.extend([datetime.now().isoformat(), nid])
                    conn.execute("UPDATE nodes SET %s WHERE id=?" % ", ".join(updates), params)
                    conn.commit()
            except Exception:
                import logging as _logging; _logging.getLogger("zhiluo.engine").warning("engine.py: swallowed exception", exc_info=True)
            # v8.9.7: add 成功后才删除 pending，防止 add 失败导致数据丢失
            conn.execute("DELETE FROM pending WHERE id=?", (pid,))
            conn.commit()
            return nid
        return None

    def pending_reject(self, pid):
        """拒绝待确认知识"""
        conn = self._get_conn()
        conn.execute("DELETE FROM pending WHERE id=?", (pid,))
        conn.commit()
        return True

    # === 超级版新增: 变更历史 (来自左脑v3.0) ===
    def log_change(self, node_id, action, old_text="", new_text=""):
        """记录节点变更历史"""
        conn = self._get_conn()
        conn.execute(
            "INSERT INTO change_log (node_id, action, old_text, new_text, timestamp) VALUES (?,?,?,?,?)",
            (node_id, action, old_text[:200], new_text[:200], datetime.now().isoformat())
        )
        conn.commit()

    def change_history(self, node_id=None, limit=20):
        """查询变更历史"""
        conn = self._get_conn()
        if node_id:
            rows = conn.execute("SELECT * FROM change_log WHERE node_id=? ORDER BY timestamp DESC LIMIT ?", (node_id, limit)).fetchall()
        else:
            rows = conn.execute("SELECT * FROM change_log ORDER BY timestamp DESC LIMIT ?", (limit,)).fetchall()
        return [dict(r) for r in rows]

# ====================== 超级版新增: entangle 纠缠场 (来自左脑v3.0) ======================
def entangle(store, words, workspace=None):
    """纠缠场: 找出多个关键词之间的隐性关联路径"""
    if len(words) < 2:
        return {"words": words, "entanglements": []}
    node_sets = []
    for w in words:
        found = find(store, w, workspace)
        node_sets.append({n["id"] for n in found})
    pairs = []
    for i in range(len(words)):
        for j in range(i + 1, len(words)):
            common = node_sets[i] & node_sets[j]
            # 通过图谱找间接关联
            indirect = set()
            for nid in node_sets[i]:
                for e in store.adj.get(nid, []):
                    if e["target"] in node_sets[j]:
                        indirect.add((nid, e["target"], e.get("type", "related")))
            pairs.append({
                "word_a": words[i],
                "word_b": words[j],
                "common_nodes": len(common),
                "indirect_links": [{"from": a, "to": b, "type": t} for a, b, t in list(indirect)[:10]],
            })
    return {"words": words, "entanglements": pairs}

# ====================== 超级版新增: 自动学习 (来自左脑v3.0) ======================
def auto_learn(store, text, workspace=None):
    """从自然语言文本中自动提取可能的知识点，放入pending队列等待确认"""
    # 提取句子
    sentences = [s.strip() for s in re.split(r"[。！？!?\n;；]+", text) if len(s.strip()) > 5]
    pending_ids = []
    for sent in sentences:
        # 过滤掉问句（大概率不是知识）
        if sent.endswith("?") or sent.endswith("？"):
            continue
        # 过滤掉太短的
        if len(sent) < 6:
            continue
        pid = store.add_pending(sent)
        pending_ids.append(pid)
    return pending_ids

# ====================== 超级版新增: 上下文注入 (来自左脑v3.0) ======================
def context_inject(store, query, max_items=5, workspace=None):
    """根据查询注入相关上下文，帮助AI理解用户意图"""
    results = find(store, query, workspace)
    if not results:
        return ""
    lines = [f"[上下文] 找到 {len(results)} 条相关知识:"]
    for n in results[:max_items]:
        tags = ", ".join(n.get("tags", [])) if n.get("tags") else ""
        tag_str = f" [{tags}]" if tags else ""
        lines.append(f"  #{n['id']} [{n['category']}]{tag_str} {n['text'][:100]}")
    return "\n".join(lines)

# ====================== 检索 ======================
def find(store, query, workspace=None):
    results = []
    seen = set()
    sh = SimHash.hash(query)
    nid = store.hash_idx.get(sh)
    if nid:
        store.touch(nid)
        return [store.get(nid)]
    for kw in SimHash.tokens(query):
        for nid in store.kw_idx.get(kw, []):
            n = store.get(nid)
            if n and n['id'] not in seen:
                seen.add(n['id'])
                results.append(n)
    for nid in store.sh_buckets.get(SimHash.bucket(sh), []):
        n = store.get(nid)
        if n and n['id'] not in seen:
            d = SimHash.hamming(sh, n.get('simhash', 0))
            if d <= 5:
                seen.add(n['id'])
                results.append(n)
        if len(results) >= 10:
            break
    # 优化4: MinHashLSH 补充近似匹配
    for nid in store.lsh.query(query):
        n = store.get(nid)
        if n and n['id'] not in seen:
            jac = store.lsh.jaccard(query, n.get('text', ''))
            if jac > 0.3:
                n['_lsh_score'] = round(jac, 4)
                seen.add(n['id'])
                results.append(n)
    if len(results) < 3:
        try:
            from vector import vec_search, _ensure_vec
            conn = store._get_conn()
            if _ensure_vec(conn):
                for nid, dist in vec_search(conn, query, top_k=5):
                    if nid not in seen:
                        n = store.get(nid)
                        if n:
                            n['_semantic'] = round(max(0, 1 - dist), 4)
                            seen.add(nid)
                            results.append(n)
        except Exception:
            import logging as _logging; _logging.getLogger("zhiluo.engine").warning("engine.py: swallowed exception", exc_info=True)
    if len(results) < 3:
        try:
            from vector import semantic_find
            for n in semantic_find(store, query, workspace, top_k=5):
                if n['id'] not in seen:
                    seen.add(n['id'])
                    results.append(n)
        except ImportError:
            pass
    if not results:
        for n in store.valid(workspace):
            if n['id'] not in seen and query in n.get('text', ''):
                seen.add(n['id'])
                results.append(n)
    _now = datetime.now()
    for r in results:
        la = r.get('last_accessed_at') or r.get('learned_at', '')
        try:
            days = (_now - datetime.fromisoformat(la[:19])).days if la else 365
        except Exception:
            days = 365
        r['_time'] = r.get('access_count', 0) / max(days, 1)
    results.sort(key=lambda n: (n.get('_semantic', 0), n.get('_lsh_score', 0), n.get('_time', 0)), reverse=True)
    for n in results[:5]:
        store.touch(n['id'])
    return results[:10]

def diffuse(store, query, workspace=None, hops=2):
    seeds = find(store, query, workspace)
    if not seeds:
        return []
    sids = {n["id"] for n in seeds}
    visited = {sid: 0.0 for sid in sids}
    q = deque([(sid, 0, 1.0) for sid in sids])
    while q:
        nid, hop, conf = q.popleft()
        if hop >= hops:
            continue
        for e in store.adj.get(nid, []):
            t, w = e["target"], e.get("weight", 1.0)
            nc = round(conf * w * (0.7 ** hop), 4)
            if t not in visited or nc > visited[t]:
                visited[t] = nc
                q.append((t, hop + 1, nc))
    result = []
    for nid, conf in sorted(visited.items(), key=lambda x: -x[1]):
        if nid in sids:
            continue
        n = store.get(nid)
        if n:
            n["_c"] = conf
            result.append(n)
    return result[:15]

def pagerank(store, damping=0.85, iterations=20, exclude_categories=None):
    if exclude_categories is None:
        exclude_categories = {"测试数据"}
    nodes = [
        n for n in store.valid()
        if (n.get("category") or "") not in exclude_categories
        and (n.get("status") or "active") not in {"archived", "merged"}
    ]
    if not nodes:
        return []
    nids = [n["id"] for n in nodes]
    n = len(nids)
    if n == 1:
        return [{"id": nids[0], "pr": 1.0}]
    out = {nid: set() for nid in nids}
    inn = {nid: set() for nid in nids}
    for nid in nids:
        for e in store.adj.get(nid, []):
            if e["target"] in out:
                out[nid].add(e["target"])
                inn[e["target"]].add(nid)
    pr = {nid: 1.0 / n for nid in nids}
    for _ in range(iterations):
        new = {}
        for nid in nids:
            s = sum(pr[src] / max(len(out[src]), 1) for src in inn[nid])
            new[nid] = (1 - damping) / n + damping * s
        pr = new
    res = [{"id": nid, "pr": round(pr[nid], 6)} for nid in nids]
    res.sort(key=lambda x: -x["pr"])
    return res

def auto_tfidf(store, text, top_k=5):
    # 优化1: jieba 关键词抽取优先
    if _HAS_JIEBA:
        kws = smart_extract_keywords(text, top_k)
        return {"tags": kws, "scores": {}}
    words = [t for t in SimHash.tokens(text) if len(t) >= 2]
    if not words:
        return {"tags": []}
    tf = Counter(words)
    max_tf = max(tf.values())
    total = max(len(store.valid()), 1)
    idf = {w: math.log((total + 1) / (len(store.kw_idx.get(w, [])) + 1)) + 1 for w in tf}
    scores = {w: round(tf[w] / max_tf * idf[w], 4) for w in tf}
    ranked = sorted(scores.items(), key=lambda x: -x[1])[:top_k]
    return {"tags": [w for w, _ in ranked], "scores": dict(ranked)}

def decay(store, half_life=30, dry_run=True):
    now = datetime.now()
    lam = math.log(2) / half_life
    updates = []
    for n in store.valid():
        ls = n.get("last_accessed_at") or n.get("learned_at", "")
        try:
            ld = datetime.fromisoformat(ls[:19])
        except Exception:
            continue
        d = (now - ld).total_seconds() / 86400
        if d <= 0:
            continue
        bonus = min(1.0, n.get("access_count", 0) / 20)
        nc = max(0.1, n["confidence"] * math.exp(-lam * d) + bonus * 0.3)
        if abs(nc - n["confidence"]) > 0.01:
            updates.append({"id": n["id"], "text": n["text"][:40], "old": round(n["confidence"], 4), "new": round(nc, 4), "days": round(d, 1)})
            if not dry_run:
                n["confidence"] = round(nc, 4)
    if not dry_run:
        store.save()
    return {"mode": "dry_run" if dry_run else "applied", "checked": len(store.valid()), "decayed": len(updates), "updates": updates[:20]}

_OPPOSITE_PAIRS = [
    ("涨", "跌"), ("升", "降"), ("增", "减"), ("加", "减"),
    ("多", "少"), ("大", "小"), ("高", "低"), ("上", "下"),
    ("好", "坏"), ("优", "劣"), ("新", "旧"), ("正", "反"),
    ("买", "卖"), ("进", "出"), ("盈", "亏"), ("利", "弊"),
    ("支持", "反对"), ("同意", "拒绝"), ("开启", "关闭"),
    ("上涨", "下跌"), ("上升", "下降"), ("增加", "减少"),
    ("提高", "降低"), ("扩大", "缩小"), ("加强", "削弱"),
    ("促进", "抑制"), ("繁荣", "衰退"), ("增长", "下滑"),
    ("涨价", "降价"), ("增多", "减少"), ("变大", "变小"), ("变好", "变坏"),
]
_OPPOSITE_SET = set()
for a, b in _OPPOSITE_PAIRS:
    _OPPOSITE_SET.add(a)
    _OPPOSITE_SET.add(b)


def conflicts(store):
    """三重冲突检测
    L1: 显式矛盾边 - 遍历 adj 找 EDGE_CONTRADICTS
    L2: 语义对立 - SimHash 相近文本中存在反义词对
    L3: 传播冲突 - cause 边和 contradicts 边指向同一实体
    """
    cs = []
    seen_pairs = set()
    nodes = store.valid()

    # L1: 显式矛盾边
    for n in nodes:
        for e in store.adj.get(n['id'], []):
            if e.get('type') == EDGE_CONTRADICTS:
                t = store.get(e['target'])
                if t:
                    pair = (n['id'], e['target'])
                    if pair not in seen_pairs and (e['target'], n['id']) not in seen_pairs:
                        seen_pairs.add(pair)
                        cs.append({'layer': 'L1', 'node': n['text'][:40],
                                   'vs': t['text'][:40], 'type': '显式矛盾边'})

    # L2: SimHash 语义对立（Bug5修复：桶预筛 + 节点上限，避免O(n²)爆炸）
    # 先按SimHash桶分组，只比较同桶节点（汉明距离<=12大概率在同桶或相邻桶）
    try:
        _L2_MAX_NODES = max(0, int(os.environ.get("ZHILUO_CONFLICT_L2_MAX_NODES", "500") or "500"))
    except Exception:
        _L2_MAX_NODES = 500
    if os.environ.get("ZHILUO_CONFLICT_L2_ENABLE", "1") == "0" or _L2_MAX_NODES == 0:
        l2_nodes = []
    else:
        l2_nodes = nodes if len(nodes) <= _L2_MAX_NODES else nodes[:_L2_MAX_NODES]
    # 建桶索引：同一桶内的节点才需要两两比较
    bucket_map = defaultdict(list)
    for n in l2_nodes:
        sh = n.get('simhash', 0)
        if sh:
            bucket_map[SimHash.bucket(sh)].append(n)
    # 还需检查相邻桶（汉明距离12可能跨桶）
    checked_buckets = set()
    for bucket_id, bucket_nodes in bucket_map.items():
        if bucket_id in checked_buckets:
            continue
        checked_buckets.add(bucket_id)
        # 当前桶 + 相邻桶（±1, ±2... 共16个桶都检查也不贵）
        candidates = list(bucket_nodes)
        for adj_bid in [(bucket_id + d) % 16 for d in range(1, 4)]:
            if adj_bid in bucket_map:
                candidates.extend(bucket_map[adj_bid])
        # 两两比较（同桶+近桶候选集）
        for i in range(len(candidates)):
            for j in range(i + 1, len(candidates)):
                a, b = candidates[i], candidates[j]
                sh_a = a.get('simhash', 0)
                sh_b = b.get('simhash', 0)
                if not sh_a or not sh_b:
                    continue
                d = SimHash.hamming(sh_a, sh_b)
                if d <= 12:
                    text_a = a.get('text', '')
                    text_b = b.get('text', '')
                    tokens_a = set(SimHash.tokens(text_a))
                    tokens_b = set(SimHash.tokens(text_b))
                    ops_in_a = tokens_a & _OPPOSITE_SET
                    ops_in_b = tokens_b & _OPPOSITE_SET
                    for op_a in ops_in_a:
                        for op_b in ops_in_b:
                            if (op_a, op_b) in _OPPOSITE_PAIRS or (op_b, op_a) in _OPPOSITE_PAIRS:
                                pair = (a['id'], b['id'])
                                if pair not in seen_pairs and (b['id'], a['id']) not in seen_pairs:
                                    seen_pairs.add(pair)
                                    cs.append({
                                        'layer': 'L2', 'node': a['text'][:40],
                                        'vs': b['text'][:40],
                                        'type': '语义对立 (%s <-> %s)' % (op_a, op_b)
                                    })
                                break

    # L3: 传播冲突
    source_map = {}
    for n in nodes:
        for e in store.adj.get(n['id'], []):
            etype = e.get('type', '')
            if etype in (EDGE_CAUSE, EDGE_CONTRADICTS):
                key = e['target']
                if key not in source_map:
                    source_map[key] = {'cause': [], 'contradicts': []}
                if etype == EDGE_CAUSE:
                    source_map[key]['cause'].append(n['id'])
                else:
                    source_map[key]['contradicts'].append(n['id'])

    for target_id, sources in source_map.items():
        if sources['cause'] and sources['contradicts']:
            for cid in sources['cause']:
                for did in sources['contradicts']:
                    pair = (cid, did)
                    if pair not in seen_pairs and (did, cid) not in seen_pairs:
                        seen_pairs.add(pair)
                        cn = store.get(cid)
                        dn = store.get(did)
                        tn = store.get(target_id)
                        if cn and dn and tn:
                            cs.append({
                                'layer': 'L3',
                                'node': '%s -> %s' % (cn['text'][:30], tn['text'][:30]),
                                'vs': '%s -> %s' % (dn['text'][:30], tn['text'][:30]),
                                'type': '传播冲突 (cause vs contradicts)'
                            })

    return {"total": len(cs), "conflicts": cs[:20]}

# ====================== 优化5: LLM 记忆合并 ======================
def consolidate(store, old_text, new_text, llm_func=None):
    """智能合并两条相似记忆。有 LLM 时用 LLM 合并，无 LLM 时用规则合并。"""
    if llm_func:
        prompt = f"请将以下两条知识合并为一条，保留关键信息，去除冗余：\n旧: {old_text}\n新: {new_text}\n合并结果:"
        try:
            merged = llm_func(prompt)
            if merged and len(merged) > 5:
                return merged.strip()
        except Exception:
            import logging as _logging; _logging.getLogger("zhiluo.engine").warning("engine.py: swallowed exception", exc_info=True)
    # 规则合并：新文本追加旧文本中独有的信息
    old_toks = set(SimHash.tokens(old_text))
    new_toks = set(SimHash.tokens(new_text))
    unique_old = [t for t in old_toks if t not in new_toks]
    if unique_old:
        extra = "；旧记录补充：" + "".join(unique_old[:10])
        return new_text + extra
    return new_text

# ====================== 优化6: Mermaid 图可视化 ======================
def mermaid_graph(store, keyword=None, max_nodes=20, workspace=None):
    """生成 Mermaid 格式知识图谱"""
    nodes = store.valid(workspace)
    if keyword:
        seeds = find(store, keyword, workspace)
        seed_ids = {n["id"] for n in seeds}
        related_ids = set()
        for sid in seed_ids:
            for e in store.adj.get(sid, []):
                related_ids.add(e["target"])
        node_ids = seed_ids | related_ids
        nodes = [n for n in nodes if n["id"] in node_ids]
    else:
        nodes = nodes[:max_nodes]
    if not nodes:
        return "graph TD\n  (empty)"
    lines = ["graph TD"]
    node_ids = {n["id"] for n in nodes}
    for n in nodes:
        label = n["text"][:30].replace('"', "'")
        cat = n.get("category", "未分类")
        lines.append(f'  N{n["id"]}["[{cat}] {label}"]')
    edge_styles = {
        EDGE_RELATED: "---",
        EDGE_CAUSE: "-->",
        EDGE_PART_OF: "-.-",
        EDGE_SYNONYM: "===",
        EDGE_REFINES: "-..->",
        EDGE_CONTRADICTS: "--x",
    }
    seen_edges = set()
    for n in nodes:
        for e in n.get("edges", []):
            tid = e.get("target")
            if tid in node_ids:
                pair = tuple(sorted([n["id"], tid]))
                if pair not in seen_edges:
                    seen_edges.add(pair)
                    style = edge_styles.get(e.get("type", EDGE_RELATED), "---")
                    etype = e.get("type", "related")
                    lines.append(f'  N{n["id"]} {style}|{etype}| N{tid}')
    return "\n".join(lines)

# ====================== 推理 ======================
def analyze(text):
    nums = re.findall(r"-?\d+\.?\d*%?", text)
    up = sum(text.count(w) for w in ["增长", "上升", "提高", "增加", "涨", "升"])
    dn = sum(text.count(w) for w in ["下降", "降低", "减少", "跌", "降"])
    vals = [float(n.rstrip("%")) for n in nums if n.replace(".", "").replace("-", "").replace("%", "").isdigit()]
    st = {}
    if vals:
        st = {"count": len(vals), "sum": round(sum(vals), 2), "avg": round(sum(vals) / len(vals), 2), "max": max(vals), "min": min(vals)}
    return {"numbers": nums[:15], "stats": st, "trend": "上升" if up > dn else ("下降" if dn > up else "平稳")}

def summarize(text, ratio=0.3):
    ss = [s.strip() for s in re.split(r"[。！？!?\n]+", text) if len(s.strip()) > 5]
    if not ss:
        return {"summary": text[:200], "key_points": []}
    toks = SimHash.tokens(text)
    freq = Counter(toks)
    sc = [(sum(freq.get(t, 0) for t in SimHash.tokens(s)) / max(len(SimHash.tokens(s)), 1) + 0.5 / (i + 1), s) for i, s in enumerate(ss)]
    sc.sort(reverse=True)
    keep = max(1, int(len(ss) * ratio))
    return {"summary": "。".join([s for _, s in sc[:3]]) + "。", "key_points": [s for _, s in sc[:keep]], "total": len(ss)}

def causal(store, keyword, workspace=None):
    seeds = find(store, keyword, workspace)
    if not seeds:
        return {"chain": [], "msg": "未找到"}
    chain = []
    visited = set()
    q = deque([seeds[0]])
    while q:
        n = q.popleft()
        if n["id"] in visited:
            continue
        visited.add(n["id"])
        chain.append({"text": n["text"], "category": n["category"], "id": n["id"]})
        for e in store.adj.get(n["id"], []):
            if e.get("type") == EDGE_CAUSE:
                t = store.get(e["target"])
                if t and t["id"] not in visited:
                    q.append(t)
    return {"chain": chain, "length": len(chain)}

def timeline(store, keyword=None, workspace=None):
    nodes = store.valid(workspace)
    if keyword:
        nodes = [n for n in nodes if keyword in n["text"]]
    nodes.sort(key=lambda n: n.get("learned_at", ""))
    return [{"time": n["learned_at"][:19], "text": n["text"], "category": n["category"]} for n in nodes]

# ====================== 感知 ======================
class Intent:
    @staticmethod
    def classify(text):
        t = text.strip()
        if any(k in t for k in ["记住", "记一下", "帮我记", "学习", "存", "记录"]):
            return "learn"
        # === 超级版新增: 自动学习意图 ===
        if any(k in t for k in ["自动学习", "自动提取", "从这段话", "读一下这段"]):
            return "auto_learn"
        # === 超级版新增: pending管理 ===
        if any(k in t for k in ["待确认", "待审", "pending"]):
            return "pending"
        if t.startswith("确认") or t.startswith("confirm"):
            return "confirm"
        # === 超级版新增: entangle ===
        if any(k in t for k in ["纠缠", "关联场", "entangle"]):
            return "entangle"
        if t.startswith("修改") or "改成" in t:
            return "update"
        if t.startswith("删除") or t.startswith("忘掉"):
            return "delete"
        if any(k in t for k in ["分析", "数据"]):
            return "analyze"
        if any(k in t for k in ["总结", "摘要", "概括"]):
            return "summarize"
        if any(k in t for k in ["纠正", "纠错", "是什么意思", "是什么"]) or "是什么意思" in t:
            return "correct"
        if any(k in t for k in ["追溯", "来源", "时间线", "因果"]):
            return "trace"
        if any(k in t for k in ["关联", "跳", "扩散", "推荐"]):
            return "search"
        if any(k in t for k in ["可视化", "图谱", "图", "graph", "mermaid"]):
            return "visualize"
        if any(k in t.lower() for k in ["pagerank", "排序", "枢纽", "重要性"]):
            return "pagerank"
        if any(k in t for k in ["仪表盘", "工作区", "自检", "备份", "导出", "记忆列表", "全局搜索"]):
            return "manage"
        if any(k in t for k in ["什么时候", "怎么", "在哪", "多少", "？", "?"]):
            return "query"
        return "query"

# ====================== 自检 ======================
class SelfCheck:
    def __init__(self, store):
        self.st = store
        self.r = []

    def run(self):
        for fn in [self._db, self._sh, self._edge, self._etype, self._empty, self._ws, self._bk, self._kw, self._cat, self._cnt, self._lsh, self._jieba, self._pending, self._changelog]:
            try:
                self.r.append(fn())
            except Exception as e:
                self.r.append(f"[X] {fn.__doc__}: {e}")
        p = sum(1 for r in self.r if r.startswith("[OK]"))
        w = sum(1 for r in self.r if r.startswith("[W]"))
        # === 超级版新增: 自动修复 ===
        fixes = self._auto_repair()
        fix_str = f"\n [修复] {', '.join(fixes)}" if fixes else ""
        return {"report": "\n".join(["[Z] 知络 v8.9.1 自检", "=" * 30] + self.r + ["", f" [OK]{p} [W]{w} [X]{len(self.r) - p - w}{fix_str}"]), "passed": p, "warned": w}

    def _db(self):
        t = len(self.st.valid())
        return f"[OK] 1.数据库: {t}条"

    def _sh(self):
        b = sum(1 for n in self.st.valid() if not n.get("simhash", 0))
        return f"[OK] 2.SimHash: 全部一致" if b == 0 else f"[W] SimHash: {b}个缺失"

    def _edge(self):
        t = sum(len(n.get("edges", [])) for n in self.st.valid())
        return f"[OK] 3.边: {t}条"

    def _etype(self):
        ts = set()
        [ts.add(e.get("type")) for n in self.st.valid() for e in n.get("edges", []) if e.get("type") in EDGE_TYPES]
        return f"[OK] 4.边类型: {ts or '无'}"

    def _empty(self):
        e = sum(1 for n in self.st.valid() if not n.get("text", "").strip())
        return f"[OK] 5.空节点: 无" if e == 0 else f"[W] 空节点: {e}个"

    def _ws(self):
        ws = Counter(n["workspace"] for n in self.st.valid())
        return f"[OK] 6.工作区: {', '.join(f'{k}:{v}' for k, v in ws.most_common()) or '无'}"

    def _bk(self):
        bs = list(DATA_DIR.glob("backup_*.db"))
        if not bs:
            return "[W] 7.备份: 无"
        d = (datetime.now() - datetime.fromtimestamp(max(b.stat().st_mtime for b in bs))).days
        return f"[OK] 7.备份: {d}天前" if d < 30 else f"[W] 备份: {d}天前"

    def _kw(self):
        t = sum(len(v) for v in self.st.kw_idx.values())
        return f"[OK] 8.关键词索引: {len(self.st.kw_idx)}个词/{t}次命中"

    def _cat(self):
        c = Counter(n["category"] for n in self.st.valid())
        return f"[OK] 9.分类: {', '.join(f'{k}:{v}' for k, v in c.most_common()) or '无'}"

    def _cnt(self):
        return f"[OK] 10.知识: {len(self.st.valid())}条"

    def _lsh(self):
        """MinHashLSH 去重索引"""
        return f"[OK] 11.MinHashLSH: {len(self.st.lsh._bands)}个桶"

    def _jieba(self):
        """jieba 分词状态"""
        return f"[OK] 12.jieba: {'已启用' if _HAS_JIEBA else '未安装(回退单字)'}"

    def _pending(self):
        """超级版: pending 待确认队列检查"""
        items = self.st.pending_list()
        if not items:
            return "[OK] 13.待确认: 无"
        return f"[W] 13.待确认: {len(items)}条未处理"

    def _changelog(self):
        """超级版: 变更历史检查"""
        hist = self.st.change_history(limit=1)
        if not hist:
            return "[OK] 14.变更历史: 无"
        return f"[OK] 14.变更历史: 最近变更 {hist[0]['timestamp'][:10]}"

    def _auto_repair(self):
        """超级版: 8项自动修复 (对齐左脑v3.0)"""
        fixes = []
        conn = self.st._get_conn()

        # 修复1: 清理空白待确认
        try:
            empty = conn.execute("SELECT COUNT(*) as c FROM pending WHERE TRIM(content)=''").fetchone()["c"]
            if empty:
                conn.execute("DELETE FROM pending WHERE TRIM(content)=''")
                conn.commit()
                fixes.append(f"清理{empty}条空白待确认")
        except Exception:
            import logging as _logging; _logging.getLogger("zhiluo.engine").warning("engine.py: swallowed exception", exc_info=True)

        # 修复2: 补全缺失的SimHash
        missing = 0
        for n in self.st.valid():
            if not n.get("simhash"):
                n["simhash"] = SimHash.hash(n["text"])
                conn.execute("UPDATE nodes SET simhash=? WHERE id=?", (n["simhash"], n["id"]))
                missing += 1
        if missing:
            conn.commit()
            self.st.save()
            fixes.append(f"补全{missing}个SimHash")

        # 修复3: 清理空文本节点
        empties = [n for n in self.st.valid() if not n.get("text", "").strip()]
        if empties:
            for n in empties:
                self.st.delete(n["id"])
            self.st.save()
            fixes.append(f"删除{len(empties)}个空节点")

        # 修复4: FTS5索引同步重建
        try:
            fts_count = conn.execute("SELECT COUNT(*) as c FROM nodes_fts").fetchone()["c"]
            node_count = conn.execute("SELECT COUNT(*) as c FROM nodes").fetchone()["c"]
            if abs(fts_count - node_count) > 0:
                conn.execute("INSERT INTO nodes_fts(nodes_fts) VALUES('rebuild')")
                conn.commit()
                fixes.append(f"FTS5索引已重建(差{abs(fts_count-node_count)})")
        except Exception:
            import logging as _logging; _logging.getLogger("zhiluo.engine").warning("engine.py: swallowed exception", exc_info=True)

        # 修复5: 去重重复边 (同一source+target+type出现多次)
        try:
            dup_count = 0
            for n in self.st.valid():
                edges = n.get("edges", [])
                seen = set()
                unique_edges = []
                for e in edges:
                    key = (e.get("target"), e.get("type", EDGE_RELATED))
                    if key in seen:
                        dup_count += 1
                    else:
                        seen.add(key)
                        unique_edges.append(e)
                if len(unique_edges) != len(edges):
                    n["edges"] = unique_edges
                    conn.execute("UPDATE nodes SET edges_json=? WHERE id=?",
                                 (json.dumps(unique_edges, ensure_ascii=False), n["id"]))
            if dup_count:
                conn.commit()
                self.st.save()
                fixes.append(f"去重{dup_count}条重复边")
        except Exception:
            import logging as _logging; _logging.getLogger("zhiluo.engine").warning("engine.py: swallowed exception", exc_info=True)

        # 修复6: 删除自环边 (节点指向自己)
        try:
            self_loop_count = 0
            for n in self.st.valid():
                nid = n["id"]
                edges = n.get("edges", [])
                filtered = [e for e in edges if e.get("target") != nid]
                if len(filtered) != len(edges):
                    self_loop_count += len(edges) - len(filtered)
                    n["edges"] = filtered
                    conn.execute("UPDATE nodes SET edges_json=? WHERE id=?",
                                 (json.dumps(filtered, ensure_ascii=False), nid))
            if self_loop_count:
                conn.commit()
                self.st.save()
                fixes.append(f"删除{self_loop_count}条自环边")
        except Exception:
            import logging as _logging; _logging.getLogger("zhiluo.engine").warning("engine.py: swallowed exception", exc_info=True)

        # 修复7: 清理孤儿变更历史 (引用了已删除节点的历史记录)
        try:
            valid_ids = {n["id"] for n in self.st.valid()}
            orphan_hist = conn.execute("SELECT id, node_id FROM change_log").fetchall()
            orphan_ids = [row["id"] for row in orphan_hist if row["node_id"] not in valid_ids]
            if orphan_ids:
                placeholders = ",".join("?" * len(orphan_ids))
                conn.execute(f"DELETE FROM change_log WHERE id IN ({placeholders})", orphan_ids)
                conn.commit()
                fixes.append(f"清理{len(orphan_ids)}条孤儿历史")
        except Exception:
            import logging as _logging; _logging.getLogger("zhiluo.engine").warning("engine.py: swallowed exception", exc_info=True)

        # 修复8: WAL文件压缩 (超过10MB时checkpoint)
        try:
            wal_path = self.st.path.with_suffix(".db-wal")
            if wal_path.exists():
                size_mb = wal_path.stat().st_size / 1024 / 1024
                if size_mb > 10:
                    conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                    conn.commit()
                    fixes.append(f"WAL已压缩({size_mb:.1f}MB)")
        except Exception:
            import logging as _logging; _logging.getLogger("zhiluo.engine").warning("engine.py: swallowed exception", exc_info=True)

        return fixes

# ====================== 授权接口（开源版空实现）======================
def license_status():
    return {"status": "ok", "msg": "开源版，无需激活"}

def activate_license(code=None):
    return {"status": "ok", "msg": "开源版，无需激活"}

# ====================== 主入口 ======================
class ZhiLuo:
    def __init__(self, path=None, workspace="global"):
        workspace = validate_workspace_name(workspace)
        self.s = MemoryStore(path, workspace=workspace)
        self.ws = workspace
        self.session_id = datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + os.urandom(2).hex()
        self._llm_func = None
        # v8.3: 初始化大脑包装层
        try:
            from brain_wrapper import create_brain
            self.brain = create_brain(self.s, workspace=workspace)
        except Exception as e:
            self.brain = None

    def set_llm(self, func):
        """设置 LLM 合并函数（可选）"""
        self._llm_func = func

    def _extract_and_link(self, nid, text):
        """从文本提取关系并建立语义边（有LLM走LLM，无LLM降级规则）"""
        try:
            if self._llm_func:
                from relation_extractor import extract_with_llm
                rels = extract_with_llm(text, self._llm_func)
            else:
                from relation_extractor import extract_relations
                rels = extract_relations(text)
            if not rels:
                return
            for r in rels:
                src_name = r.get("source", "")
                tgt_name = r.get("target", "")
                rtype = r.get("relation", "related")
                weight = r.get("weight", 0.5)
                if not src_name or not tgt_name or src_name == tgt_name:
                    continue
                src_id = self._find_node_by_text(src_name, exclude=nid)
                tgt_id = self._find_node_by_text(tgt_name, exclude=nid)
                if src_id and tgt_id:
                    self.s.add_edge(src_id, tgt_id, etype=rtype, weight=weight)
        except Exception:
            import logging as _logging; _logging.getLogger("zhiluo.engine").warning("engine.py: swallowed exception", exc_info=True)

    def _find_node_by_text(self, keyword, exclude=None):
        """按文本模糊查找节点，精确匹配优先"""
        best = None
        for n in self.s.valid():
            if exclude is not None and n["id"] == exclude:
                continue
            text = n.get("text", "")
            if text == keyword:
                return n["id"]
            if keyword in text and best is None:
                best = n["id"]
            try:
                import jieba
                for tok in jieba.lcut(keyword):
                    if tok in text and len(tok) >= 2:
                        if best is None:
                            best = n["id"]
            except Exception:
                import logging as _logging; _logging.getLogger("zhiluo.engine").warning("engine.py: swallowed exception", exc_info=True)
        return best

    def _sync_co_net(self, text):
        """写入时同步共现网络"""
        try:
            if self.brain and hasattr(self.brain, 'co_net'):
                self.brain.co_net.learn(text, source='learn')
        except Exception:
            import logging as _logging; _logging.getLogger("zhiluo.engine").warning("engine.py: swallowed exception", exc_info=True)

    def run(self, text):
        """旧通道：自然语言意图路由。保留向后兼容。"""
        return self._exec(Intent.classify(text), text)

    # v8.9.7: 新通道 — 结构化调用，语义化参数，可被 linter/IDE 检查
    def call(self, action, **kwargs):
        """新通道：显式 action + 关键字参数调用知络引擎。
        
        用法:
          lb.call("learn", text="一条知识")
          lb.call("query", keyword="搜索词")
          lb.call("search", keyword="关联词", mode="graph")
          lb.call("analyze", mode="conflicts")
          lb.call("summarize", text="长文本")
          lb.call("visualize", keyword="可选关键词")
          lb.call("delete", nid=123)
        
        旧通道 lb.run() 仍可用。新通道支持 IDE 自动补全和类型检查。
        """
        if action == "learn":
            return self._exec("learn", kwargs.get("text", ""))
        if action == "query":
            return self._exec("query", kwargs.get("keyword", ""))
        if action == "search":
            return self._exec("search", kwargs.get("keyword", ""))
        if action == "analyze":
            mode = kwargs.get("mode", "")
            if mode == "conflicts":
                return self._exec("analyze", "冲突")
            if mode == "decay":
                return self._exec("analyze", "衰减")
            return self._exec("analyze", kwargs.get("text", ""))
        if action == "summarize":
            return self._exec("summarize", kwargs.get("text", ""))
        if action == "visualize":
            return self._exec("visualize", kwargs.get("keyword", ""))
        if action == "update":
            return self._exec("update", kwargs.get("text", ""))
        if action == "delete":
            nid = kwargs.get("nid")
            if nid:
                self.s.delete(int(nid))
                self.s.save()
                return f"[Z] 已删除 #{nid}"
            return self._exec("delete", kwargs.get("text", ""))
        if action == "pagerank":
            return self._exec("pagerank", kwargs.get("text", ""))
        if action == "pending":
            return self._exec("pending", kwargs.get("text", ""))
        if action in ("confirm", "entangle", "auto_learn", "manage", "trace", "correct"):
            return self._exec(action, kwargs.get("text", ""))
        # 兜底：回退到旧通道
        return self._exec(action, kwargs.get("text", action))

    def _exec(self, a, args):
        try:
            if a == "learn":
                p = re.sub(r"^(帮我)?记住[：:]?|^(帮我)?记一下[：:]?|^(帮我)?学习[：:]?", "", args).strip()
                if not p:
                    return "[Z] 内容为空"
                sh = SimHash.hash(p)
                # 第一层：SimHash 精确去重
                for nid in self.s.sh_buckets.get(SimHash.bucket(sh), []):
                    n = self.s.get(nid)
                    if n and SimHash.hamming(sh, n.get("simhash", 0)) <= 2:
                        # 优化5: LLM 智能合并而非覆盖
                        merged = consolidate(self.s, n["text"], p, self._llm_func)
                        n["text"] = merged
                        n["updated_at"] = datetime.now().isoformat()
                        self.s.save()
                        self._sync_co_net(p)
                        return f"[Z] 已合并更新 #{nid}"
                # 第二层：MinHashLSH 近似去重（优化4）
                for candidate_nid in self.s.lsh.query(p):
                    n = self.s.get(candidate_nid)
                    if n:  # v8.9.7: 修复死代码 — 原 `n["id"] != nid` 恒为 False
                        jac = self.s.lsh.jaccard(p, n.get("text", ""))
                        if jac > 0.5:
                            merged = consolidate(self.s, n["text"], p, self._llm_func)
                            n["text"] = merged
                            n["updated_at"] = datetime.now().isoformat()
                            self.s.save()
                            self._sync_co_net(p)
                            return f"[Z] 已合并更新 #{n['id']} (相似度{jac:.0%})"
                nid = self.s.add(p, self.ws, session_id=self.session_id)
                if len(SimHash.tokens(p)) >= 2:
                    self.s.get(nid)["tags"] = auto_tfidf(self.s, p).get("tags", [])
                self.s.save()
                # ── 新增：关系提取+建边 ──
                self._extract_and_link(nid, p)
                self._sync_co_net(p)
                return f"[Z] 已记住 #{nid}（{self.s.get(nid)['category']}）"
            elif a == "query":
                r = find(self.s, args, self.ws)
                if not r:
                    return f"[Z] 未找到「{args}」。"
                # 优化2: 返回截断
                return "\n".join([f"[Z] {len(r)}条:"] + [f"  - #{n['id']} [{n['category']}] {n['text'][:80]}" for n in r])
            elif a == "search":
                kw = re.sub(r"^关联[搜索]?[：:]?|推荐|跳|扩散", "", args).strip()
                r = diffuse(self.s, kw, self.ws)
                if not r:
                    return f"[Z] 未找到「{kw}」的关联。"
                # 优化2: 返回截断
                return "\n".join([f"[Z] 「{kw}」关联:"] + [f"  - #{n['id']} [{n['category']}] {n['text'][:80]}  (c={n['_c']:.2f})" for n in r])
            elif a == "analyze":
                if "冲突" in args or "矛盾" in args:
                    r = conflicts(self.s)
                    if r["total"] == 0:
                        return "[Z] 未检测到冲突"
                    lines = [f"[Z] 冲突检测 ({r['total']}对):"]
                    for c in r["conflicts"]:
                        layer = c.get('layer', 'L1')
                        ctype = c.get('type', '')
                        lines.append(f"  ✗ [{layer}] {c['node']}  ↔  {c['vs']}  ({ctype})")
                    return "\n".join(lines)
                if "衰减" in args:
                    r = decay(self.s, dry_run=True)
                    lines = [f"[Z] 置信度衰减 (模式:{r['mode']}, 检查{r['checked']}条, 衰减{r['decayed']}条):"]
                    for u in r["updates"][:10]:
                        lines.append(f"  #{u['id']} {u['text']} | {u['old']}→{u['new']} ({u['days']}天)")
                    return "\n".join(lines)
                p = re.sub(r"^分析[：:]?", "", args).strip()
                r = analyze(p)
                lines = ["[Z] 分析:"]
                if r["numbers"]:
                    lines.append(f"  数字: {' '.join(r['numbers'])}")
                if r["stats"]:
                    s = r["stats"]
                    lines.append(f"  统计: {s['count']}个, 均{s['avg']}, 最大{s['max']}")
                lines.append(f"  趋势: {r['trend']}")
                return "\n".join(lines)
            elif a == "summarize":
                p = re.sub(r"^总结[：:]?|^摘要[：:]?", "", args).strip()
                r = summarize(p)
                return f"[Z] 总结 ({r.get('total', 0)}句->{len(r['key_points'])}点):\n  {r['summary']}"
            elif a == "correct":
                w = re.sub(r"纠正[：:]?|纠错[：:]?|是什么意思|是什么", "", args).strip().strip("'\"")
                for n in self.s.valid():
                    if w in n["text"]:
                        return f"[Z]「{w}」存在: {n['text'][:80]}"
                sh = SimHash.hash(w)
                best = None
                bd = 999
                for n in self.s.valid():
                    d = SimHash.hamming(sh, n.get("simhash", 0))
                    if d < bd:
                        bd = d
                        best = n
                if best and bd <= 20:
                    return f"[Z]「{w}」最接近: {best['text'][:80]}"
                return f"[Z] 未找到「{w}」"
            elif a == "trace":
                if "因果" in args:
                    kw = re.sub(r"因果|追溯", "", args).strip()
                    r = causal(self.s, kw, self.ws)
                    if not r["chain"]:
                        return f"[Z] 未找到因果链。"
                    return "\n".join([f"[Z] 因果链 ({r['length']}环):"] + [f"  {i+1}. [{n['category']}] {n['text'][:80]}" for i, n in enumerate(r["chain"])])
                if "时间" in args:
                    kw = re.sub(r"时间线|时间轴|追溯", "", args).strip()
                    tl = timeline(self.s, kw or None, self.ws)
                    if not tl:
                        return "[Z] 无时间戳知识。"
                    return "\n".join([f"[Z] 时间线 ({len(tl)}条):"] + [f"  {n['time'][:10]} [{n['category']}] {n['text'][:80]}" for n in tl])
                return "[Z] 追溯: 因果/时间线"
            elif a == "visualize":
                # 优化6: Mermaid 可视化
                kw = re.sub(r"可视化|图谱|图|graph|mermaid", "", args).strip()
                return mermaid_graph(self.s, kw or None, workspace=self.ws)
            elif a == "update":
                parts = args.split("|")
                if len(parts) != 2:
                    return "[Z] 格式: 旧文本|新文本"
                results = find(self.s, parts[0].strip(), self.ws)
                if not results:
                    return f"[Z] 未找到「{parts[0]}」。"
                n = results[0]
                n["text"] = parts[1].strip()
                n["updated_at"] = datetime.now().isoformat()
                self.s.save()
                self._sync_co_net(parts[1].strip())
                return f"[Z] 已更新 #{n['id']}"
            elif a == "delete":
                results = find(self.s, args.strip(), self.ws)
                if not results:
                    return f"[Z] 未找到「{args}」。"
                n = results[0]
                self.s.delete(n["id"])
                self.s.save()
                return f"[Z] 已删除 #{n['id']}"
            elif a == "pagerank":
                r = pagerank(self.s)
                if not r:
                    return "[Z] 无知识可排序"
                lines = ["[Z] PageRank 知识枢纽:"]
                for i, item in enumerate(r[:10]):
                    n = self.s.get(item["id"])
                    if n:
                        lines.append(f"  {i+1}. #{item['id']} (PR={item['pr']:.4f}) [{n['category']}] {n['text'][:60]}")
                return "\n".join(lines)
            elif a == "manage":
                if "仪表盘" in args:
                    st = self.s.stats()
                    return f"知识:{st['total']}条 边:{st['edges']}条\n" + ' '.join(f"{k}:{v}" for k, v in list(st['categories'].items())[:5])
                if "自检" in args:
                    return SelfCheck(self.s).run()["report"]
                if "备份" in args:
                    import shutil as _su
                    p = DATA_DIR / f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.db"
                    _su.copy2(self.s.path, p)
                    return f"[Z] 已备份 {p}"
                if "恢复" in args:
                    bs = sorted(DATA_DIR.glob("backup_*.db"))
                    if not bs:
                        return "[Z] 没有备份文件"
                    import shutil as _su2
                    _su2.copy2(str(bs[-1]), self.s.path)
                    self.s = MemoryStore(self.s.path, self.ws)
                    self.s.load()
                    return f"[Z] 已从 {Path(bs[-1]).name} 恢复"
                if "导出" in args:
                    fmt = "json"
                    if "csv" in args:
                        fmt = "csv"
                    elif "markdown" in args or "md" in args:
                        fmt = "markdown"
                    elif "graphml" in args:
                        fmt = "graphml"
                    return self._export(fmt)
                if "变更" in args or "历史" in args:
                    hist = self.s.change_history()
                    if not hist:
                        return "[Z] 无变更历史"
                    return "\n".join([f"[Z] 变更历史 ({len(hist)}条):"] + [f"  {h['timestamp'][:19]} #{h['node_id']} {h['action']}: {h.get('new_text','')[:40]}" for h in hist[:15]])
                if "工作区" in args:
                    ws = Counter(n["workspace"] for n in self.s.valid())
                    return "\n".join([f"工作区:"] + [f"  {k}: {v}条" for k, v in sorted(ws.items(), key=lambda x: -x[1])])
                return "管理: 仪表盘/自检/备份/导出/工作区"
            # === 超级版新增: pending 待确认队列 ===
            elif a == "pending":
                items = self.s.pending_list()
                if not items:
                    return "[Z] 没有待确认知识"
                return "\n".join([f"[Z] 待确认 ({len(items)}条):"] + [f"  - {p['id']} {p['content'][:60]}" for p in items])
            elif a == "confirm":
                pid = re.sub(r"确认|confirm", "", args).strip()
                nid = self.s.pending_confirm(pid)
                if nid:
                    return f"[Z] 已确认 {pid} → #{nid}"
                return f"[Z] 未找到 {pid}"
            # === 超级版新增: entangle 纠缠场 ===
            elif a == "entangle":
                words = re.sub(r"纠缠|关联场|entangle", "", args).strip().split()
                if len(words) < 2:
                    return "[Z] 用法: entangle 词1 词2 [词3...]"
                r = entangle(self.s, words, self.ws)
                lines = [f"[Z] 纠缠场分析:"]
                for p in r["entanglements"]:
                    lines.append(f"  {p['word_a']} ↔ {p['word_b']}: 共同节点{p['common_nodes']}个, 间接关联{len(p['indirect_links'])}条")
                return "\n".join(lines)
            # === 超级版新增: auto_learn 自动学习 ===
            elif a == "auto_learn":
                p = re.sub(r"^自动学习[：:]?|^自动提取[：:]?|^从这段话[：:]?|^读一下这段[：:]?", "", args).strip()
                if not p:
                    return "[Z] 请提供要学习的内容"
                ids = auto_learn(self.s, p, self.ws)
                if not ids:
                    return "[Z] 未提取到知识点"
                return f"[Z] 已提取 {len(ids)} 条知识点到待确认队列，用 pending 查看"
            return f"[Z] 未识别: {a}"
        except Exception as e:
            return f"[Z] 出错: {e}"

    def _export(self, fmt):
        nodes = self.s.valid()
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        p = DATA_DIR / f"export_{ts}.{fmt}"
        if fmt == "json":
            with open(p, "w", encoding="utf-8") as _f:
                json.dump({"nodes": nodes}, _f, ensure_ascii=False, indent=2)
        elif fmt == "csv":
            import csv
            with open(p, "w", encoding="utf-8-sig", newline="") as _f:
                w = csv.writer(_f)
            w.writerow(["id", "workspace", "category", "text", "tags", "confidence", "learned_at"])
            for n in nodes:
                w.writerow([n["id"], n["workspace"], n["category"], n["text"], n.get("tags", ""), n["confidence"], n["learned_at"]])
        elif fmt == "graphml":
            with open(p, "w", encoding="utf-8") as f:
                f.write('<?xml version="1.0" encoding="UTF-8"?>\n<graphml xmlns="http://graphml.graphdrawing.org/xmlns">\n<graph id="G" edgedefault="undirected">\n')
                for n in nodes:
                    txt = n.get("text", "")[:50].replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")
                    f.write(f'  <node id="n{n["id"]}"><data key="label">{txt}</data></node>\n')
                seen = set()
                for n in nodes:
                    for e in n.get("edges", []):
                        sid, tid = n["id"], e["target"]
                        if (sid, tid) not in seen and (tid, sid) not in seen:
                            seen.add((sid, tid))
                            f.write(f'  <edge source="n{sid}" target="n{tid}"/>\n')
                f.write('</graph>\n</graphml>\n')
        elif fmt == "markdown" or fmt == "md":
            # === 超级版新增: Markdown 导出 (来自左脑v3.0) ===
            with open(p, "w", encoding="utf-8") as f:
                f.write(f"# 知识库导出\n\n导出时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
                f.write(f"总计: {len(nodes)} 条知识\n\n---\n\n")
                cat_groups = defaultdict(list)
                for n in nodes:
                    cat_groups[n.get("category", "未分类")].append(n)
                for cat, items in sorted(cat_groups.items()):
                    f.write(f"\n## {cat} ({len(items)}条)\n\n")
                    for n in items:
                        tags = ", ".join(n.get("tags", [])) if n.get("tags") else ""
                        tag_str = f" `#{tags}`" if tags else ""
                        conf = f" (置信度:{n.get('confidence', 1.0):.2f})" if n.get("confidence", 1.0) < 1.0 else ""
                        f.write(f"- **#{n['id']}** {n['text']}{tag_str}{conf}\n")
                        # 边关系
                        for e in n.get("edges", []):
                            etype = e.get("type", "related")
                            f.write(f"  - → #{e['target']} ({etype})\n")
        return f"[Z] 导出 {len(nodes)}条到 {p}"

    def workbuddy_main(self, action, args=""):
        return self._exec(action, args)

    def sync_inject(self):
        nodes = self.s.valid(self.ws)
        lines = [f"你在「{self.ws}」积累了 {len(nodes)} 条知识。"]
        cat = Counter(n["category"] for n in nodes).most_common(3)
        if cat:
            lines.append("类别: " + ", ".join(f"{c}({n})" for c, n in cat))
        (DATA_DIR.parent / "_inject.md").write_text("\n".join(lines), encoding="utf-8")
        return "\n".join(lines)


def main():
    if len(sys.argv) > 1:
        lb = ZhiLuo()
        r = lb.run(" ".join(sys.argv[1:]))
        try:
            print(r)
        except UnicodeEncodeError:
            sys.stdout.buffer.write((str(r) + "\n").encode("utf-8", errors="strict"))
    else:
        if os.environ.get("ZHILUO_ALLOW_REPL", "0") != "1":
            print("[Z] REPL 默认关闭，已跳过。设置 ZHILUO_ALLOW_REPL=1 才进入交互模式。", file=sys.stderr)
            return
        lb = ZhiLuo()
        print("[Z] 知络 v8.9.1 统一版")
        while True:
            try:
                l = input(">>> ").strip()
                if l in ("quit", "exit"):
                    break
                if l:
                    r = lb.run(l)
                    try:
                        print(r)
                    except UnicodeEncodeError:
                        sys.stdout.buffer.write((str(r) + "\n").encode("utf-8", errors="strict"))
            except (EOFError, KeyboardInterrupt):
                break

if __name__ == "__main__":
    main()

# === @deprecated v8.9.7 — 中期预留接口，未实现，保留供参考 ===
class VectorBackend:
    def search(self, text, top_k=20):
        raise NotImplementedError
    def add(self, nid, text):
        raise NotImplementedError
    def delete(self, nid):
        raise NotImplementedError


# v8.9.7: 语义化别名 — 解决同名函数在多个模块中签名冲突的问题。
# 旧名保留向后兼容，新代码推荐使用语义化别名。
search_knowledge = find           # 关键词检索（原名 find，与 MinHashLSH.find 区分）
search_graph = diffuse            # 图谱扩散搜索（原名 diffuse）
detect_conflicts = conflicts      # 三重冲突检测（原名 conflicts）
rank_importance = pagerank        # PageRank 枢纽排名（原名 pagerank）
visualize_graph = mermaid_graph   # Mermaid 图谱可视化（原名 mermaid_graph）
merge_memories = consolidate      # LLM 记忆合并（原名 consolidate）


