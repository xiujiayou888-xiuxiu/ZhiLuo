# -*- coding: utf-8 -*-
"""
知络基础版 — 知识库配置模块 kb_config.py

单Agent版本：去掉多Agent共享配置、异步队列、争议上限等复杂功能。
保留核心配置：KB路径、扫描时间、待审核TTL、SimHash阈值、LLM配置。

路径发现优先级：
  1. 环境变量 ZHILUO_KB_PATH
  2. 环境变量 ZHILUO_DATA_DIR（自动追加 /workspaces）
  3. 中央配置 %USERPROFILE%/.zhiluo/config.json 中的 kb_path
  4. 本地兜底 ~/zhiluo/data/workspaces
"""

import os
import json
import sqlite3
import re
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parent


def _default_kb_path():
    data = os.environ.get("ZHILUO_DATA_DIR", "")
    if data:
        return Path(data) / "workspaces"
    return Path.home() / "zhiluo" / "data" / "workspaces"


DEFAULT_KB_PATH = _default_kb_path()


# ---- 自动加载 .env 文件 ----
def _load_dotenv():
    env_file = ROOT / ".env"
    if not env_file.exists():
        return
    try:
        with open(env_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#') or '=' not in line:
                    continue
                key, _, val = line.partition('=')
                key = key.strip()
                val = val.strip().strip('"').strip("'")
                if key and val and key not in os.environ:
                    os.environ[key] = val
    except Exception:
        pass


_load_dotenv()

# 中央配置路径
CENTRAL_CONFIG_DIR = Path(os.environ.get(
    "ZHILUO_CONFIG_DIR",
    Path.home() / ".zhiluo"
))
CENTRAL_CONFIG_FILE = CENTRAL_CONFIG_DIR / "config.json"


def _now():
    return datetime.now().isoformat()


def _load_central_config():
    if CENTRAL_CONFIG_FILE.exists():
        try:
            with open(CENTRAL_CONFIG_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    return {}


def _save_central_config(config):
    CENTRAL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    config["updated_at"] = _now()
    tmp = CENTRAL_CONFIG_FILE.with_suffix(CENTRAL_CONFIG_FILE.suffix + ".tmp")
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    tmp.replace(CENTRAL_CONFIG_FILE)


def get_config_value(key, default=None):
    return _load_central_config().get(key, default)


def set_config_value(key, value):
    config = _load_central_config()
    config[key] = value
    if "created_at" not in config:
        config["created_at"] = _now()
    _save_central_config(config)
    return CENTRAL_CONFIG_FILE


def normalize_scan_time(value):
    raw = str(value or "").strip()
    m = re.fullmatch(r"([01]?\d|2[0-3]):([0-5]\d)", raw)
    if not m:
        raise ValueError("scan_time must be HH:MM, e.g. 09:00")
    return "%02d:%02d" % (int(m.group(1)), int(m.group(2)))


def get_scan_time(default="09:00"):
    raw = os.environ.get("ZHILUO_SCAN_TIME") or get_config_value("scan_time", default)
    try:
        return normalize_scan_time(raw)
    except ValueError:
        return normalize_scan_time(default)


def set_scan_time(value):
    value = normalize_scan_time(value)
    set_config_value("scan_time", value)
    return value


def get_pending_ttl_days(default=3):
    raw = os.environ.get("ZHILUO_PENDING_TTL_DAYS") or get_config_value("pending_ttl_days", default)
    try:
        ttl = int(str(raw).strip())
    except (TypeError, ValueError):
        ttl = int(default)
    return max(0, ttl)


def set_pending_ttl_days(value):
    try:
        ttl = int(str(value).strip())
    except (TypeError, ValueError):
        raise ValueError("pending_ttl_days must be an integer")
    if ttl < 0:
        raise ValueError("pending_ttl_days must be >= 0")
    set_config_value("pending_ttl_days", ttl)
    return ttl


def get_simhash_threshold(default=0.92):
    raw = os.environ.get("ZHILUO_SIMHASH_THRESHOLD") or get_config_value("simhash_threshold", default)
    try:
        value = float(str(raw).strip())
    except (TypeError, ValueError):
        value = float(default)
    return max(0.50, min(value, 0.99))


def set_simhash_threshold(value):
    try:
        threshold = float(str(value).strip())
    except (TypeError, ValueError):
        raise ValueError("simhash_threshold must be a number")
    if threshold < 0.50 or threshold > 0.99:
        raise ValueError("simhash_threshold must be between 0.50 and 0.99")
    set_config_value("simhash_threshold", threshold)
    return threshold


def _normalize_kb_path(raw_path):
    if not raw_path:
        return None
    p = Path(raw_path).expanduser()
    if p.suffix.lower() in (".db", ".sqlite", ".sqlite3"):
        p = p.parent
    try:
        resolved = p.resolve()
    except Exception:
        return None
    if resolved.exists() and not resolved.is_dir():
        return None
    return resolved


def get_kb_path():
    """获取知识库路径。优先级：环境变量 > 中央配置 > 默认路径"""
    # 1. 环境变量 ZHILUO_KB_PATH
    env_path = os.environ.get("ZHILUO_KB_PATH")
    env_kb = _normalize_kb_path(env_path)
    if env_kb:
        return env_kb

    # 2. 环境变量 ZHILUO_DATA_DIR
    data_dir = os.environ.get("ZHILUO_DATA_DIR")
    if data_dir:
        data_kb = _normalize_kb_path(str(Path(data_dir) / "workspaces"))
        if data_kb:
            return data_kb

    # 3. 中央配置
    config = _load_central_config()
    kb_path = config.get("kb_path")
    cfg_kb = _normalize_kb_path(kb_path)
    if cfg_kb:
        return cfg_kb

    # 4. 本地兜底
    return DEFAULT_KB_PATH


def get_global_db():
    db = get_kb_path() / "global.db"
    _ensure_wal(db)
    return db


def _ensure_wal(db_path):
    db = Path(db_path)
    db.parent.mkdir(parents=True, exist_ok=True)
    cache = getattr(_ensure_wal, "_cache", set())
    key = str(db.resolve())
    if key in cache:
        return
    try:
        con = sqlite3.connect(str(db), timeout=30)
        con.execute("PRAGMA busy_timeout=30000")
        con.execute("PRAGMA journal_mode=WAL")
        con.execute("PRAGMA synchronous=NORMAL")
        con.close()
        cache.add(key)
        _ensure_wal._cache = cache
    except Exception:
        pass


def get_data_dir():
    kb = get_kb_path()
    if kb.name == "workspaces":
        return kb.parent
    return kb


def set_kb_path(kb_path):
    """写入中央配置的 KB 路径"""
    CENTRAL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    safe_path = _normalize_kb_path(kb_path)
    if not safe_path:
        raise ValueError("invalid knowledge-base path: %s" % kb_path)
    safe_path.mkdir(parents=True, exist_ok=True)
    config = _load_central_config()
    config["kb_path"] = str(safe_path)
    config["version"] = "basic-1.0"
    if "created_at" not in config:
        config["created_at"] = _now()
    with open(CENTRAL_CONFIG_FILE, 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    return CENTRAL_CONFIG_FILE


def init_kb(kb_path):
    """初始化 KB 目录（创建目录结构 + 数据库）"""
    kb_path = Path(kb_path)
    kb_path.mkdir(parents=True, exist_ok=True)

    db_path = kb_path / "global.db"
    if not db_path.exists():
        con = sqlite3.connect(str(db_path), timeout=30)
        con.execute("PRAGMA busy_timeout=30000")
        con.executescript("""
            CREATE TABLE IF NOT EXISTS nodes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text TEXT NOT NULL,
                workspace TEXT NOT NULL DEFAULT 'global',
                category TEXT DEFAULT '未分类',
                simhash INTEGER DEFAULT 0,
                minhash_sig TEXT DEFAULT '[]',
                access_count INTEGER DEFAULT 0, tags TEXT,
                confidence REAL DEFAULT 1.0, created_at TEXT,
                learned_at TEXT, updated_at TEXT, last_accessed_at TEXT,
                session_id TEXT, source TEXT DEFAULT 'user', edges_json TEXT DEFAULT '[]',
                embedding TEXT, para TEXT, created_by TEXT DEFAULT 'default',
                visibility TEXT DEFAULT 'team', trust_score REAL DEFAULT 1.0,
                source_url TEXT, content_hash TEXT, fetched_at TEXT,
                status TEXT DEFAULT 'active', confirmed_by TEXT DEFAULT '',
                text_jieba TEXT DEFAULT ''
            );
            CREATE TABLE IF NOT EXISTS edges (
                id TEXT PRIMARY KEY, source_id TEXT, target_id TEXT,
                relation TEXT, weight REAL, created_at TEXT
            );
            CREATE TABLE IF NOT EXISTS obsidian_export_state (
                id TEXT PRIMARY KEY, vault_path TEXT,
                last_export_at TEXT, exported_count INTEGER,
                created_at TEXT, updated_at TEXT
            );
            CREATE VIRTUAL TABLE IF NOT EXISTS nodes_fts USING fts5(
                text, category, tags, content='nodes', content_rowid='id'
            );
        """)
        con.commit()
        con.close()

    # 写入中央配置
    set_kb_path(kb_path)
    return kb_path


def get_central_config():
    return _load_central_config()


def get_llm_config() -> dict:
    return {
        "allow": os.environ.get("ZHILUO_LLM_ALLOW", "0"),
        "api_key": os.environ.get("ZHILUO_LLM_API_KEY", ""),
        "api_url": os.environ.get("ZHILUO_LLM_API_URL", ""),
        "model": os.environ.get("ZHILUO_LLM_MODEL", "deepseek-chat"),
        "allowed_hosts": os.environ.get("ZHILUO_LLM_ALLOWED_HOSTS", ""),
    }
