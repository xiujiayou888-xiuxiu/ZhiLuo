# -*- coding: utf-8 -*-
"""
知络基础版 — Obsidian 同步模块

精简自原版 obsidian_export.py + bridge.py。
功能：导出知识库到 Obsidian vault（Markdown + YAML frontmatter），
      从 vault 导入新 Markdown 文件到知识库。
不修改数据库结构。
"""

import json
import os
import re
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parent
try:
    from kb_config import get_global_db
    GLOBAL_DB = get_global_db()
except ImportError:
    GLOBAL_DB = ROOT / "data" / "workspaces" / "global.db"

DEFAULT_VAULT = str(Path.home() / "zhiluo-vault")

# 目录 → 分类映射
DIR_TO_CATEGORY = {
    "00-收件箱": "未分类",
    "01-项目": "项目",
    "02-知识库": "通用",
    "03-输出": "输出",
    "04-踩坑库": "踩坑",
    "05-Skills": "方法论",
}

# 保护目录（不同步）
PROTECTED_DIRS = {
    ".obsidian", ".trash", ".git", "__pycache__",
    "_attachments", "_assets", "templates", "模板",
    "仪表盘", "README",
}


def _now():
    return datetime.now().isoformat()


def _get_conn():
    con = sqlite3.connect(str(GLOBAL_DB), timeout=30)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA busy_timeout=30000")
    # 确保 obsidian_export_state 表存在（不改结构，首次自动建）
    con.execute("""
        CREATE TABLE IF NOT EXISTS obsidian_export_state (
            id TEXT PRIMARY KEY, vault_path TEXT,
            last_export_at TEXT, exported_count INTEGER,
            created_at TEXT, updated_at TEXT
        )
    """)
    con.commit()
    return con


# ═══════════════════════════════════════════════════════════════
#  Vault 路径管理
# ═══════════════════════════════════════════════════════════════

def get_vault_path():
    """获取已配置的 vault 路径。"""
    con = _get_conn()
    try:
        cur = con.execute(
            "SELECT vault_path FROM obsidian_export_state ORDER BY updated_at DESC LIMIT 1"
        )
        r = cur.fetchone()
        return r[0] if r else DEFAULT_VAULT
    finally:
        con.close()


def set_vault_path(path):
    """设置 vault 路径。"""
    con = _get_conn()
    try:
        did = uuid.uuid4().hex[:12]
        con.execute(
            "INSERT INTO obsidian_export_state(id, vault_path, created_at, updated_at) "
            "VALUES(?,?,?,?)",
            (did, str(path), _now(), _now()),
        )
        con.commit()
    finally:
        con.close()


def auto_detect_vault():
    """自动检测 Obsidian vault。"""
    # 已配置的直接返回
    existing = get_vault_path()
    if existing != DEFAULT_VAULT and Path(existing).exists():
        return existing

    # 尝试读取 Obsidian 配置
    import platform
    system = platform.system()
    obsidian_config = None
    if system == "Windows":
        obsidian_config = Path(os.environ.get("APPDATA", "")) / "obsidian" / "obsidian.json"
    elif system == "Darwin":
        obsidian_config = Path.home() / "Library" / "Application Support" / "obsidian" / "obsidian.json"
    elif system == "Linux":
        obsidian_config = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "obsidian" / "obsidian.json"

    if obsidian_config and obsidian_config.exists():
        try:
            with open(obsidian_config, 'r', encoding='utf-8') as f:
                config = json.load(f)
            vaults = config.get("vaults", {})
            for vault_info in vaults.values():
                vpath = vault_info.get("path", "")
                if vpath and Path(vpath).exists():
                    # 优先选含"zhiluo"的
                    if "zhiluo" in vpath.lower():
                        set_vault_path(vpath)
                        return vpath
            # 取第一个
            for vault_info in vaults.values():
                vpath = vault_info.get("path", "")
                if vpath and Path(vpath).exists():
                    set_vault_path(vpath)
                    return vpath
        except Exception:
            pass

    # 兜底：创建默认 vault
    Path(DEFAULT_VAULT).mkdir(parents=True, exist_ok=True)
    set_vault_path(DEFAULT_VAULT)
    return DEFAULT_VAULT


# ═══════════════════════════════════════════════════════════════
#  导出：DB → Obsidian Markdown
# ═══════════════════════════════════════════════════════════════

def _safe_filename(s, max_len=40):
    """安全文件名：过滤非法字符。"""
    bad = '/\\:*?"<>|\n\r\t'
    out = "".join(c if c not in bad else "_" for c in s)
    out = re.sub(r"_+", "_", out)
    return (out.strip("_ ") or "untitled")[:max_len]


def _extract_title(text):
    """从文本中提取中文标题。"""
    if not text:
        return None
    # 取第一句
    first = re.split(r'[。！？；\n]', text)[0].strip()
    # 去掉括号内容
    cleaned = re.sub(r'[（(][^）)]*[）)]', '', first)
    cleaned = cleaned.split('——')[0].strip()
    has_cn = any('\u4e00' <= c <= '\u9fff' for c in cleaned[:10])
    if has_cn and len(cleaned) >= 2:
        return cleaned[:35].rstrip('，, ')
    # 回退：取纯中文最长段
    spans = re.findall(r'[\u4e00-\u9fff]{2,}', text)
    if spans:
        return max(spans, key=len)[:35]
    return None


def _node_filename(node):
    """生成 Markdown 文件名。"""
    text = node.get("text", "")
    nid = node.get("id", 0)
    title = _extract_title(text)
    if title:
        return "%s-%06d.md" % (_safe_filename(title, max_len=35), nid)
    cat = node.get("category", "未分类")
    return "%s-%06d.md" % (_safe_filename(cat, max_len=20), nid)


def _yaml_frontmatter(node):
    """生成 YAML frontmatter。"""
    tags = node.get("tags", "")
    tag_list = []
    if isinstance(tags, str) and tags.strip() and tags.strip() not in ("[]", ""):
        # 可能是逗号分隔的字符串或 JSON 数组
        t = tags.strip()
        if t.startswith("["):
            try:
                tag_list = [x.strip().strip('"').strip("'") for x in json.loads(t) if x.strip()]
            except Exception:
                tag_list = [x.strip() for x in t.strip("[]").split(",") if x.strip()]
        else:
            tag_list = [x.strip() for x in t.split(",") if x.strip()]
    elif isinstance(tags, list):
        tag_list = [str(x).strip() for x in tags if str(x).strip()]

    lines = [
        "---",
        "zhiluo_id: %s" % node.get("id", 0),
        "category: %s" % (node.get("category", "未分类")),
        "source: %s" % (node.get("source", "")),
        "created: %s" % (str(node.get("created_at", ""))[:10]),
        "confidence: %.2f" % float(node.get("confidence") or node.get("trust_score") or 1.0),
    ]
    if tag_list:
        lines.append("tags: [%s]" % ", ".join(tag_list))
    lines.append("---")
    return "\n".join(lines)


def _category_dir(category, vault_path):
    """分类 → Obsidian 目录路径。"""
    cat = category or "未分类"
    # 特殊目录映射
    if cat in ("踩坑", "踩坑库"):
        return Path(vault_path) / "04-踩坑库"
    if cat in ("未分类", "_待分类_", ""):
        return Path(vault_path) / "00-收件箱"
    if cat in ("项目管理", "项目"):
        return Path(vault_path) / "01-项目"
    if cat in ("方法论", "Skills", "SOP"):
        return Path(vault_path) / "05-Skills"
    if cat in ("输出", "报告"):
        return Path(vault_path) / "03-输出"
    # 默认放知识库
    return Path(vault_path) / "02-知识库" / _safe_filename(cat, max_len=20)


def export_to_vault(vault_path=None, keyword=""):
    """导出知识库到 Obsidian vault。

    参数:
        vault_path: vault 路径（默认自动检测）
        keyword: 过滤关键词（可选）

    返回: {"exported": N, "vault_path": str, "skipped": N}
    """
    if not vault_path:
        vault_path = get_vault_path()
    vault = Path(vault_path)
    vault.mkdir(parents=True, exist_ok=True)

    # 创建 .obsidian 基础配置（如果不存在）
    obsidian_dir = vault / ".obsidian"
    if not obsidian_dir.exists():
        obsidian_dir.mkdir(parents=True, exist_ok=True)

    # 读取所有节点
    con = _get_conn()
    try:
        if keyword:
            rows = con.execute(
                "SELECT * FROM nodes WHERE text LIKE ? AND (status IS NULL OR status!='archived') ORDER BY id",
                ("%" + keyword + "%",),
            ).fetchall()
        else:
            rows = con.execute(
                "SELECT * FROM nodes WHERE status IS NULL OR status!='archived' ORDER BY id"
            ).fetchall()

        exported = 0
        skipped = 0
        exported_ids = set()

        for row in rows:
            node = dict(row)
            nid = node.get("id")
            text = node.get("text", "").strip()
            if not text or len(text) < 10:
                skipped += 1
                continue

            # 确定目录
            cat_dir = _category_dir(node.get("category", ""), vault)
            cat_dir.mkdir(parents=True, exist_ok=True)

            # 生成文件名
            filename = _node_filename(node)
            filepath = cat_dir / filename

            # 生成内容：YAML frontmatter + body
            fm = _yaml_frontmatter(node)
            content = "%s\n\n%s\n" % (fm, text)

            # 写入
            filepath.write_text(content, encoding="utf-8")
            exported += 1
            exported_ids.add(nid)

        # 更新导出状态
        con.execute(
            "INSERT INTO obsidian_export_state(id, vault_path, last_export_at, exported_count, created_at, updated_at) "
            "VALUES(?,?,?,?,?,?)",
            (uuid.uuid4().hex[:12], str(vault), _now(), exported, _now(), _now()),
        )
        con.commit()

        return {
            "exported": exported,
            "skipped": skipped,
            "vault_path": str(vault),
            "total": len(rows),
        }
    finally:
        con.close()


# ═══════════════════════════════════════════════════════════════
#  导入：Obsidian Markdown → DB
# ═══════════════════════════════════════════════════════════════

def _parse_frontmatter(content):
    """解析 YAML frontmatter。"""
    fm = {}
    body = content
    m = re.match(r"^---\s*\n(.*?)\n---\s*\n?", content, re.DOTALL)
    if m:
        for line in m.group(1).split("\n"):
            line = line.strip()
            if ":" in line:
                key, _, val = line.partition(":")
                fm[key.strip()] = val.strip().strip('"').strip("'")
        body = content[m.end():].strip()
    return fm, body


def import_from_vault(vault_path=None, import_new=True):
    """从 Obsidian vault 导入 Markdown 到知识库。

    参数:
        vault_path: vault 路径（默认自动检测）
        import_new: 是否导入新文件（无 zhiluo_id 的）

    返回: {"imported": N, "updated": N, "skipped": N, "failed": N}
    """
    if not vault_path:
        vault_path = get_vault_path()
    vault = Path(vault_path)
    if not vault.exists():
        return {"imported": 0, "updated": 0, "skipped": 0, "failed": 0,
                "error": "vault 路径不存在: %s" % vault_path}

    imported = 0
    updated = 0
    skipped = 0
    failed = 0

    con = _get_conn()
    try:
        # 获取已有节点ID列表
        existing_ids = {
            int(r[0]) for r in con.execute("SELECT id FROM nodes").fetchall()
        }

        for md_file in vault.rglob("*.md"):
            # 跳过保护目录
            parts = set(md_file.parts)
            if parts & PROTECTED_DIRS:
                continue
            # 跳过 .obsidian 目录
            if ".obsidian" in str(md_file):
                continue

            try:
                content = md_file.read_text(encoding="utf-8")
            except Exception:
                skipped += 1
                continue

            fm, body = _parse_frontmatter(content)
            if not body or len(body.strip()) < 10:
                skipped += 1
                continue

            zhiluo_id = fm.get("zhiluo_id")
            if zhiluo_id:
                try:
                    nid = int(zhiluo_id)
                    if nid in existing_ids:
                        # 更新已有节点
                        con.execute(
                            "UPDATE nodes SET text=?, updated_at=? WHERE id=?",
                            (body.strip(), _now(), nid),
                        )
                        updated += 1
                        continue
                except (ValueError, TypeError):
                    pass

            # 新文件导入
            if import_new:
                try:
                    # 从目录推断分类
                    category = "未分类"
                    for part in md_file.parent.parts:
                        for dir_key, cat_val in DIR_TO_CATEGORY.items():
                            if dir_key in part:
                                category = cat_val
                                break

                    # 计算 content_hash 去重
                    import hashlib
                    content_hash = hashlib.sha1(body.encode("utf-8")).hexdigest()
                    existing = con.execute(
                        "SELECT id FROM nodes WHERE content_hash=? LIMIT 1",
                        (content_hash,),
                    ).fetchone()
                    if existing:
                        skipped += 1
                        continue

                    con.execute(
                        "INSERT INTO nodes(text, workspace, category, source, "
                        "confidence, trust_score, created_at, content_hash, status) "
                        "VALUES(?,?,?,?,?,?,?,?,?)",
                        (body.strip(), "global", category, "obsidian_import",
                         0.8, 0.8, _now(), content_hash, "active"),
                    )
                    imported += 1
                except Exception:
                    failed += 1

        con.commit()
    finally:
        con.close()

    return {
        "imported": imported,
        "updated": updated,
        "skipped": skipped,
        "failed": failed,
        "vault_path": str(vault),
    }


# ═══════════════════════════════════════════════════════════════
#  同步统计
# ═══════════════════════════════════════════════════════════════

def sync_status():
    """查看同步状态。"""
    vault_path = get_vault_path()
    con = _get_conn()
    try:
        total = con.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]
        state = con.execute(
            "SELECT * FROM obsidian_export_state ORDER BY updated_at DESC LIMIT 1"
        ).fetchone()
        last_export = dict(state) if state else None
    finally:
        con.close()

    vault_exists = Path(vault_path).exists()
    vault_files = 0
    if vault_exists:
        vault_files = sum(1 for _ in Path(vault_path).rglob("*.md")
                          if ".obsidian" not in str(_))

    return {
        "vault_path": vault_path,
        "vault_exists": vault_exists,
        "vault_md_files": vault_files,
        "db_total_nodes": total,
        "last_export": last_export,
    }
