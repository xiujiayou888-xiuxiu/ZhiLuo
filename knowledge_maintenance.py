# -*- coding: utf-8 -*-
"""知络基础版 — 知识维护模块

单Agent精简版：保留分类、补边、边修复、统计，去掉兜底队列/自动学习/日报合成/去重合并。
"""

import hashlib
import json
import re
import shutil
import sqlite3
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

try:
    import jieba.analyse
    _HAS_JIEBA = True
except Exception:
    _HAS_JIEBA = False

try:
    from kb_config import get_global_db
except Exception:
    ROOT = Path(__file__).resolve().parent

    def get_global_db():
        return ROOT / "data" / "workspaces" / "global.db"


WEAK_CATEGORIES = {"", "未分类", "_待分类_", "rss_feed", None}

CATEGORY_RULES = [
    ("测试数据", ["测试节点", "测试数据", "测试知识", "压力测试", "para测试", "demo seed", "load_demo", "演示数据"]),
    ("知络", ["知络", "zhiluo", "外脑", "记忆", "知识库", "pre_answer_context", "learn()", "query()", "selfcheck"]),
    ("Agent协作", ["agent", "智能体", "codex", "workbuddy", "mcp", "多agent", "多 agent", "共享知识库", "自动接入"]),
    ("Obsidian", ["obsidian", "vault", "双链", "frontmatter", "markdown", "导出", "首页", "图谱视图"]),
    ("安全", ["安全", "漏洞", "注入", "权限", "白名单", "密钥", "cookie", "authorization", "路径穿越", "脱敏"]),
    ("自动化", ["自动", "扫描", "每日扫描", "抓取", "rss", "定时", "监控", "导出", "同步", "批量"]),
    ("AI变现", ["ai变现", "变现", "商业化", "赚钱", "付费", "订单", "营收", "收入", "客户", "交付", "商业"]),
    ("AI创业", ["创业", "创投", "融资", "投资", "初创", "上市", "ceo", "公司", "风向标", "浪潮"]),
    ("AI项目", ["producthunt", "github", "trending", "开源", "项目", "工具", "应用", "repo", "hn", "hacker news"]),
    ("AI模型", ["ai", "gpt", "claude", "gemini", "openai", "anthropic", "大模型", "llm", "moe", "transformer", "transformers", "prompt", "rag", "微调", "token", "英伟达", "谷歌", "meta"]),
    ("机器人", ["机器人", "具身", "智能车", "自动驾驶", "工厂", "产线", "物理ai", "精细操作"]),
    ("电商", ["电商", "tiktok", "costco", "山姆", "商品", "品牌", "跨境", "货架", "会员店", "消费"]),
    ("餐饮", ["牛肉", "猪肉", "火锅", "川菜", "菜品", "毛利", "前厅", "供应链", "采购", "底料", "翻台率", "麻辣"]),
    ("内容创作", ["抖音", "视频", "脚本", "选题", "流量", "爆款", "内容", "拍摄", "发布", "复盘"]),
    ("项目管理", ["计划", "升级", "需求", "版本", "sop", "蓝图", "里程碑", "安装", "修复", "优化"]),
    ("技术", ["python", "代码", "api", "接口", "数据库", "sqlite", "bug", "框架", "前端", "后端", "服务器"]),
    ("产品", ["产品", "功能", "体验", "用户", "需求", "设计", "原型", "工具"]),
    ("运营", ["运营", "转化", "留存", "用户画像", "看板", "指标"]),
    ("营销", ["营销", "广告", "投放", "获客", "私域", "社群", "销售", "销量", "销售额", "促销"]),
    ("财务", ["财务", "预算", "成本", "利润", "收入", "支出", "报价", "发票", "元", "万"]),
    ("会议", ["会议", "纪要", "讨论", "决议", "参会", "议程"]),
    ("学习", ["学习", "教程", "课程", "笔记", "知识点", "总结", "文档", "方法"]),
    ("生活", ["生活", "健身", "做饭", "菜谱", "运动", "饮食", "休息", "旅行"]),
]

PARA_BY_CATEGORY = {
    "测试数据": "X",
    "项目管理": "P",
    "知络": "A",
    "Agent协作": "A",
    "Obsidian": "A",
    "安全": "A",
    "自动化": "A",
    "AI变现": "A",
    "AI与变现": "A",
    "AI模型": "R",
    "AI项目": "R",
    "AI创业": "R",
    "机器人": "R",
    "电商": "R",
    "餐饮": "A",
    "内容创作": "A",
    "技术": "R",
    "产品": "A",
    "运营": "A",
    "营销": "A",
    "财务": "A",
    "会议": "P",
    "学习": "R",
    "生活": "A",
    "通用": "R",
}

CATEGORY_MERGE = {
    "AI模型": "AI与变现",
    "AI项目": "AI与变现",
    "AI创业": "AI与变现",
    "AI变现": "AI与变现",
    "Agent协作": "知络",
    "Obsidian": "知络",
    "自动化": "知络",
    "产品": "项目与技术",
    "技术": "项目与技术",
    "项目管理": "项目与技术",
    "电商": "商业运营",
    "营销": "商业运营",
    "运营": "商业运营",
    "内容创作": "商业运营",
    "财务": "商业运营",
}


def _connect(db_path=None):
    con = sqlite3.connect(str(db_path or get_global_db()), timeout=30)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA busy_timeout=30000")
    return con


import threading as _threading
_rules_cache = {}           # dict[db_path] → [(category, [keywords]), ...]
_rules_lock = _threading.Lock()


def _ensure_category_rules_table(con):
    con.execute("""
        CREATE TABLE IF NOT EXISTS category_rules (
            id INTEGER PRIMARY KEY,
            category TEXT NOT NULL,
            keyword TEXT NOT NULL,
            source TEXT DEFAULT 'seed',
            created_at TEXT DEFAULT (datetime('now','localtime')),
            UNIQUE(category, keyword)
        )
    """)
    cnt = con.execute("SELECT COUNT(*) FROM category_rules").fetchone()[0]
    if cnt == 0:
        for cat, kws in CATEGORY_RULES:
            for kw in kws:
                con.execute(
                    "INSERT OR IGNORE INTO category_rules(category, keyword, source) VALUES(?,?,?)",
                    (cat, kw, "seed")
                )
        con.commit()


def _load_rules(db_path=None):
    global _rules_cache
    _db = str(db_path or get_global_db())
    with _rules_lock:
        if _db in _rules_cache:
            return _rules_cache[_db]
        con = _connect(db_path)
        try:
            _ensure_category_rules_table(con)
            rows = con.execute("SELECT category, keyword FROM category_rules ORDER BY category").fetchall()
        finally:
            con.close()
        rules = {}
        for r in rows:
            rules.setdefault(r["category"], []).append(r["keyword"])
        _rules_cache[_db] = list(rules.items())
        return _rules_cache[_db]


def add_category_rule(category, keywords, source="learned", db_path=None):
    global _rules_cache
    if isinstance(keywords, str):
        keywords = [k.strip() for k in keywords.split(",") if k.strip()]
    con = _connect(db_path)
    try:
        _ensure_category_rules_table(con)
        for kw in keywords:
            con.execute(
                "INSERT OR IGNORE INTO category_rules(category, keyword, source) VALUES(?,?,?)",
                (category, kw, source)
            )
        con.commit()
    finally:
        con.close()
    _db = str(db_path or get_global_db())
    _rules_cache.pop(_db, None)
    return True


def backup_db(db_path=None):
    db = Path(db_path or get_global_db())
    backup_dir = db.parent / "backups"
    backup_dir.mkdir(parents=True, exist_ok=True)
    out = backup_dir / ("global_maintenance_%s.db" % datetime.now().strftime("%Y%m%d_%H%M%S"))
    if db.exists():
        shutil.copy2(str(db), str(out))
    return str(out)


_ALLOWED_TABLES = {"nodes", "edges", "pending", "scan_jobs", "process_log", "disputes"}


def _columns(con, table="nodes"):
    if table not in _ALLOWED_TABLES:
        raise ValueError("unsupported table: %s" % table)
    return {r[1] for r in con.execute("PRAGMA table_info(" + table + ")").fetchall()}


def classify_text(text, source="", tags="", db_path=None):
    """分类文本。关键词打分 + CATEGORY_MERGE 合并。"""
    hay = " ".join([text or "", tags or ""]).lower()
    scores = []
    for category, keywords in _load_rules(db_path):
        score = 0
        for kw in keywords:
            k = kw.lower()
            if k and k in hay:
                score += 2 if len(k) >= 3 else 1
        scores.append((category, score))
    scores.sort(key=lambda x: -x[1])
    if scores and scores[0][1] > 0:
        return scores[0][0], scores[0][1]
    if source == "rss":
        return "实时热点", 1
    if "http" in hay or "www." in hay:
        return "资源", 1
    return "_待分类_", 0


def normalize_category(cat):
    if not cat:
        return "通用"
    return CATEGORY_MERGE.get(cat, cat)


def _json_edges(raw):
    try:
        edges = json.loads(raw or "[]")
        return edges if isinstance(edges, list) else []
    except Exception:
        return []


def _dump_edges(edges):
    clean = []
    seen = set()
    for e in edges:
        if not isinstance(e, dict):
            continue
        try:
            target = int(e.get("target"))
        except Exception:
            continue
        etype = str(e.get("type") or "related")
        if (target, etype) in seen:
            continue
        seen.add((target, etype))
        item = {"target": target, "type": etype, "weight": round(float(e.get("weight", 1.0) or 1.0), 4)}
        if e.get("source"):
            item["source"] = str(e.get("source"))
        clean.append(item)
    return json.dumps(clean, ensure_ascii=False)


def edge_stats(db_path=None, workspace=None):
    con = _connect(db_path)
    try:
        q = "SELECT id, workspace, edges_json, status FROM nodes"
        params = []
        if workspace:
            q += " WHERE workspace=?"
            params.append(workspace)
        rows = con.execute(q, params).fetchall()
        valid_ids = {int(r["id"]) for r in rows}
        directed = 0
        invalid = 0
        duplicate = 0
        nodes_with_edges = 0
        undirected = set()
        by_type = Counter()
        for r in rows:
            sid = int(r["id"])
            seen = set()
            edges = _json_edges(r["edges_json"])
            if edges:
                nodes_with_edges += 1
            for e in edges:
                try:
                    tid = int(e.get("target"))
                except Exception:
                    invalid += 1
                    continue
                etype = str(e.get("type") or "related")
                key = (tid, etype)
                if key in seen:
                    duplicate += 1
                    continue
                seen.add(key)
                if tid not in valid_ids or tid == sid:
                    invalid += 1
                    continue
                directed += 1
                by_type[etype] += 1
                undirected.add(tuple(sorted((sid, tid))) + (etype,))
        orphan_sample = []
        orphan_ids = set()
        active_orphan_ids = set()
        for r in rows:
            edges = _json_edges(r["edges_json"])
            if not edges:
                nid = int(r["id"])
                orphan_ids.add(nid)
                status = r["status"] if "status" in r.keys() else None
                if status is None or status not in ("merged", "archived"):
                    active_orphan_ids.add(nid)
        orphan_count = len(orphan_ids)
        active_orphan_count = len(active_orphan_ids)
        if active_orphan_count > 0:
            id_list = list(active_orphan_ids)[:50]
            placeholders = ",".join("?" for _ in id_list)
            sample_rows = con.execute(
                "SELECT id, text FROM nodes WHERE id IN (" + placeholders + ") "
                "AND (status IS NULL OR status NOT IN ('merged','archived')) LIMIT 5",
                [str(i) for i in id_list]
            ).fetchall()
            orphan_sample = [{"id": r["id"], "text": (r["text"] or "")[:80]} for r in sample_rows]
        return {
            "nodes": len(rows),
            "directed_edges": directed,
            "unique_edges": len(undirected),
            "nodes_with_edges": nodes_with_edges,
            "orphan_nodes": orphan_count,
            "active_orphan_nodes": active_orphan_count,
            "orphan_sample": orphan_sample,
            "invalid_edges": invalid,
            "duplicate_edges": duplicate,
            "by_type": dict(by_type),
        }
    finally:
        con.close()


def category_stats(db_path=None):
    con = _connect(db_path)
    try:
        rows = con.execute(
            "SELECT COALESCE(NULLIF(category,''),'未分类') AS category, COUNT(*) AS n "
            "FROM nodes GROUP BY COALESCE(NULLIF(category,''),'未分类')"
        ).fetchall()
        counts = {r["category"]: r["n"] for r in rows}
        total = sum(counts.values())
        weak = sum(v for k, v in counts.items() if k in WEAK_CATEGORIES)
        return {
            "total": total,
            "categorized": total - weak,
            "uncategorized": weak,
            "coverage": round((total - weak) / max(total, 1) * 100, 2),
            "categories": counts,
        }
    finally:
        con.close()


def batch_classify(db_path=None, only_weak=True, dry_run=False):
    con = _connect(db_path)
    changed = 0
    by_category = Counter()
    try:
        cols = _columns(con)
        has_para = "para" in cols
        rows = con.execute("SELECT id, text, category, source, tags" + (", para" if has_para else "") + " FROM nodes").fetchall()
        for r in rows:
            old = r["category"]
            if only_weak and old not in WEAK_CATEGORIES:
                continue
            new_cat, _score = classify_text(r["text"], r["source"], r["tags"], db_path)
            if not new_cat or new_cat == old:
                continue

            merged_cat = CATEGORY_MERGE.get(new_cat, new_cat)
            by_category[new_cat] += 1
            changed += 1
            if dry_run:
                continue
            if has_para:
                old_para = r["para"] if "para" in r.keys() else ""
                new_para = old_para or PARA_BY_CATEGORY.get(merged_cat, "R")
                con.execute(
                    "UPDATE nodes SET category=?, para=?, updated_at=? WHERE id=? AND category=?",
                    (merged_cat, new_para, datetime.now().isoformat(), r["id"], old),
                )
            else:
                con.execute(
                    "UPDATE nodes SET category=?, updated_at=? WHERE id=? AND category=?",
                    (merged_cat, datetime.now().isoformat(), r["id"], old),
                )
            if old == "_待分类_" and merged_cat != "_待分类_":
                try:
                    kws = _keywords(r["text"], top_k=5)
                    if kws:
                        add_category_rule(merged_cat, kws, source="learned", db_path=db_path)
                except Exception:
                    import logging as _logging; _logging.getLogger("zhiluo.maintenance").warning("knowledge_maintenance.py: swallowed exception", exc_info=True)
        if not dry_run:
            con.commit()
        return {"changed": changed, "by_category": dict(by_category), "dry_run": dry_run}
    finally:
        con.close()


def _keywords(text, top_k=10):
    text = (text or "").strip()
    if not text:
        return []
    if _HAS_JIEBA:
        try:
            return [w.lower() for w in jieba.analyse.extract_tags(text, topK=top_k) if len(w.strip()) > 1]
        except Exception:
            import logging as _logging; _logging.getLogger("zhiluo.maintenance").warning("knowledge_maintenance.py: swallowed exception", exc_info=True)
    tokens = re.findall(r"[A-Za-z0-9_]{2,}|[\u4e00-\u9fff]{2,}", text.lower())
    stop = {"这个", "一个", "我们", "你们", "他们", "如果", "但是", "因为", "所以", "进行", "可以", "需要", "已经", "没有"}
    return [t for t in tokens if t not in stop][:top_k]


def _add_edge_to_map(edges_by_id, sid, tid, weight, etype="related"):
    if sid == tid:
        return False
    edges = edges_by_id.setdefault(sid, [])
    for e in edges:
        if int(e.get("target", -1)) == tid and (e.get("type") or "related") == etype:
            if float(e.get("weight", 0) or 0) < weight:
                e["weight"] = round(weight, 4)
            return False
    edges.append({"target": int(tid), "type": etype, "weight": round(float(weight), 4), "source": "auto_maintenance"})
    return True


def build_related_edges(db_path=None, workspace=None, max_links_per_node=2, min_score=0.46, dry_run=False, target_ids=None):
    con = _connect(db_path)
    added = 0
    touched = set()
    try:
        q = "SELECT id, text, workspace, category, edges_json FROM nodes WHERE status IS NULL OR status NOT IN ('merged','archived')"
        params = []
        if workspace:
            q += " AND workspace=?"
            params.append(workspace)
        rows = [dict(r) for r in con.execute(q, params).fetchall()]
        if target_ids is not None:
            target_set = {int(t) for t in target_ids}
            rows = [r for r in rows if int(r["id"]) in target_set]
        id_to_row = {int(r["id"]): r for r in rows}
        edges_by_id = {int(r["id"]): _json_edges(r.get("edges_json")) for r in rows}

        keyword_index = defaultdict(set)
        keywords_by_id = {}
        by_category = defaultdict(list)
        for r in rows:
            nid = int(r["id"])
            kws = set(_keywords(r["text"], top_k=12))
            keywords_by_id[nid] = kws
            for kw in kws:
                keyword_index[kw].add(nid)
            by_category[r.get("category") or "通用"].append(nid)

        for nid, row in id_to_row.items():
            existing_targets = {int(e.get("target")) for e in edges_by_id.get(nid, []) if str(e.get("target", "")).isdigit()}
            kws = keywords_by_id.get(nid, set())
            if not kws:
                continue
            candidates = Counter()
            for kw in kws:
                for cid in keyword_index.get(kw, ()):
                    if cid != nid and cid not in existing_targets:
                        candidates[cid] += 1
            scored = []
            cat = row.get("category") or "通用"
            for cid, shared in candidates.items():
                ckws = keywords_by_id.get(cid, set())
                if not ckws:
                    continue
                denom = max(len(kws | ckws), 1)
                score = shared / denom
                if (id_to_row[cid].get("category") or "通用") == cat:
                    score += 0.18
                if score >= min_score:
                    scored.append((score, cid))
            scored.sort(reverse=True)
            current_added = 0
            for score, cid in scored[:max_links_per_node]:
                if _add_edge_to_map(edges_by_id, nid, cid, score):
                    added += 1
                    touched.add(nid)
                    current_added += 1
                if _add_edge_to_map(edges_by_id, cid, nid, score):
                    added += 1
                    touched.add(cid)
                if current_added >= max_links_per_node:
                    break

        if not dry_run:
            now = datetime.now().isoformat()
            for nid in touched:
                con.execute(
                    "UPDATE nodes SET edges_json=?, updated_at=? WHERE id=?",
                    (_dump_edges(edges_by_id[nid]), now, nid),
                )
            con.commit()

        # 孤儿救助第二遍
        orphan_ids = [nid for nid in id_to_row if len(edges_by_id.get(nid, [])) == 0]
        orphan_rescue_added = 0
        orphan_rescue_touched = set()

        if orphan_ids:
            for nid in orphan_ids:
                row = id_to_row[nid]
                kws = keywords_by_id.get(nid, set())
                if not kws:
                    continue
                candidates = Counter()
                existing_targets = {int(e.get("target")) for e in edges_by_id.get(nid, []) if str(e.get("target", "")).isdigit()}
                for kw in kws:
                    for cid in keyword_index.get(kw, ()):
                        if cid != nid and cid not in existing_targets:
                            candidates[cid] += 1
                scored = []
                cat = row.get("category") or "通用"
                for cid, shared in candidates.items():
                    ckws = keywords_by_id.get(cid, set())
                    if not ckws:
                        continue
                    denom = max(len(kws | ckws), 1)
                    score = shared / denom
                    if (id_to_row[cid].get("category") or "通用") == cat:
                        score += 0.18
                    if score >= 0.25:
                        scored.append((score, cid))
                scored.sort(reverse=True)
                for score, cid in scored[:1]:
                    if _add_edge_to_map(edges_by_id, nid, cid, score):
                        orphan_rescue_added += 1
                        orphan_rescue_touched.add(nid)
                    if _add_edge_to_map(edges_by_id, cid, nid, score):
                        orphan_rescue_added += 1
                        orphan_rescue_touched.add(cid)
                    break

            if not dry_run:
                now2 = datetime.now().isoformat()
                for nid in orphan_rescue_touched:
                    con.execute(
                        "UPDATE nodes SET edges_json=?, updated_at=? WHERE id=?",
                        (_dump_edges(edges_by_id[nid]), now2, nid),
                    )
                con.commit()

        return {
            "added_edges": added,
            "touched_nodes": len(touched),
            "dry_run": dry_run,
            "orphan_rescue": {
                "orphans_before": len(orphan_ids),
                "added_edges": orphan_rescue_added,
                "touched_nodes": len(orphan_rescue_touched),
            },
        }
    finally:
        con.close()


def repair_edge_json(db_path=None, dry_run=False):
    con = _connect(db_path)
    changed = 0
    removed = 0
    try:
        valid = {int(r[0]) for r in con.execute("SELECT id FROM nodes").fetchall()}
        rows = con.execute("SELECT id, edges_json FROM nodes").fetchall()
        for r in rows:
            nid = int(r["id"])
            edges = _json_edges(r["edges_json"])
            clean = []
            seen = set()
            for e in edges:
                try:
                    tid = int(e.get("target"))
                except Exception:
                    removed += 1
                    continue
                etype = str(e.get("type") or "related")
                if tid == nid or tid not in valid or (tid, etype) in seen:
                    removed += 1
                    continue
                seen.add((tid, etype))
                clean.append(e)
            new_raw = _dump_edges(clean)
            if new_raw != (r["edges_json"] or "[]"):
                changed += 1
                if not dry_run:
                    con.execute("UPDATE nodes SET edges_json=?, updated_at=? WHERE id=?", (new_raw, datetime.now().isoformat(), nid))
        if not dry_run:
            con.commit()
        return {"changed_nodes": changed, "removed_edges": removed, "dry_run": dry_run}
    finally:
        con.close()


def maintain(db_path=None, classify=True, link=True, repair=True, dry_run=False):
    """知识库综合维护（基础版精简）。

    执行：批量分类 + 自动补边 + 边修复，去掉日报/合成/合并/自动学习。
    """
    before_categories = category_stats(db_path)
    before_edges = edge_stats(db_path)
    backup = "" if dry_run else backup_db(db_path)
    result = {
        "backup": backup,
        "before": {"categories": before_categories, "edges": before_edges},
        "actions": {},
    }

    if classify:
        try:
            result["actions"]["classify"] = batch_classify(db_path, only_weak=True, dry_run=dry_run)
        except Exception as e:
            result["actions"]["classify"] = {"error": str(e)}
    if repair:
        try:
            result["actions"]["repair_edges"] = repair_edge_json(db_path, dry_run=dry_run)
        except Exception as e:
            result["actions"]["repair_edges"] = {"error": str(e)}
    if link:
        try:
            result["actions"]["link"] = build_related_edges(db_path, dry_run=dry_run)
        except Exception as e:
            result["actions"]["link"] = {"error": str(e)}

    # 孤儿节点预警
    if link and not dry_run:
        try:
            after_edges = edge_stats(db_path)
            active_orphan = after_edges.get("active_orphan_nodes", 0)
            if active_orphan > 100:
                result["actions"]["orphan_warning"] = {
                    "active_orphan_nodes": active_orphan,
                    "orphan_nodes": after_edges.get("orphan_nodes", 0),
                    "warning": "活跃孤立节点过多(>100)，建议运行 deep_orphan_link()",
                }
        except Exception:
            import logging as _logging; _logging.getLogger("zhiluo.maintenance").warning("knowledge_maintenance.py: swallowed exception", exc_info=True)

    result["after"] = {"categories": category_stats(db_path), "edges": edge_stats(db_path)}
    return result


def format_maintenance_report(result):
    before_c = result["before"]["categories"]
    after_c = result["after"]["categories"]
    before_e = result["before"]["edges"]
    after_e = result["after"]["edges"]
    lines = ["[Z] 知识库维护完成"]
    if result.get("backup"):
        lines.append("备份: %s" % result["backup"])
    lines.append("分类覆盖: %.2f%% -> %.2f%%，未分类: %d -> %d" % (
        before_c["coverage"], after_c["coverage"],
        before_c["uncategorized"], after_c["uncategorized"],
    ))
    lines.append("关系边: %d -> %d directed / %d unique，孤立节点: %d -> %d" % (
        before_e["directed_edges"], after_e["directed_edges"],
        after_e["unique_edges"], before_e["orphan_nodes"], after_e["orphan_nodes"],
    ))
    actions = result.get("actions", {})
    if "classify" in actions:
        lines.append("批量分类: %d 条" % actions["classify"].get("changed", 0))
    if "repair_edges" in actions:
        lines.append("边清理: %d 节点，移除 %d 条异常边" % (
            actions["repair_edges"].get("changed_nodes", 0),
            actions["repair_edges"].get("removed_edges", 0),
        ))
    if "link" in actions:
        lines.append("自动补边: %d 条，触达 %d 节点" % (
            actions["link"].get("added_edges", 0),
            actions["link"].get("touched_nodes", 0),
        ))
    top = sorted(after_c["categories"].items(), key=lambda x: -x[1])[:10]
    lines.append("主要分类: " + " ".join("%s:%d" % (k, v) for k, v in top))
    return "\n".join(lines)


def deep_orphan_link(db_path=None, min_score=0.25, max_links=1, dry_run=False):
    con = _connect(db_path)
    try:
        all_ids = {int(r[0]) for r in con.execute(
            "SELECT id FROM nodes WHERE status IS NULL OR status NOT IN ('merged','archived')"
        ).fetchall()}
        edged = set()
        rows = con.execute("SELECT id, edges_json FROM nodes").fetchall()
        for r in rows:
            edges = _json_edges(r["edges_json"])
            if edges:
                edged.add(int(r["id"]))
        orphans = all_ids - edged
        if not orphans:
            return {"orphans_found": 0, "added_edges": 0, "touched_nodes": 0, "message": "没有孤立节点"}
        result = build_related_edges(
            db_path=db_path,
            max_links_per_node=max_links,
            min_score=min_score,
            dry_run=dry_run,
        )
        result["orphans_found"] = len(orphans)
        return result
    finally:
        con.close()
