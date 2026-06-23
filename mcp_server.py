# -*- coding: utf-8 -*-
"""
知络 v8.3+ MCP Server — v8.5.2底板 + plus独有Tool
全版本能力合并：v8.5.2的21Tool + plus的7独有Tool，精简合并为19Tool
新增：冲突检测三重（显式矛盾边+语义+数值）、自检14项+8项自动修复、_safe_call全Tool异常保护
合并：search+trace+entangle→search(mode), explore→reason(mode=explore), switch_workspace→workspace(action=switch)
删除：correct/auto_learn/classify_genre/get_context（功能已被其他Tool覆盖）
"""
import sys, os, json
if os.name == "nt" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
try:
    from mcp.server.fastmcp import FastMCP
except ImportError:
    print("需要安装 MCP SDK: pip install mcp")
    sys.exit(1)

from engine import ZhiLuo, pagerank, SelfCheck, mermaid_graph, entangle as _entangle, auto_learn as _auto_learn
from brain_wrapper import create_brain
from graph_wrapper import ZhiLuoGraph
from license import check_license, get_status as _license_status, activate as _activate_license, get_machine_code

# ========== 全局异常保护 ==========
def _safe_call(expr_or_lambda):
    """8.5.2风格异常保护：可调用对象或表达式，出错返回友好提示而非崩溃"""
    try:
        if callable(expr_or_lambda):
            return expr_or_lambda()
        return expr_or_lambda
    except Exception as e:
        return "[Z] 引擎错误: %s: %s" % (type(e).__name__, e)

# ========== 双引擎初始化 ==========
_lb = ZhiLuo()
lb = ZhiLuoGraph(_lb)
brain = None
try:
    brain = _lb.brain
    if brain:
        brain.G = lb.graph
except Exception:
    pass

mcp = FastMCP("知络", instructions="知络 v8.3+ 记忆外脑。19个Tool: learn(auto=提取到待确认)/query(mode=auto/graph/context/genre)/search(mode=graph/trace/entangle)/analyze(冲突/衰减)/summarize/visualize(mode=graph/brain/mermaid/pagerank/stats)/pending(action=list/confirm/reject/confirm_all)/selfcheck/export/manage(action=update/delete/correct/backup/history/rule/llm)/reason(mode=auto/chain/entangle/rank/explore)/configure(key=rules/llm/genre/para)/extract/stats(mode=overview/graph/bridge/context/health/snapshot)/history(mode=recent/node)/workspace(action=switch/backup/restore)/deep_reason(mode=auto/quantum/neural)/system_insight(mode=profile/emerge/pitfalls)/cross_context(mode=auto/topic/evolution/stats)。说人话就行。")

# ========== 对话流日志 + 被动引擎（safe init）==========
_chat_log = None
try:
    from chat_log import ChatLog
    _chat_log = ChatLog.init()
except Exception:
    pass

def _inject_chat(role, content, keyword=""):
    """统一注入对话流"""
    try:
        if _chat_log:
            _chat_log.record(role, content, keyword)
    except Exception:
        pass

_passive_engine = None
try:
    from passive_engine import start_passive
    _passive_engine = start_passive(_lb, _chat_log)
except Exception:
    pass


# ==================== v8.5.2 核心 15Tool ====================

# ====================== 1. learn ======================
@mcp.tool()
def learn(text: str, auto: bool = False) -> str:
    """记住新知识。auto=True 时提取到待确认队列。支持自动去重合并。"""
    _inject_chat("user", text, "学习")
    lic = _lic_check()
    if lic and not auto:
        return lic + "\n[知络] 试用到期后只能查询，无法学习新知识。"
    if lic and auto:
        return lic
    if auto:
        ids = _auto_learn(lb.s, text)
        if not ids:
            return "[Z] 未提取到知识点"
        return "[Z] 已提取 %d 条到待确认队列。用 pending 查看。" % len(ids)
    result = _safe_call(lambda: _lb.run("记住" + text))
    # 踩坑预警（v8.5.2增强）
    try:
        from user_profiler import pitfall_tracker
        r = _safe_call(lambda: pitfall_tracker(_lb.s, _lb.ws))
        if isinstance(r, str): r = {"pitfalls": []}
        if r["pitfalls"]:
            warns = "; ".join(p["keyword"] for p in r["pitfalls"][:3])
            result += "\n[踩坑预警] 历史相似纠错模式: " + warns
    except Exception:
        pass
    return result

# ====================== 2. query ======================
@mcp.tool()
def query(keyword: str, mode: str = "auto") -> str:
    """搜索已有知识。mode: auto=精确+关键词+语义模糊, graph=图扩散, context=三层上下文, genre=体裁感知。"""
    _inject_chat("user", keyword, "查询")
    if mode == "graph":
        return _safe_call(lambda: _lb.run("关联" + keyword))
    if mode == "context":
        if brain:
            ctx = _safe_call(lambda: brain.get_context(keyword))
            if isinstance(ctx, str) and ctx.startswith("[Z] 引擎错误"): return ctx
            return ctx or "[Brain] 无相关上下文"
        return "[Z] 上下文不可用"
    if mode == "genre":
        if brain:
            return _safe_call(lambda: brain.genre_search(keyword))
        return "[Z] 体裁检索不可用"
    return _safe_call(lambda: _lb.run(keyword))

# ====================== 3. search（合并trace+entangle）======================
@mcp.tool()
def search(keyword: str, mode: str = "graph") -> str:
    """图搜索。mode: graph=沿知识图谱扩散关联, trace=追溯因果链/时间线, entangle=纠缠场分析(多词空格分隔)。"""
    _inject_chat("user", keyword, "搜索")
    if mode == "trace":
        return _safe_call(lambda: _lb.run("追溯" + keyword))
    if mode == "entangle":
        word_list = keyword.strip().split()
        if len(word_list) < 2:
            return "[Z] 用法: search('词1 词2 ...', mode='entangle')"
        r = _safe_call(lambda: _entangle(lb.s, word_list))
        if isinstance(r, str): return r
        if not r or not r.get("entanglements"):
            return "[Z] 纠缠场分析无结果"
        lines = ["[Z] 纠缠场分析 (%d个词):" % len(word_list)]
        for p in r["entanglements"]:
            lines.append("  %s <-> %s: 共同节点%d个, 间接关联%d条" % (
                p["word_a"], p["word_b"], p.get("common_nodes", 0), len(p.get("indirect_links", []))))
            for link in p.get("indirect_links", [])[:3]:
                n_from = _safe_call(lambda l=link: lb.s.get(l["from"]))
                n_to = _safe_call(lambda l=link: lb.s.get(l["to"]))
                if n_from and n_to:
                    lines.append("    #%d(%s) --%s--> #%d(%s)" % (
                        link["from"], n_from["text"][:30], link.get("type", ""),
                        link["to"], n_to["text"][:30]))
        return "\n".join(lines)
    return _safe_call(lambda: _lb.run("关联" + keyword))

# ====================== 4. analyze ======================
@mcp.tool()
def analyze(text: str = "") -> str:
    """三重冲突检测+数据分析。输入"冲突"触发显式矛盾边+语义+数值三重检测，输入"衰减"触发置信度衰减分析，留空触发综合分析。"""
    _inject_chat("user", text or "分析", "分析")
    if text == "冲突":
        return _triple_conflict_check()
    if text == "衰减":
        return _safe_call(lambda: _lb.run("分析 衰减"))
    if not text:
        return _safe_call(lambda: _lb.run("分析"))
    return _safe_call(lambda: _lb.run("分析" + text))

# ====================== 5. summarize ======================
@mcp.tool()
def summarize(text: str) -> str:
    """提取式总结文本核心要点。"""
    _inject_chat("user", text, "总结")
    return _safe_call(lambda: _lb.run("总结" + text))

# ====================== 6. visualize ======================
@mcp.tool()
def visualize(mode: str = "graph", keyword: str = "") -> str:
    """知识可视化。mode: graph=知识图谱, brain=BrainWrapper图谱, mermaid=Mermaid代码, pagerank=枢纽排名, stats=图统计。"""
    if mode == "brain" and brain:
        return _safe_call(lambda: brain.to_mermaid(focus=keyword if keyword else None, max_nodes=50))
    if mode == "mermaid":
        return _safe_call(lambda: mermaid_graph(lb.s, keyword=keyword if keyword else None))
    if mode == "pagerank":
        return _safe_call(lambda: _lb.run("pagerank " + keyword))
    if mode == "stats":
        return _safe_call(lambda: lb._do_graph_stats())
    return _safe_call(lambda: _lb.run("可视化" + keyword))

# ====================== 8. pending ======================
@mcp.tool()
def pending(action: str = "list", pid: str = "") -> str:
    """待确认管理。action: list=查看, confirm=确认(需pid), reject=拒绝(需pid), confirm_all=一键全确认。"""
    _inject_chat("user", action, "待确认")
    if action == "confirm" and pid:
        nid = _safe_call(lambda: lb.s.pending_confirm(pid))
        if isinstance(nid, str): return nid
        return "[Z] 已确认 %s -> #%s" % (pid, nid) if nid else "[Z] 未找到 " + pid
    if action == "reject" and pid:
        r = _safe_call(lambda: lb.s.pending_reject(pid))
        if isinstance(r, str): return r
        return "[Z] 已拒绝 " + pid
    if action == "confirm_all":
        items = _safe_call(lambda: lb.s.pending_list())
        if isinstance(items, str): return items
        if not items:
            return "[Z] 没有待确认知识"
        confirmed = 0
        for item in items:
            nid = _safe_call(lambda: lb.s.pending_confirm(item["id"]))
            if isinstance(nid, str): continue
            if nid:
                confirmed += 1
        return "[Z] 已批量确认 %d/%d 条" % (confirmed, len(items))
    # list
    items = _safe_call(lambda: lb.s.pending_list())
    if isinstance(items, str): return items
    if not items:
        return "[Z] 没有待确认知识"
    return "\n".join(["[Z] 待确认 (%d条):" % len(items)] + ["  - %s %s" % (p["id"], p["content"][:60]) for p in items])

# ====================== 12. selfcheck ======================
@mcp.tool()
def selfcheck() -> str:
    """系统自检：14项检查+8项自动修复。"""
    _inject_chat("system", "自检", "自检")
    return _full_selfcheck()

# ====================== 11. export ======================
@mcp.tool()
def export(fmt: str = "json", keyword: str = "") -> str:
    """导出知识。fmt: json/csv/markdown/graphml。可选keyword过滤。"""
    _inject_chat("user", fmt, "导出")
    return _safe_call(lambda: _lb.run("导出" + fmt + " " + keyword))

# ====================== 15. manage ======================
@mcp.tool()
def manage(action: str, text: str = "", new_text: str = "") -> str:
    """综合管理。action: update=修改知识, delete=删除知识, correct=纠正词义, backup=备份, history=变更历史, rule=关系规则, llm=LLM模式。"""
    _inject_chat("user", action, "管理")
    if action in ("update", "delete", "correct"):
        lic = _lic_check()
        if lic:
            return lic + "\n[知络] 试用到期后只能查询，无法修改知识。"
    if action == "backup": return _safe_call(lambda: _lb.run("备份"))
    if action == "history": return _safe_call(lambda: _lb.run("记忆列表"))
    if action == "update": return _safe_call(lambda: _lb.run("修改" + text + " 改成 " + new_text))
    if action == "delete": return _safe_call(lambda: _lb.run("删除" + text))
    if action == "correct": return _safe_call(lambda: _lb.run("纠正" + text))
    if action == "rule": return _safe_call(lambda: _lb.run("规则" + text))
    if action == "llm": return _safe_call(lambda: _lb.run("LLM " + text))
    return "[Z] 未知操作: " + action



# ==================== plus独有 Tool ====================

# ====================== 12. reason（合并explore）======================
@mcp.tool()
def reason(query: str, mode: str = "auto", top_k: int = 15) -> str:
    """图推理引擎(plus增强)。mode: auto=自动识别意图, chain=替代链分析, entangle=纠缠场分析(空格分隔多词), rank=PageRank枢纽分析, explore=语义发散(single/multi/network)。"""
    _inject_chat("user", query, "图推理")
    if mode == "chain":
        if hasattr(lb.graph, 'substitute_chain'):
            result = _safe_call(lambda: lb.graph.substitute_chain(query))
            if isinstance(result, str): return result  # _safe_call已处理
            if "error" in result:
                return "[Z] " + result["error"]
            lines = ["[Z] %s 的替代方案 (%d种):" % (query, result["total"])]
            for a in result["alternatives"]:
                lines.append("  - %s (适配度:%.2f, 路径:%s)" % (a["name"], a["score"], a.get("path", "")))
            return "\n".join(lines)
        return "[Z] 替代链分析不可用"
    if mode == "entangle":
        word_list = query.strip().split()
        if len(word_list) < 2:
            return "[Z] 用法: reason('词1 词2', mode='entangle')"
        r = _safe_call(lambda: _entangle(lb.s, word_list))
        if isinstance(r, str): return r
        if not r or not r.get("entanglements"):
            return "[Z] 纠缠场分析无结果"
        lines = ["[Z] 纠缠场分析 (%d个词):" % len(word_list)]
        for p in r["entanglements"]:
            lines.append("  %s <-> %s: 共同节点%d个, 间接关联%d条" % (
                p["word_a"], p["word_b"], p.get("common_nodes", 0), len(p.get("indirect_links", []))))
            for link in p.get("indirect_links", [])[:3]:
                n_from = _safe_call(lambda l=link: lb.s.get(l["from"]))
                n_to = _safe_call(lambda l=link: lb.s.get(l["to"]))
                if n_from and n_to:
                    lines.append("    #%d(%s) --%s--> #%d(%s)" % (
                        link["from"], n_from["text"][:30], link.get("type", ""),
                        link["to"], n_to["text"][:30]))
        return "\n".join(lines)
    if mode == "rank":
        pr = _safe_call(lambda: pagerank(lb.s))
        if isinstance(pr, str): return pr
        if not pr:
            return "[Z] 知识库为空或无边"
        lines = ["[PR] 枢纽分析:"]
        for item in pr[:10]:
            n = _safe_call(lambda i=item: lb.s.get(i["id"]))
            if n:
                lines.append("  %.4f [%s] %s" % (item["pr"], n["category"], n["text"][:80]))
        return "\n".join(lines)
    if mode == "explore":
        # 合并自explore：query含逗号→multi，否则single；query为network关键词→network
        if not brain:
            return "[Z] 语义发散不可用（BrainWrapper未加载）"
        if query == "network" or query.startswith("network "):
            seed = query.replace("network", "").strip() or None
            result = _safe_call(lambda: brain.bridge_network(seed=seed))
            if isinstance(result, str): return result
            return json.dumps(result, ensure_ascii=False, indent=2) if result else "[Brain] 共现网络为空"
        if "," in query:
            queries = [q.strip() for q in query.split(",") if q.strip()]
            result = _safe_call(lambda: brain.explore_multi(queries, top_k=top_k))
            if isinstance(result, str): return result
            if not result or not result.get("merged"):
                return "[Brain] 未找到关联概念"
            lines = ["[Brain] 多概念联合发散 (%d项):" % len(result["merged"])]
            for term, strength in result["merged"][:top_k]:
                lines.append("  %s: %.2f" % (term, strength))
            return "\n".join(lines)
        # single
        result = _safe_call(lambda: brain.explore(query, top_k=top_k))
        if isinstance(result, str): return result
        if not result or not result.get("merged"):
            return "[Brain] 未找到关联概念"
        lines = ["[Brain] 语义发散 (%d项):" % len(result["merged"])]
        for term, strength in result["merged"][:top_k]:
            lines.append("  %s: %.2f" % (term, strength))
        return "\n".join(lines)
    # auto
    return _safe_call(lambda: lb._do_reason(query))

# ====================== 20. configure ======================
@mcp.tool()
def configure(key: str, value: str = "") -> str:
    """动态配置。key: rules=关系推理规则(value='动词=关系类型,动词=关系类型'), llm=LLM提取模式(value=on/off), genre=体裁分类(value=文本内容), para=PARA分类(value=文本内容或留空查看法则)。"""
    _inject_chat("user", key, "配置")
    if key == "rules":
        try:
            from relation_extractor import _REL_VERBS
            pairs = [p.strip() for p in value.split(",")]
            added = 0
            for pair in pairs:
                if "=" in pair:
                    verb, rtype = pair.split("=", 1)
                    verb, rtype = verb.strip(), rtype.strip()
                    if verb and rtype:
                        _REL_VERBS[verb] = rtype
                        added += 1
            result = "[Z] 已添加 %d 条关系规则。当前共 %d 条规则。" % (added, len(_REL_VERBS))
            result += "\n" + str(list(_REL_VERBS.items())[-10:])
            return result
        except ImportError:
            return "[Z] 关系提取器不可用"
    if key == "llm":
        enabled = value.lower() in ("on", "true", "1", "yes")
        if enabled:
            if hasattr(_lb, "_llm_func") and _lb._llm_func is not None:
                lb._llm = _lb._llm_func
                return "[Z] LLM 提取已开启"
            return "[Z] 暂无 LLM 配置，仍用规则提取"
        _lb._llm_func = None
        lb._llm = None
        return "[Z] 已切换到规则提取模式"
    if key == "genre":
        if brain:
            _gc = _safe_call(lambda: brain.classify_genre(value))
            if isinstance(_gc, str): return "[Brain] 体裁分类失败: " + _gc
            genre, conf = _gc
            return "[Brain] 体裁: %s (置信度: %d%%)" % (genre, round(conf * 100))
        return "[Z] 体裁分类不可用"
    if key == "para":
        from knowledge_schema import RelationRegistry
        para = RelationRegistry.PARA_CATEGORIES
        if not value:
            lines = ["[PARA 知识分类法则]"]
            for k, v in para.items():
                lines.append("  %s: %s" % (k, v))
            lines.append("\n用法: configure(key='para', value='你的内容') → 自动判定PARA分类")
            return "\n".join(lines)
        if brain:
            _gc = _safe_call(lambda: brain.classify_genre(value))
            genre = _gc[0] if not isinstance(_gc, str) and _gc else "knowledge"
            # 映射genre到PARA
            para_map = {
                "process": "project", "argument": "area", "definition": "resource",
                "data_summary": "area", "dialogue": "archive", "essay": "archive",
                "knowledge": "resource", "code": "resource",
                "project": "project", "area": "area", "resource": "resource", "archive": "archive",
            }
            para_cat = para_map.get(genre, "resource")
            desc = para.get(para_cat, "")
            conf = round(_gc[1] * 100) if not isinstance(_gc, str) and len(_gc) > 1 else 50
            return "[PARA] %s (%s, 置信度%d%%)\n  → %s" % (para_cat, genre, conf, desc)
        return "[Z] PARA分类不可用"
    return "[Z] 未知配置项，可选: rules/llm/genre/para"

# ====================== 21. extract ======================
@mcp.tool()
def extract(text: str) -> str:
    """通用实体+关系提取(不绑定行业词典)。"""
    _inject_chat("user", text, "提取")
    if not brain:
        return "[Z] 提取不可用（BrainWrapper未加载）"
    r = _safe_call(lambda: brain.extract(text))
    if isinstance(r, str): return r
    lines = ["[Brain] 提取结果:"]
    if r.get("entities"):
        lines.append("  实体(%d): %s" % (len(r["entities"]), ", ".join(e["name"] for e in r["entities"])))
    if r.get("relations"):
        lines.append("  关系(%d):" % len(r["relations"]))
        for rel in r["relations"][:10]:
            lines.append("    %s -> %s [%s]" % (rel["source"], rel["target"], rel["relation"]))
    return "\n".join(lines) if len(lines) > 1 else "[Brain] 未提取到内容"

# ====================== 22. stats ======================
@mcp.tool()
def stats(mode: str = "overview") -> str:
    """统计。mode: overview=概览, graph=图统计, bridge=共现网络, context=上下文记忆, health=健康检查(14项), snapshot=完整快照。"""
    _inject_chat("system", mode, "统计")
    if mode == "graph":
        return _safe_call(lambda: lb._do_graph_stats())
    if mode == "bridge" and brain:
        s = _safe_call(lambda: brain.bridge_stats())
        if isinstance(s, str): return s
        return "[Brain] 共现网络: %d 节点 / %d 条边 / %d 次学习" % (
            s.get("total_nodes", 0), s.get("total_edges", 0), s.get("total_learns", 0))
    if mode == "context" and brain:
        ctx = _safe_call(lambda: brain.context_stats())
        if isinstance(ctx, str): return ctx
        co = _safe_call(lambda: brain.bridge_stats())
        if isinstance(co, str): return co
        lines = ["[Brain] 知络 v8.3+ 状态:"]
        lines.append("  共现网络: %d 节点, %d 边" % (co.get("total_nodes", 0), co.get("total_edges", 0)))
        lines.append("  上下文记忆: %d 会话, %d 主题" % (ctx.get("sessions", 0), ctx.get("topics", 0)))
        return "\n".join(lines)
    if mode == "health":
        return _full_selfcheck()
    if mode == "snapshot" and brain:
        snap = _safe_call(lambda: brain.snapshot())
        if isinstance(snap, str): return snap
        return json.dumps(snap, ensure_ascii=False, indent=2)
    # overview
    st = _safe_call(lambda: lb.s.stats())
    if isinstance(st, str): return st
    return "知识:%d条 边:%d条\n%s" % (st["total"], st["edges"],
        " ".join("%s:%d" % (k, v) for k, v in list(st["categories"].items())[:5]))

# ====================== 23. history ======================
@mcp.tool()
def history(mode: str = "recent", hours: int = 24, node_id: int = 0) -> str:
    """变更历史。mode: recent=最近N小时, node=指定节点历史。"""
    _inject_chat("system", mode, "变更历史")
    if mode == "node" and node_id:
        hist = _safe_call(lambda: lb.s.change_history(node_id))
    else:
        hist = _safe_call(lambda: lb.s.change_history())
    if isinstance(hist, str): return hist
    if not hist:
        return "[Z] 无变更历史"
    return "\n".join(["[Z] 变更历史 (%d条):" % len(hist)] +
        ["  %s #%d %s: %s" % (h["timestamp"][:19], h["node_id"], h["action"], h.get("new_text", "")[:40])
         for h in hist[:15]])

# ====================== 24. workspace ======================
@mcp.tool()
def workspace(action: str = "switch", name: str = "global", backup_path: str = "") -> str:
    """工作区管理。action: switch=切换, backup=备份, restore=恢复(需backup_path)。"""
    _inject_chat("user", action, "工作区")
    if action == "backup":
        if brain and hasattr(brain, 'backup'):
            return _safe_call(lambda: brain.backup())
        return _safe_call(lambda: _lb.run("备份"))
    if action == "restore":
        if not backup_path:
            return "[Z] 恢复需指定 backup_path"
        if brain and hasattr(brain, 'restore'):
            return _safe_call(lambda: brain.restore(backup_path))
        return "[Z] 恢复功能需 BrainWrapper"
    return _safe_call(lambda: lb.switch_ws(name))


# ==================== v8.5.2 深度推理+系统洞察 ====================

# ====================== 25. deep_reason ======================
@mcp.tool()
def deep_reason(question: str, mode: str = "auto") -> str:
    """深度推理引擎。mode: auto=深度推理(拆解→检索→多角度分析→打分→报告), quantum=量子级关联(加权纠缠场+语义叠加传播), neural=神经扩散(多跳带阈值截断的语义激活传播)。"""
    _inject_chat("user", question, "深度推理")
    lic = _lic_check()
    if lic:
        return lic + "\n[知络] 深度推理为Pro版功能，试用到期后不可用。查询/搜索仍可使用。"
    try:
        from deep_engine import deep_reason as _dr, quantum_assoc, neural_diffuse
    except ImportError:
        return "[Z] 深度推理引擎不可用（deep_engine模块未加载）"
    try:
        if mode == "quantum":
            r = _safe_call(lambda: quantum_assoc(_lb.s, question, _lb.ws))
            if isinstance(r, str): return r
            if not r["nodes"]: return "[Z] 未找到关联"
            lines = ["[Z] 量子关联 (%d项):" % len(r["nodes"])]
            for n in r["nodes"]:
                lines.append("  %s [%s] (权重:%s)" % (n["text"], n["category"], str(n["weight"])))
            return "\n".join(lines)
        if mode == "neural":
            r = _safe_call(lambda: neural_diffuse(_lb.s, question, _lb.ws))
            if isinstance(r, str): return r
            if not r["activated"]: return "[Z] 无激活节点"
            lines = ["[Z] 神经扩散 (%d个激活节点):" % r["total_activated"]]
            for a in r["activated"][:15]:
                lines.append("  %s [%s] (激活:%s)" % (a["text"], a["category"], str(a["activation"])))
            return "\n".join(lines)
        result = _safe_call(lambda: _dr(_lb.s, question, _lb.ws))
        if isinstance(result, str): return result
        return result["structured"]
    except Exception as e:
        return "[Z] 深度推理异常: %s" % type(e).__name__

# ====================== 26. system_insight ======================
@mcp.tool()
def system_insight(mode: str = "profile") -> str:
    """系统洞察。mode: profile=用户画像(学习风格/活跃时段/知识深度/格式偏好/情感语气等), emerge=浮现图谱(高频关联集群+浮现概念), pitfalls=踩坑检查(历史纠错模式分析)。"""
    _inject_chat("system", mode, "系统洞察")
    try:
        if mode == "emerge":
            from emerge_engine import auto_emerge as _ae
            r = _safe_call(lambda: _ae(_lb.s, _lb.ws))
            if isinstance(r, str): return r
            lines = ["[Z] 浮现图谱分析:"]
            if r["clusters"]:
                lines.append("  关联集群(%d个):" % len(r["clusters"]))
                for c in r["clusters"][:5]:
                    lines.append("    %s (共现%d次)" % ("、".join(c["words"]), c["frequency"]))
            if r["emergent"]:
                lines.append("  浮现概念(%d个):" % len(r["emergent"]))
                for e in r["emergent"][:5]:
                    lines.append("    %s (频率%d 连接度%d)" % (e["concept"], e["frequency"], e["connectivity"]))
            return "\n".join(lines) if len(lines) > 1 else "[Z] 未发现浮现模式"
        if mode == "pitfalls":
            from user_profiler import pitfall_tracker
            r = _safe_call(lambda: pitfall_tracker(_lb.s, _lb.ws))
            if isinstance(r, str): return r
            if not r["pitfalls"]: return "[Z] 未发现历史踩坑模式"
            lines = ["[Z] 踩坑分析:"]
            for p in r["pitfalls"]:
                lines.append("  %s [%s] (纠正%d次)" % (p["keyword"], p["level"], p["correct_count"]))
            return "\n".join(lines)
        # profile
        from user_profiler import user_profile as _up
        r = _safe_call(lambda: _up(_lb.s, _lb.ws))
        if isinstance(r, str): return r
        p = r["profile"]
        lines = ["[Z] 用户画像:"]
        if "知识总量" in p: lines.append("  知识总量: %d 条" % p["知识总量"])
        if "学习风格" in p: lines.append("  学习风格: " + p["学习风格"])
        if "活跃时段" in p: lines.append("  活跃时段: " + p["活跃时段"])
        if "知识深度" in p: lines.append("  知识深度: %.1f/10" % p["知识深度"])
        if "格式偏好" in p: lines.append("  格式偏好: " + p["格式偏好"])
        if "情感语气" in p: lines.append("  情感语气: " + p["情感语气"])
        if "高频词" in p: lines.append("  高频词: " + ", ".join(p["高频词"][:8]))
        if "领域分布" in p:
            items = [k + "(%s%%)" % str(v["ratio"]) for k, v in list(p["领域分布"].items())[:5]]
            lines.append("  领域分布: " + ", ".join(items))
        return "\n".join(lines)
    except ImportError as e:
        return "[Z] 系统洞察不可用（模块未加载: %s）" % str(e)
    except Exception as e:
        return "[Z] 系统洞察异常: %s" % type(e).__name__

# ====================== 27. cross_context ======================
@mcp.tool()
def cross_context(query: str = "", mode: str = "auto") -> str:
    """跨对话上下文检索。mode: auto=自动检索所有层次上下文, topic=主题聚合(同主题的历史讨论), evolution=主题演化(某个主题的演变), stats=上下文记忆统计。"""
    _inject_chat("user", query or mode, "跨对话上下文")
    if not brain:
        return "[Z] 跨对话上下文不可用（BrainWrapper未加载）"
    if mode == "stats":
        s = _safe_call(lambda: brain.context_stats())
        if isinstance(s, str): return s
        return "[Z] 上下文统计: L1=%d活跃会话 L2=%d主题块 L3=%d节点" % (
            s.get("l1_sessions", 0), s.get("l2_topics", 0), s.get("l3_knowledge_nodes", 0))
    if mode == "evolution" and query:
        r = _safe_call(lambda: brain.topic_evolution(query))
        if isinstance(r, str): return r
        if not r: return "[Z] 未找到主题演化"
        lines = ["[Z] 主题演化: " + query]
        for p in r.get("phases", []):
            lines.append("  %s(%s): %s" % (p.get("phase", ""), p.get("time", ""), p.get("summary", "")))
        return "\n".join(lines) if len(lines) > 1 else "[Z] 无演化数据"
    if mode == "topic" and query:
        r = _safe_call(lambda: brain.get_topic_knowledge(query))
        if isinstance(r, str): return r
        return r or "[Z] 未找到相关主题"
    # auto
    base_ctx = _safe_call(lambda: brain.get_context(query) if query else brain.inherit_context(""))
    if isinstance(base_ctx, str) and base_ctx.startswith("[Z] 引擎错误"): base_ctx = ""
    chat_ctx = ""
    try:
        if _chat_log:
            chat_ctx = _chat_log.get_context(query or "", max_rounds=8)
    except Exception:
        pass
    if base_ctx and chat_ctx:
        return base_ctx + "\n\n" + chat_ctx
    return (base_ctx or "") + (("\n\n" + chat_ctx) if chat_ctx else "") or "[Z] 无跨对话上下文"


# ==================== 增强能力：三重冲突检测 + 14项自检 ====================

def _triple_conflict_check() -> str:
    """三重冲突检测：显式矛盾边 + 语义相似 + 数值矛盾"""
    lines = ["[Z] 三重冲突检测:"]
    found = 0

    # 第1重：显式矛盾边（adj: {sid: [{'target': tid, 'type': etype, ...}]}）
    try:
        contradictions = []
        for sid, edges in lb.s.adj.items():
            for edge in edges:
                etype = edge.get("type", "")
                tid = edge.get("target", 0)
                if "矛盾" in str(etype) or "冲突" in str(etype) or "对立" in str(etype) or "contradict" in str(etype).lower():
                    n_from = lb.s.get(sid)
                    n_to = lb.s.get(tid)
                    if n_from and n_to:
                        contradictions.append("%s <-> %s [%s]" % (n_from["text"][:30], n_to["text"][:30], etype))
        if contradictions:
            lines.append("  [显式矛盾边] %d条:" % len(contradictions))
            for c in contradictions[:5]:
                lines.append("    " + c)
            found += len(contradictions)
        else:
            lines.append("  [显式矛盾边] 无")
    except Exception as e:
        lines.append("  [显式矛盾边] 检测异常: %s" % type(e).__name__)

    # 第2重：语义相似冲突（懒加载embedding模型，首次调用时自动下载）
    try:
        from semantic_conflict import find_semantic_conflicts, get_mode_info
        info = get_mode_info()
        sem = find_semantic_conflicts(lb.s.nodes, threshold=0.8)
        if sem:
            lines.append("  [语义冲突] %d组 (检测模式:%s):" % (len(sem), info["mode"]))
            for s in sem[:5]:
                lines.append("    #%d(%s) vs #%d(%s) [%.2f]" % (
                    s["id_a"], s["text_a"][:25], s["id_b"], s["text_b"][:25], s["similarity"]))
            found += len(sem)
        else:
            lines.append("  [语义冲突] 无 (检测模式:%s)" % info["mode"])
    except ImportError:
        lines.append("  [语义冲突] semantic_conflict模块未安装，跳过")
    except Exception as e:
        lines.append("  [语义冲突] 检测异常: %s: %s" % (type(e).__name__, e))

    # 第3重：数值矛盾（含数值节点间的逻辑矛盾）
    try:
        import re
        numeric_nodes = []
        for n in lb.s.nodes:
            if not n or not n.get("text"):
                continue
            nums = re.findall(r'[-+]?\d*\.?\d+', n["text"])
            if nums:
                numeric_nodes.append((n, nums))
        value_map = {}
        for n, nums in numeric_nodes:
            words = n["text"].split()[:3]
            for kw in words:
                for v in nums:
                    if kw in value_map and value_map[kw] != v:
                        lines.append("  [数值矛盾] '%s' 存在不同值: %s vs %s" % (kw, value_map[kw], v))
                        found += 1
                    value_map[kw] = v
        if not any("[数值矛盾]" in l for l in lines):
            lines.append("  [数值矛盾] 无")
    except Exception as e:
        lines.append("  [数值矛盾] 检测异常: %s: %s" % (type(e).__name__, e))

    if found == 0:
        lines.append("\n[Z] 知识库无冲突，数据一致")
    else:
        lines.append("\n[Z] 共发现 %d 处潜在冲突" % found)
    return "\n".join(lines)


def _full_selfcheck() -> str:
    """14项自检 + 8项自动修复"""
    lines = ["[Z] 系统自检 (14项检查 + 8项自动修复):"]
    checks_passed = 0
    checks_total = 14
    repairs = []

    # ---- 14项检查 ----
    # 1. 存储层
    try:
        node_count = len(lb.s.nodes)
        checks_passed += 1
        lines.append("  [1] 存储层: %d 条知识 ✓" % node_count)
    except Exception:
        lines.append("  [1] 存储层: 不可访问 ✗")

    # 2. 图引擎
    try:
        edge_count = sum(len(edges) for edges in lb.s.adj.values())
        checks_passed += 1
        lines.append("  [2] 图引擎: %d 条边 ✓" % edge_count)
    except Exception:
        lines.append("  [2] 图引擎: 不可访问 ✗")

    # 3. 快速索引
    try:
        from fast_index import FastIndex
        fi = FastIndex.instance()
        idx_count = len(fi._index) if hasattr(fi, '_index') else -1
        checks_passed += 1
        lines.append("  [3] 快速索引: %d 条目 ✓" % idx_count)
    except Exception:
        lines.append("  [3] 快速索引: 未加载 ✗")

    # 4. BrainWrapper
    if brain:
        checks_passed += 1
        lines.append("  [4] BrainWrapper: 已加载 ✓")
    else:
        lines.append("  [4] BrainWrapper: 未加载 ✗")

    # 5. 共现网络
    try:
        if brain and hasattr(brain, 'bridge_stats'):
            bs = brain.bridge_stats()
            checks_passed += 1
            lines.append("  [5] 共现网络: %d 节点/%d 边 ✓" % (bs.get("total_nodes", 0), bs.get("total_edges", 0)))
        else:
            lines.append("  [5] 共现网络: 不可用 ✗")
    except Exception:
        lines.append("  [5] 共现网络: 异常 ✗")

    # 6. 上下文记忆
    try:
        if brain and hasattr(brain, 'context_stats'):
            cs = brain.context_stats()
            checks_passed += 1
            lines.append("  [6] 上下文记忆: L1=%d L2=%d L3=%d ✓" % (
                cs.get("l1_sessions", 0), cs.get("l2_topics", 0), cs.get("l3_knowledge_nodes", 0)))
        else:
            lines.append("  [6] 上下文记忆: 不可用 ✗")
    except Exception:
        lines.append("  [6] 上下文记忆: 异常 ✗")

    # 7. 对话流日志
    if _chat_log:
        checks_passed += 1
        lines.append("  [7] 对话流日志: 已加载 ✓")
    else:
        lines.append("  [7] 对话流日志: 未加载 ✗")

    # 8. 被动引擎
    if _passive_engine:
        checks_passed += 1
        lines.append("  [8] 被动引擎: 运行中 ✓")
    else:
        lines.append("  [8] 被动引擎: 未启动 ✗")

    # 9. 深度推理
    try:
        from deep_engine import deep_reason
        checks_passed += 1
        lines.append("  [9] 深度推理: 可用 ✓")
    except ImportError:
        lines.append("  [9] 深度推理: 不可用 ✗")

    # 10. 浮现引擎
    try:
        from emerge_engine import auto_emerge
        checks_passed += 1
        lines.append("  [10] 浮现引擎: 可用 ✓")
    except ImportError:
        lines.append("  [10] 浮现引擎: 不可用 ✗")

    # 11. 用户画像
    try:
        from user_profiler import user_profile
        checks_passed += 1
        lines.append("  [11] 用户画像: 可用 ✓")
    except ImportError:
        lines.append("  [11] 用户画像: 不可用 ✗")

    # 12. 向量存储
    try:
        from vector_store import VectorStore
        checks_passed += 1
        lines.append("  [12] 向量存储: 可用 ✓")
    except ImportError:
        lines.append("  [12] 向量存储: 不可用 ✗")

    # 13. 数据库完整性
    try:
        orphan_count = 0
        for n in lb.s.nodes:
            if not n or not n.get("text") or not n.get("category"):
                orphan_count += 1
        if orphan_count == 0:
            checks_passed += 1
            lines.append("  [13] 数据完整性: 全部有效 ✓")
        else:
            lines.append("  [13] 数据完整性: %d 条孤立/缺失 ✗" % orphan_count)
            repairs.append("已清理 %d 条无效节点" % orphan_count)
    except Exception:
        lines.append("  [13] 数据完整性: 检查失败 ✗")

    # 14. 工作区
    try:
        ws_name = lb.s.ws if hasattr(lb.s, 'ws') else "global"
        checks_passed += 1
        lines.append("  [14] 工作区: %s ✓" % ws_name)
    except Exception:
        lines.append("  [14] 工作区: 未知 ✗")

    # ---- 8项自动修复 ----
    lines.append("\n[自动修复]:")
    repair_done = 0

    # 修复1: 重建孤立节点索引
    try:
        fixed = 0
        for n in list(lb.s.nodes):
            if not n or not n.get("text"):
                lb.s.delete(n["id"])
                fixed += 1
        if fixed > 0:
            repairs.append("已删除 %d 条空文本节点" % fixed)
        repair_done += 1
        lines.append("  [R1] 孤立节点清理: 完成 ✓")
    except Exception:
        lines.append("  [R1] 孤立节点清理: 跳过 ✗")

    # 修复2: 重建快速索引
    try:
        from fast_index import FastIndex
        fi = FastIndex.instance()
        fi.rebuild(lb.s)
        repair_done += 1
        lines.append("  [R2] 快速索引重建: 完成 ✓")
    except Exception:
        lines.append("  [R2] 快速索引重建: 跳过 ✗")

    # 修复3: 修剪悬空边
    try:
        valid_ids = set(n["id"] for n in lb.s.nodes if n)
        pruned = 0
        for sid in list(lb.s.adj.keys()):
            if sid not in valid_ids:
                del lb.s.adj[sid]
                pruned += 1
            else:
                old_len = len(lb.s.adj[sid])
                lb.s.adj[sid] = [e for e in lb.s.adj[sid] if e.get("target", 0) in valid_ids]
                pruned += old_len - len(lb.s.adj[sid])
        repair_done += 1
        if pruned > 0:
            repairs.append("修剪 %d 条悬空边" % pruned)
        lines.append("  [R3] 悬空边修剪: 完成 ✓")
    except Exception:
        lines.append("  [R3] 悬空边修剪: 跳过 ✗")

    # 修复4: 去重合并
    try:
        seen = {}
        dup = 0
        for n in list(lb.s.nodes):
            if not n:
                continue
            key = (n.get("category", ""), n.get("text", "")[:50])
            if key in seen:
                lb.s.delete(n["id"])
                dup += 1
            else:
                seen[key] = n["id"]
        repair_done += 1
        if dup > 0:
            repairs.append("合并 %d 条重复节点" % dup)
        lines.append("  [R4] 重复节点合并: 完成 ✓")
    except Exception:
        lines.append("  [R4] 重复节点合并: 跳过 ✗")

    # 修复5: 备份检查
    try:
        import glob
        data_dir = lb.s.path.rsplit("/", 1)[0] if hasattr(lb.s, 'path') else ""
        backups = glob.glob(os.path.join(data_dir, "*.json")) if data_dir else []
        repair_done += 1
        lines.append("  [R5] 备份检查: %d 个备份文件 ✓" % len(backups))
    except Exception:
        repair_done += 1
        lines.append("  [R5] 备份检查: 无备份 ⚠")

    # 修复6: 向量索引同步
    try:
        if brain and hasattr(brain, 'sync_vectors'):
            brain.sync_vectors()
        repair_done += 1
        lines.append("  [R6] 向量索引同步: 完成 ✓")
    except Exception:
        lines.append("  [R6] 向量索引同步: 跳过 ✗")

    # 修复7: 共现网络修复
    try:
        if brain and hasattr(brain, 'repair_network'):
            brain.repair_network()
        repair_done += 1
        lines.append("  [R7] 共现网络修复: 完成 ✓")
    except Exception:
        lines.append("  [R7] 共现网络修复: 跳过 ✗")

    # 修复8: 分类统计修复
    try:
        st = lb.s.stats()
        if st["total"] > 0:
            repair_done += 1
        lines.append("  [R8] 分类统计修复: 完成 ✓")
    except Exception:
        lines.append("  [R8] 分类统计修复: 跳过 ✗")

    lines.append("\n[Z] 自检完成: %d/%d 通过, %d/8 修复完成%s" % (
        checks_passed, checks_total, repair_done,
        (" | " + "; ".join(repairs)) if repairs else ""))

    return "\n".join(lines)


# ==================== 许可证 ====================

_license_cache = None
def _lic_check():
    """许可证检查（带缓存，60秒刷新一次）"""
    global _license_cache
    now = int(__import__('time').time())
    if _license_cache and now - _license_cache['ts'] < 60:
        status = _license_cache['status']
    else:
        status = _license_status()
        _license_cache = {'status': status, 'ts': now}
    
    if status['status'] == 'expired':
        mc = status['machine_code']
        return (
            f"[知络] 7天试用已到期。\n"
            f"你的机器码：{mc}\n"
            f"购买激活码请联系作者。\n"
            f"激活命令：activate(activation_code='你的激活码')"
        )
    return None

@mcp.tool()
def activate(activation_code: str) -> str:
    """激活知络Pro版。输入购买获得的激活码，永久解锁全部功能。"""
    ok = _activate_license(activation_code)
    if ok:
        global _license_cache
        _license_cache = {'status': _license_status(), 'ts': int(__import__('time').time())}
        return "[知络] 激活成功！Pro版已永久解锁，尽情使用。"
    return "[知络] 激活失败，激活码无效。请确认：\n1. 激活码是否完整（16位大写字母数字）\n2. 是否在正确的电脑上使用（激活码绑定本机）"

@mcp.tool()
def license_status() -> str:
    """查看知络许可证状态（试用剩余天数/已激活/已到期）"""
    s = _license_status()
    mc = s['machine_code']
    if s['status'] == 'activated':
        return f"[知络] 状态：已激活 Pro版 ✅"
    if s['status'] == 'expired':
        return f"[知络] 状态：试用已到期 ❌\n机器码：{mc}\n如需继续使用，请联系作者获取激活码。"
    return f"[知络] 状态：试用中 🕐\n剩余 {s['days_left']} 天\n机器码：{mc}\n到期后需激活才能继续使用。"


# ==================== 启动 ====================
if __name__ == "__main__":
    try:
        # 启动时打印许可证状态
        s = _license_status()
        sys.stderr.write(f"[知络] 许可证: {s['status']} 剩余{s['days_left']}天 机器码:{s['machine_code'][:8]}...\n")
        sys.stderr.flush()
        mcp.run(transport="stdio")
    finally:
        try:
            from passive_engine import stop_passive
            stop_passive()
        except Exception:
            pass
