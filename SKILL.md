# 知络·超级版 v8.5.4

> **Agent 记忆外脑 — 每个对话的必经管路**

知络是安装在智能体侧的记忆引擎，负责**记住、检索、推理、分析**所有对话中产生的知识。
**每个对话开始时、每次智能体回答前，都应该先经过知络**。

---

## 核心工作流

### 1. 对话开始 → 必经管路

每次启动新对话，智能体应自动调用：

```
cross_context("", mode="auto")          → 加载跨对话上下文
system_insight(mode="profile")           → 加载用户画像
stats(mode="overview")                   → 加载知识库概览
```

### 2. 每个问题 → 先查后答

收到用户提问时，先查知络再回答：

```
query(keyword, mode="auto")   → 搜索已有知识
search(keyword, mode="graph") → 关联搜索
cross_context(query)          → 跨对话上下文
```

### 3. 每个回答 → 自动学习

每次生成回答后，自动调用：

```
learn(text)               → 记住关键知识
learn(text, auto=True)    → 自动提取知识点到待确认队列
```

---

## 19 个工具

### 核心记忆（高频使用）

| 工具 | 功能 | 必会 |
|------|------|------|
| learn(text, auto=False) | 记住一条新知识 | ✅ |
| query(keyword, mode="auto") | 搜索已有知识。mode: auto/graph/context/genre | ✅ |
| search(keyword, mode="graph") | 图搜索。mode: graph/trace/entangle | ✅ |
| analyze(text="") | 三重冲突检测+数据分析 | ✅ |
| visualize(mode, keyword) | 可视化。mode: graph/brain/mermaid/pagerank | ✅ |
| selfcheck() | 系统自检：14项检查+8项自动修复 | ✅ |

### 辅助工具（按需使用）

| 工具 | 功能 |
|------|------|
| summarize(text) | 提取式总结文本核心要点 |
| pending(action, pid) | 待确认管理 |
| export(fmt, keyword) | 导出知识 |
| manage(action, text, new_text) | 综合管理 |
| workspace(action, name) | 工作区管理 |

### 增强能力（进阶使用）

| 工具 | 功能 |
|------|------|
| reason(query, mode) | 图推理引擎：auto/chain/entangle/rank |
| configure(key, value) | 动态配置：rules/llm/genre |
| extract(text) | 通用实体+关系提取 |
| stats(mode) | 综合统计：overview/graph/bridge/context/health/snapshot |
| history(mode, hours, node_id) | 变更历史 |
| deep_reason(question, mode) | 深度推理：auto/quantum/neural |
| system_insight(mode) | 系统洞察：profile/emerge/pitfalls |
| cross_context(query, mode) | 跨对话上下文：auto/topic/evolution/stats |

---

## MCP 配置

```json
{
  "mcpServers": {
    "知络": {
      "command": "python",
      "args": ["路径/mcp_server.py"],
      "env": {}
    }
  }
}
```
