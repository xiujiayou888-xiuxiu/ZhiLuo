# 知络基础版

单Agent本地知识库 — 帮你沉淀资料、复盘、踩坑和可复用经验。

**定位**：完整版（46工具 / 15元月）的精简版，保留23个核心工具，6.8元/月。

---

## 快速安装

### 1. 环境要求
- Python 3.10+
- Windows / macOS / Linux

### 2. 一键安装
```bat
install.bat
```

或手动：
```bash
python -m venv venv
venv\Scripts\python -m pip install --upgrade pip
venv\Scripts\pip install -r requirements.txt
```

### 3. MCP配置

在你的Agent配置文件中添加（Codex/WorkBuddy/ZCode等）：

```json
{
  "mcpServers": {
    "zhiluo-basic": {
      "command": "D:/skill项目/知络-基础版/venv/Scripts/python.exe",
      "args": ["D:/skill项目/知络-基础版/mcp_server.py"],
      "env": {}
    }
  }
}
```

### 4. 首次使用

对AI说：`setup(action="quick")`

或指定路径：`setup(action="quick", kb_path="D:/my-knowledge")`

---

## 22个MCP工具

### 基础知识循环（8个）
| 工具 | 功能 |
|------|------|
| `pre_answer_context` | 回答前检索相关记忆 |
| `learn` | 写入知识（SimHash/MinHash去重合并） |
| `note` | 快速记录（轻量learn） |
| `query` | 关键词搜索（FTS5+SimHash多策略） |
| `search` | 图扩散关联搜索 |
| `stats` | 知识库统计概览 |
| `selfcheck` | 14项系统自检+修复 |
| `setup` | 首次配置向导 |

### 分析推理（5个）
| 工具 | 功能 |
|------|------|
| `qa` | 本地知识库问答（TF-IDF，不依赖外部LLM） |
| `reason` | 推理分析（检索+关联扩散+聚合） |
| `summarize` | 长文本摘要 |
| `analyze` | 冲突检测 + 信任衰减分析 |
| `visualize` | Mermaid知识图谱可视化 |

### 实用工具（6个）
| 工具 | 功能 |
|------|------|
| `pitfall` | 踩坑记录（list/report/search） |
| `export` | 导出JSON/Markdown |
| `manage` | 编辑/更新/删除知识 |
| `history` | 最近写入历史 |
| `pending` | 待审核知识列表 |
| `confirm` | 确认待审核知识 |

### 系统管理（4个）
| 工具 | 功能 |
|------|------|
| `configure` | 查看/修改配置项 |
| `maintain` | 知识库维护（分类+补边+修复） |
| `workspace` | 多工作区切换 |
| `obsidian_sync` | **Obsidian 双向同步**（导出/导入） |

---

## 与完整版的区别

| 特性 | 基础版 (6.8元/月) | 完整版 (15元/月) |
|------|-------------------|-------------------|
| 知识读写 | ✅ | ✅ |
| 本地问答 | ✅ | ✅ |
| 踩坑记录 | ✅ | ✅ |
| 知识维护 | ✅ 基础 | ✅ 高级（日报/合成/自动学习） |
| 多Agent共享 | ❌ | ✅ |
| 后台自动扫描 | ❌ | ✅ 每日9:00/23:00 |
| Obsidian同步 | ✅ | ✅ |
| AI雷达/RSS | ❌ | ✅ |
| 争议裁决 | ❌ | ✅ |
| 深度推理 | ❌ | ✅ 认知匕首+深度推理 |
| 自动合成 | ❌ | ✅ |
| 报告生成 | ❌ | ✅ 周报/洞察 |
| PPT导出 | ❌ | ✅ |
| 向量语义搜索 | ❌ | ✅ |
| 工具数量 | 23个 | 46个 |

---

## 可选：配置LLM

在项目根目录创建 `.env` 文件：

```env
ZHILUO_LLM_API_KEY=sk-your-api-key
ZHILUO_LLM_API_URL=https://api.deepseek.com/v1/chat/completions
ZHILUO_LLM_MODEL=deepseek-chat
```

配置后，`summarize` 和关系提取功能将获得更好的效果。
