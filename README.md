# ZhiLuo — Give Your AI Agent a Persistent Brain

AI agents are smart, but they forget everything the moment a conversation ends. Every chat starts from zero. **ZhiLuo fixes that.**

A lightweight, local, MCP-native memory engine. No cloud, no API bills, no setup nightmares. Just persistent memory that works.

---

## What Makes It Different

Most agent memory solutions are either cloud-locked paywalls or half-baked wrappers around a vector DB. ZhiLuo does things differently:

| Feature | What It Means |
|---|---|
| **Triple Conflict Detection** | Catches contradictory facts — explicit edges + semantic similarity + numeric comparison. Your knowledge base doesn't silently rot. |
| **Deep Reasoning Engine** | Breaks questions down, retrieves from multiple angles, scores conclusions. Not just search — actual reasoning chains. |
| **Passive Monitoring** | Background health checks. Detects knowledge surges, profile drift, contradiction accumulation. Alerts before things break. |
| **6-Type Knowledge Graph** | related / causes / contradicts / hypernym / associates / precedes. PageRank surfaces what matters most. |
| **Cross-Conversation Memory** | Three-layer context (short/mid/long term). Agent remembers across sessions, across days. |
| **Self-Healing** | 14 health checks + 8 auto-fixes. Corrupted index? Fixed. Broken graph? Rebuilt. |
| **Confidence Decay** | 30-day half-life. Old unverified facts quietly fade, fresh knowledge rises. |
| **Pure Local** | SQLite + FTS5 + jieba. Zero API calls. Zero latency. Your data stays on your machine. |

---

## Quick Start

### Install

```bash
pip install jieba networkx
```

### 3 Lines to a Brain

```python
from zhiluo_loader import ZhiLuo

lb = ZhiLuo()

# Store knowledge
lb.run("记住: 牛肉批发价从35涨到了48，涨幅37%")

# Search
lb.run("牛肉现在什么价")

# Cross-conversation recall
brain = lb.brain
brain.get_context("供应商价格变动")
```

### Use as MCP Server

Add to your MCP config:

```json
{
  "mcpServers": {
    "zhiluo": {
      "command": "python",
      "args": ["mcp_server.py"],
      "cwd": "/path/to/zhiluo"
    }
  }
}
```

Then your AI agent gets 19 tools: `learn`, `query`, `search`, `analyze`, `deep_reason`, `selfcheck`, `visualize`, `summarize`, and more.

---

## Architecture

```
User Input
    ↓
v8.6 Incremental Layer (intent classification → genre detection → context binding)
    ↓
v7.1 Solid Base (jieba tokenization → SimHash dedup → FTS5 index → SQLite store)
    ↓
Knowledge Graph (NetworkX) + Passive Monitor (background health checks)
```

- **27 modules**, all in pure Python (no compiled binaries required)
- Three-tier index: Hash (O(1)) → Keyword → FTS5 full-text
- Progressive degradation: missing jieba? Falls back to character-split. No sqlite-vec? Falls back to TF-IDF.

---

## File Map

| File | Purpose |
|---|---|
| `engine.py` | Core: SimHash, MemoryStore, Intent classification |
| `brain_wrapper.py` | High-level wrapper, ties everything together |
| `mcp_server.py` | MCP protocol server, 19 tools |
| `deep_engine.py` | Deep reasoning: decompose → retrieve → analyze → score |
| `semantic_conflict.py` | Triple conflict detection |
| `passive_engine.py` | Background health monitoring |
| `graph_engine.py` | NetworkX knowledge graph, PageRank |
| `context_memory.py` | Three-layer cross-conversation memory |
| `genre_retrieval.py` | Genre-aware search (process/argument/definition/data/dialogue) |
| `tools.py` | Utilities: visualize, export, backup/restore |
| `zhiluo_loader.py` | Entry point |
| `emerge_engine.py` | Knowledge emergence: surface hidden patterns |
| `user_profiler.py` | User profile tracking and drift detection |
| `chat_log.py` | Conversation logging and replay |

---

## Comparison

| Feature | ZhiLuo | Mem0 | Zep | LangChain Memory |
|---|---|---|---|---|
| **100% Local** | ✅ | ❌ (cloud default) | ❌ (cloud default) | ✅ |
| **Zero Token Cost** | ✅ | ❌ | ❌ | ❌ |
| **Knowledge Graph** | ✅ 6-type + PageRank | ❌ | ✅ basic | ❌ |
| **Conflict Detection** | ✅ Triple | ❌ | ❌ | ❌ |
| **Deep Reasoning** | ✅ Built-in | ❌ | ❌ | ❌ |
| **Self-Healing** | ✅ 14 checks | ❌ | ❌ | ❌ |
| **MCP Native** | ✅ 19 tools | ❌ | ❌ | ❌ |
| **Setup** | `pip install` | Docker + API key | Docker + API key | pip + config |

---

## FAQ

**Is this production-ready?**
It's running in production on my own AI agent setup. It's stable, but the MCP ecosystem is young. Test with your own use case first.

**Can it handle 100K+ knowledge entries?**
SQLite handles it fine. The graph engine (NetworkX) starts to slow down around 50K nodes. Future versions will add Neo4j backend.

**Why not use mem0 / Zep / LangChain Memory?**
Those are great. ZhiLuo is for people who want something local, free, MCP-native, and don't want to configure a stack of services.

**Who made this?**
An independent developer who got annoyed that AI agents forget everything. Built with AI assistance — the product design, architecture decisions, and iteration direction are human; the code execution is AI-driven.

---

## Roadmap

- [ ] PyPI package (`pip install zhiluo`)
- [ ] MCP official directory listing
- [ ] Neo4j backend for 100K+ scale
- [ ] English NLP support (spaCy integration)
- [ ] Cloud sync option (optional, self-hosted)
- [ ] Pro tier: team collaboration, cloud backup, priority support

---

## Contributing

Issues and pull requests are welcome! If ZhiLuo saves you time, consider giving it a ⭐ — it helps others discover it.

## License

MIT License — use it, modify it, build on it. Just don't blame me if your agent becomes too smart.

## Contact

Questions or ideas? Reach out: [18599936462@coze.email](mailto:18599936462@coze.email)
