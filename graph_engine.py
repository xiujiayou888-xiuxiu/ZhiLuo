# -*- coding: utf-8 -*-
"""
知络图引擎 v7.1 — NetworkX 多跳推理 + 成本传播
全行业通用：任何实体-关系-实体都能建图、遍历、推理
"""
import networkx as nx
from collections import defaultdict
import math

class GraphEngine:
    """全行业通用的图推理引擎"""
    
    def __init__(self):
        self.G = nx.DiGraph()
        # 实体名 -> node_id 映射
        self.entity_map = {}
        self._next_id = 0
    
    def _get_or_create(self, name, etype="entity"):
        """获取或创建实体节点"""
        if name in self.entity_map:
            return self.entity_map[name]
        nid = self._next_id
        self._next_id += 1
        self.G.add_node(nid, name=name, type=etype)
        self.entity_map[name] = nid
        return nid
    
    def add_entity(self, name, etype="entity", meta=None, domain="通用", entity_type="entity", confidence=0.5):
        """添加实体（含四维标签）"""
        _EVENT_MARKERS = {"价格波动","成本上升","成本下降","价格下降"}
        if name in _EVENT_MARKERS:
            return None
        import time as _time
        nid = self._get_or_create(name, etype)
        # 四维标签
        self.G.nodes[nid]["domain"] = domain
        self.G.nodes[nid]["entity_type"] = entity_type
        self.G.nodes[nid]["confidence"] = max(confidence, self.G.nodes[nid].get("confidence", 0))
        self.G.nodes[nid]["timestamp"] = _time.time()
        if meta:
            for k, v in meta.items():
                self.G.nodes[nid][k] = v
        return nid
    
    def add_relation(self, source, target, rtype="related", weight=1.0, meta=None,
                     source_domain="通用", source_type="entity", source_confidence=0.5,
                     target_domain="通用", target_type="entity", target_confidence=0.5):
        """添加关系（自动创建实体，含四维标签）"""
        _EVENT_MARKERS = {"价格波动","成本上升","成本下降","价格下降"}
        if source in _EVENT_MARKERS or target in _EVENT_MARKERS:
            return None, None
        import time as _time
        sid = self._get_or_create(source)
        tid = self._get_or_create(target)
        # 四维标签（只首次写入，不覆盖已有）
        for nid, d, dt, dc in [(sid, source_domain, source_type, source_confidence),
                               (tid, target_domain, target_type, target_confidence)]:
            if "domain" not in self.G.nodes[nid]:
                self.G.nodes[nid]["domain"] = d
            if "entity_type" not in self.G.nodes[nid]:
                self.G.nodes[nid]["entity_type"] = dt
            if "confidence" not in self.G.nodes[nid]:
                self.G.nodes[nid]["confidence"] = dc
            if "timestamp" not in self.G.nodes[nid]:
                self.G.nodes[nid]["timestamp"] = _time.time()
        self.G.add_edge(sid, tid, relation=rtype, weight=weight, timestamp=_time.time())
        if meta:
            for k, v in meta.items():
                self.G[sid][tid][k] = v
        return sid, tid
    
    def find_entity(self, name):
        """查找实体：长词优先精确匹配，短词不误配长词"""
        results = []
        for nid, data in self.G.nodes(data=True):
            n = data.get("name", "")
            if n == name:
                results.insert(0, (nid, data))
            elif n.startswith(name + "的") or name.startswith(n + "的"):
                if n != name:
                    results.append((nid, data))
        results.sort(key=lambda x: -len(x[1].get("name", "")))
        return results
    
    def diffuse(self, name, max_hops=3):
        """图扩散：从实体出发，沿边遍历所有可达节点"""
        matches = self.find_entity(name)
        if not matches:
            return []
        start = matches[0][0]
        # BFS 遍历
        visited = {}
        queue = [(start, 0)]
        while queue:
            node, hops = queue.pop(0)
            if node in visited or hops > max_hops:
                continue
            visited[node] = hops
            for neighbor in self.G.successors(node):
                if neighbor not in visited:
                    queue.append((neighbor, hops + 1))
            for neighbor in self.G.predecessors(node):
                if neighbor not in visited:
                    queue.append((neighbor, hops + 1))


        # 返回结果
        results = []
        for nid, hops in sorted(visited.items(), key=lambda x: x[1]):
            data = self.G.nodes[nid]
            results.append({
                "id": nid,
                "name": data.get("name", ""),
                "type": data.get("type", ""),
                "hops": hops,
            })
        return results
    
    def find_paths(self, source_name, target_name, max_depth=5):
        """找两个实体之间的所有路径"""
        s_matches = self.find_entity(source_name)
        t_matches = self.find_entity(target_name)
        if not s_matches or not t_matches:
            return []
        s = s_matches[0][0]
        t = t_matches[0][0]
        try:
            # 双向搜索：正向 + 反向（忽略边方向）
            paths = list(nx.all_simple_paths(self.G, s, t, cutoff=max_depth))
            paths += list(nx.all_simple_paths(self.G, t, s, cutoff=max_depth))
        except nx.NetworkXError:
            paths = []
        # 格式化路径
        # 去重（双向搜索可能出重复路径）
        seen_paths = set()
        result = []
        for path in paths:
            chain = []
            for i, nid in enumerate(path):
                data = self.G.nodes[nid]
                chain.append({"name": data.get("name", ""), "type": data.get("type", "")})
                if i < len(path) - 1:
                    edge = self.G[path[i]][path[i+1]]
                    chain.append({"rel": edge.get("relation", "related")})
            pk = "->".join(str(nid) for nid in path)
            if pk in seen_paths:
                continue
            seen_paths.add(pk)
            result.append(chain)
        return result
    
    def cost_propagation(self, entity_name, change_pct, max_hops=5, min_change=0.01, damping_config=None):
        """
        成本传播算法 v7.1（增强版）
        输入：某个实体价格变化百分比，沿关系边传播影响
        新增：
          - damping_config：按关系类型自定义衰减系数（可配置）
          - 双向传播（successors + predecessors）
          - 差异化衰减（part_of/affect 高传播，related 低传播）
          - 截断阈值可配（min_change）
        """
        _DEFAULT_DAMPING = {
            "part_of": 1.0,
            "affect": 1.0,
            "cause": 1.0,
            "substitute": 0.8,
            "synonym": 1.0,
            "related": 0.5,
            "default": 0.3,
        }
        damp = damping_config if damping_config else _DEFAULT_DAMPING

        matches = self.find_entity(entity_name)
        if not matches:
            return {"error": "未找到实体: " + entity_name}
        start = matches[0][0]

        impacts = {start: {"name": entity_name, "change": change_pct, "hops": 0, "path": [entity_name]}}
        queue = [(start, change_pct, 0, [entity_name])]

        while queue:
            node, current_change, hops, path = queue.pop(0)
            if hops >= max_hops:
                continue

            for neighbor in self.G.successors(node):
                edge = self.G[node][neighbor]
                rel = edge.get("relation", "related")
                weight = edge.get("weight", 1.0)
                damping_factor = damp.get(rel, damp["default"])
                propagated = current_change * weight * damping_factor

                n_data = self.G.nodes[neighbor]
                n_name = n_data.get("name", "")
                if abs(propagated) < min_change:
                    continue
                if neighbor in impacts:
                    if abs(propagated) > abs(impacts[neighbor]["change"]):
                        impacts[neighbor] = {"name": n_name, "change": round(propagated, 4),
                                             "hops": hops + 1, "path": path + [n_name]}
                else:
                    impacts[neighbor] = {"name": n_name, "change": round(propagated, 4),
                                         "hops": hops + 1, "path": path + [n_name]}
                queue.append((neighbor, propagated, hops + 1, path + [n_name]))

            for neighbor in self.G.predecessors(node):
                try:
                    edge = self.G[neighbor][node]
                except KeyError:
                    continue
                rel = edge.get("relation", "related")
                weight = edge.get("weight", 1.0)
                damping_factor = damp.get(rel, damp["default"])
                propagated = current_change * weight * damping_factor
                if abs(propagated) < min_change:
                    continue
                n_data = self.G.nodes[neighbor]
                n_name = n_data.get("name", "")
                if neighbor in impacts:
                    if abs(propagated) > abs(impacts[neighbor]["change"]):
                        impacts[neighbor] = {"name": n_name, "change": round(propagated,4),
                                             "hops": hops+1, "path": path + [n_name]}
                else:
                    impacts[neighbor] = {"name": n_name, "change": round(propagated,4),
                                         "hops": hops+1, "path": path + [n_name]}
                queue.append((neighbor, propagated, hops + 1, path + [n_name]))

        sorted_impacts = sorted(impacts.values(), key=lambda x: -abs(x["change"]))
        for imp in sorted_impacts:
            nid = None
            for nid_candidate, data in self.G.nodes(data=True):
                if data.get("name") == imp["name"]:
                    nid = nid_candidate
                    break
            if nid is not None:
                conf = self.G.nodes[nid].get("confidence", 0.5)
                if conf < 0.6:
                    imp["warning"] = "待验证"
        return {
            "source": entity_name,
            "initial_change": change_pct,
            "affected_count": len(sorted_impacts) - 1,
            "impacts": sorted_impacts,
        }

    def substitute_chain(self, entity_name, max_hops=5):
        """替代链推理（3.1）：传递性 + 循环检测 + 适配度评分"""
        matches = self.find_entity(entity_name)
        if not matches:
            return {"error": "未找到实体: " + entity_name}
        start = matches[0][0]

        # BFS 沿 substitute 边扩散
        chain = []
        seen = {}  # node -> best_score
        queue = [(start, 0, [entity_name])]

        while queue:
            node, hops, path = queue.pop(0)
            if hops >= max_hops:
                continue

            for neighbor in list(self.G.successors(node)) + list(self.G.predecessors(node)):
                # 检查是否有 substitute 边
                edge_data = None
                if neighbor in self.G[node]:
                    if self.G[node][neighbor].get("relation") == "substitute":
                        edge_data = self.G[node][neighbor]
                if edge_data is None and node in self.G[neighbor]:
                    if self.G[neighbor][node].get("relation") == "substitute":
                        edge_data = self.G[neighbor][node]
                if edge_data is None:
                    continue

                n_name = self.G.nodes[neighbor].get("name", "")
                # 跳过源实体本身
                if n_name == entity_name:
                    continue

                conf = self.G.nodes[neighbor].get("confidence", 0.5)
                weight = edge_data.get("weight", 0.6)
                score = round(conf * weight, 3)

                # 只保留最高分
                if neighbor in seen and seen[neighbor] >= score:
                    continue
                seen[neighbor] = score

                new_path = path + [n_name]
                # 检查是否已经加入 chain（同名不重复加入）
                already_in = any(c["name"] == n_name for c in chain)
                if not already_in:
                    chain.append({"name": n_name, "hops": hops + 1,
                                  "path": new_path, "score": score})
                queue.append((neighbor, hops + 1, new_path))

        chain.sort(key=lambda x: -x["score"])
        return {"source": entity_name, "alternatives": chain, "total": len(chain)}


    def reason(self, query, context=None):
        """通用推理入口"""
        # 解析查询意图
        query_lower = query.lower()

        # 意图1：替代链（"X可以替代什么" / "X的替代方案" / "什么可以替代X"）
        if "替代" in query or "代替" in query or "替换" in query:
            import re
            # "什么可以替代X" -> 提取X
            m1 = re.search(r"[可以能]替代\s*(.+)", query)
            m2 = re.search(r"(.+?)[的]?替代[方案]", query)
            m3 = re.search(r"(.+?)[可以能]替代", query)
            entity = None
            if m1:
                entity = m1.group(1).strip()
            elif m2:
                entity = m2.group(1).strip()
            elif m3:
                entity = m3.group(1).strip()
            if entity:
                result = self.substitute_chain(entity)
                if "error" in result:
                    return result
                return result
            return {"error": "无法识别替代查询目标"}
        
        # 意图2：成本传播（"X涨Y%会影响什么"）
        import re
        cost_match = re.search(r'(.+?)[涨价涨跌降]+(\d+\.?\d*)\s*%?', query)
        if cost_match and ("影响" in query or "传播" in query or "会怎样" in query or "怎么办" in query):
            entity = cost_match.group(1).strip()
            pct = float(cost_match.group(2)) / 100.0
            # 判断涨跌
            if "降" in query or "跌" in query:
                pct = -pct
            return self.cost_propagation(entity, pct)
        
        # 意图2：路径查找（"X和Y什么关系"）
        if "关系" in query or "路径" in query or "怎么连" in query:
            parts = re.split(r'[和与跟]', query)
            parts = [p.strip() for p in parts if p.strip()]
            if len(parts) >= 2:
                # 去掉"什么关系"等词
                p1 = re.sub(r'[什么关系路径怎么连].*', '', parts[0]).strip()
                p2 = re.sub(r'[什么关系路径怎么连].*', '', parts[1]).strip()
                paths = self.find_paths(p1, p2)
                return {"source": p1, "target": p2, "paths": paths}
        
        # 意图3：扩散查询（"X相关的东西"）
        if "相关" in query or "关联" in query or "扩散" in query:
            entity = re.sub(r'[相关关联扩散的东西有什么]', '', query).strip()
            if entity:
                return {"entity": entity, "diffuse": self.diffuse(entity)}
        
        return {"error": "无法理解查询意图，支持：成本传播/路径查找/关联扩散"}
    
    def stats(self):
        return {
            "nodes": self.G.number_of_nodes(),
            "edges": self.G.number_of_edges(),
        }
    
    def to_dict(self):
        """序列化为字典（持久化用）"""
        nodes = []
        for nid, data in self.G.nodes(data=True):
            nodes.append({
                "id": nid, "name": data.get("name", ""), "type": data.get("type", ""),
                "domain": data.get("domain", "通用"),
                "entity_type": data.get("entity_type", "entity"),
                "confidence": data.get("confidence", 1.0),
                "timestamp": data.get("timestamp", 0),
            })
        edges = []
        for s, t, data in self.G.edges(data=True):
            edges.append({
                "source": self.G.nodes[s].get("name", ""),
                "target": self.G.nodes[t].get("name", ""),
                "relation": data.get("relation", "related"),
                "weight": data.get("weight", 1.0),
            })
        return {"nodes": nodes, "edges": edges, "entity_map": self.entity_map, "next_id": self._next_id}
    
    def from_dict(self, data):
        """从字典恢复"""
        self.G = nx.DiGraph()
        self.entity_map = {}
        self._next_id = data.get("next_id", 0)
        for node in data.get("nodes", []):
            attrs = {"name": node.get("name",""), "type": node.get("type","entity")}
            for k in ("domain", "entity_type", "confidence", "timestamp"):
                if k in node:
                    attrs[k] = node[k]
            self.G.add_node(node["id"], **attrs)
            self.entity_map[node["name"]] = node["id"]
        for edge in data.get("edges", []):
            sid = self._get_or_create(edge["source"])
            tid = self._get_or_create(edge["target"])
            self.G.add_edge(sid, tid, relation=edge["relation"], weight=edge["weight"])
