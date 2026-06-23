# -*- coding: utf-8 -*-
"""
知络行业知识自动采集器 v7.1
方案：从知络已有词典出发，用 LLM 生成行业实体关系
（替代爬百度百科——国内爬虫限制太多）
"""
import sys, os, json, re
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from relation_extractor import (
    extract_filtered, extract_relations, extract_entities,
    add_industry_entities, INDUSTRY_ENTITIES, reset_freq,
)

# 行业模板：每个行业的实体-关系生成提示词
INDUSTRY_PROMPTS = {
    "餐饮": """你是一个餐饮行业知识专家。请根据以下餐饮实体列表，生成它们之间已知的关系。
每个关系用一行表示：实体A -> 关系类型 -> 实体B
关系类型: part_of(组成/需要)、affect(影响)、cause(因果)、related(相关)、synonym(替代)

实体列表：{entities}

要求：
1. 只生成真实存在的关系，不要编造
2. 每行一个关系
3. 数量：15-30条
""",
    "制造": """你是一个制造业知识专家。请根据以下制造业实体列表，生成它们之间已知的关系。
每个关系用一行表示：实体A -> 关系类型 -> 实体B
关系类型: part_of(组成/需要)、affect(影响)、cause(因果)、related(相关)、synonym(替代)

实体列表：{entities}

要求同上。""",
    "建筑": """你是一个建筑业知识专家。请根据以下建筑实体列表，生成它们之间已知的关系。""",
    "零售": """你是一个零售业知识专家。请根据以下零售实体列表，生成它们之间已知的关系。""",
}

# 知识挖掘提示词（不需要 LLM，直接从百科风格模板生成）
KNOWLEDGE_TEMPLATES = {
    "餐饮": [
        # 食材 -> 菜品（part_of）
        "{ingredient}是{ingredient}的原料之一",
        "{dish}需要{ingredient}和{ingredient}",
        "{ingredient}涨价影响{ingredient}",
        # 菜品分类
        "{dish}是{dish}的一种",
        # 食材替换
        "{ingredient}可以替代{ingredient}",
        # 供应链
        "供应商提供{ingredient}",
        "供应商提供{ingredient}",
        "{ingredient}涨价{ingredient}成本上升",
    ],
}



def generate_knowledge_from_templates(industry, verbose=False):
    """从行业模板生成规则化知识（零API依赖，全行业通用）"""
    entities = INDUSTRY_ENTITIES.get(industry, [])
    if not entities:
        return []

    # 按类型自动分类：产品/菜品 vs 原材料/零件
    product_keywords = {
        "餐饮": {"面","蛋","丝","鸡","豆腐","菜","食"},
        "制造": {"汽车","手机","电脑","零件","组件","产品","成品"},
        "建筑": {"工程","地基","墙体","楼板","屋顶","装修","施工","竣工"},
        "零售": {"SKU","商品","品类","仓库","物流"},
    }
    pk = product_keywords.get(industry, set())
    
    products = [e for e in entities if any(kw in e for kw in pk)] if pk else []
    materials = [e for e in entities if e not in products]
    
    if not products:
        # 如果无法自动分类，做最大的几个为产品
        products = sorted(entities, key=len, reverse=True)[:3]
        materials = [e for e in entities if e not in products]
    
    texts = []

    # 策略1：产品需要材料
    for prod in products[:5]:
        for mat in materials[:5]:
            if mat != prod:
                texts.append(f"{prod}需要{mat}")

    # 策略2：材料涨价影响产品
    for mat in materials[:5]:
        for prod in products[:5]:
            texts.append(f"{mat}涨价影响{prod}")

    # 策略3：供应商提供材料
    if "供应商" in entities:
        for mat in materials[:5]:
            texts.append(f"供应商提供{mat}")

    # 策略4：材料替代
    for i, mat1 in enumerate(materials[:5]):
        for mat2 in materials[i+1:5]:
            texts.append(f"{mat1}可以替代{mat2}")

    if verbose:
        print(f"  模板生成: {len(texts)} 条文本")

    # 用降噪管道提取关系
    all_relations = []
    reset_freq()
    for t in texts:
        result = extract_filtered(t, source_type="词典")
        all_relations.extend(result["relations"])

    # 去重
    seen = set()
    unique = []
    for r in all_relations:
        key = (r["source"], r["target"], r["relation"])
        if key not in seen:
            seen.add(key)
            unique.append(r)

    return unique

def generate_knowledge_from_seeds(seeds, industry, llm_func=None):
    """从种子实体出发，用 LLM 生成关系"""
    if llm_func is None:
        # 无 LLM 时回退到模板
        return generate_knowledge_from_templates(industry, verbose=True)
    
    prompt = f"""从以下文本中提取实体和关系，返回JSON格式：
{{"entities": [{{"name":"实体名","type":"类型"}}], "relations": [{{"source":"实体1","target":"实体2","relation":"part_of|affect|cause|related|synonym","weight":0.0-1.0}}]}}

实体列表：{json.dumps(seeds, ensure_ascii=False)}
行业：{industry}

请根据你对{industry}行业的了解，生成这些实体之间真实存在的业务关系。"""
    
    try:
        result = llm_func(prompt)
        if isinstance(result, str):
            s, e = result.find("{"), result.rfind("}") + 1
            if s >= 0 and e > s:
                data = json.loads(result[s:e])
                return data.get("relations", [])
        return result if isinstance(result, list) else []
    except Exception:
        return generate_knowledge_from_templates(industry, verbose=True)


def crawl_industry(industry, llm_func=None, verbose=True):
    """采集一个行业的实体关系知识"""
    entities = INDUSTRY_ENTITIES.get(industry, [])
    if not entities:
        if verbose: print(f"[CRAWL] 行业 {industry} 无词典")
        return {"industry": industry, "relations": [], "entities_count": 0}
    
    if verbose: print(f"[CRAWL] {industry}: {len(entities)} 个种子实体")
    
    if llm_func:
        relations = generate_knowledge_from_seeds(entities, industry, llm_func)
    else:
        relations = generate_knowledge_from_templates(industry, verbose=verbose)
    
    if verbose:
        print(f"[CRAWL] 生成 {len(relations)} 条关系")
        if relations:
            for r in relations[:5]:
                print(f"  {r['source']} --{r['relation']}--> {r['target']}")
    
    return {
        "industry": industry,
        "relations": relations,
        "entities_count": len(entities),
    }


def import_to_graph(ge, result):
    """灌入图引擎"""
    added = 0
    for r in result["relations"]:
        conf = r.get("confidence", r.get("weight", 0.5))
        ge.add_relation(r["source"], r["target"],
                        rtype=r["relation"], weight=conf)
        added += 1
    return added


if __name__ == "__main__":
    import sys
    industry = sys.argv[1] if len(sys.argv) > 1 else "餐饮"
    
    print(f"=== 知络自动采集器 v7.1 ===\n行业: {industry}\n")
    
    result = crawl_industry(industry, verbose=True)
    
    from graph_engine import GraphEngine
    ge = GraphEngine()
    n = import_to_graph(ge, result)
    print(f"\n灌入图引擎: {n}条边, {ge.stats()['nodes']}节点")
    
    # 测试成本传播
    if industry == "餐饮" and result["relations"]:
        r = ge.cost_propagation("牛肉", 0.20)
        if "error" not in r:
            print(f"\n牛肉涨20%影响 {r['affected_count']} 个实体:")
            for imp in r["impacts"][:5]:
                print(f"  {imp['name']}: {imp['change']*100:+.1f}%")
