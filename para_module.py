"""
PARA ???????
P=??(Project), A=??(Area), R=??(Resource), X=??(Archive)
"""

def set_para(store, node_id_or_result, para):
    """???????PARA??"""
    import re
    if isinstance(node_id_or_result, str):
        m = re.search(r'#(\d+)', node_id_or_result)
        if m:
            nid = int(m.group(1))
        else:
            return
    else:
        nid = node_id_or_result
    node = store.get(nid)
    if node:
        node["para"] = para
        store.save()
        conn = store._get_conn()
        try:
            conn.execute("ALTER TABLE nodes ADD COLUMN para TEXT")
        except Exception:
            pass
        conn.execute("UPDATE nodes SET para=? WHERE id=?", (para, nid))
        conn.commit()


def filter_by_para(store, para, nodes=None):
    """?PARA????????"""
    if para and para.upper() in ("P", "A", "R", "X"):
        target = para.upper()
        if nodes is None:
            nodes = store.valid()
        return [n for n in nodes if n.get("para", "") == target]
    return nodes if nodes is not None else (store.valid() if hasattr(store, "valid") else [])


def stats_by_para(store):
    """??PARA????"""
    from collections import Counter
    nodes = store.valid() if hasattr(store, "valid") else []
    return dict(Counter(n.get("para", "") for n in nodes))
