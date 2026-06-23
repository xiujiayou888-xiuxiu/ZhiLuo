# -*- coding: utf-8 -*-
"""对话流记忆模块: 环形缓冲 + 自动注入上下文"""
import time
from collections import OrderedDict

class ChatLog:
    """对话流环形缓冲。记录最近N轮对话，自动注入到跨对话上下文中"""
    
    _instance = None
    _buffer = OrderedDict()
    _max_rounds = 50
    _data_dir = None
    _counter = 0
    
    @classmethod
    def init(cls, data_dir=None):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._buffer = OrderedDict()
            cls._data_dir = str(data_dir) if data_dir else None
        return cls._instance
    
    @classmethod
    def record(cls, role, content, keyword=""):
        """记录一轮对话。role: user/assistant/system"""
        inst = cls.init()
        cls._counter += 1
        round_id = cls._counter
        timestamp = time.time()
        inst._buffer[round_id] = {
            "role": role,
            "content": content[:500],
            "keyword": keyword[:50],
            "time": timestamp
        }
        # 超过上限则淘汰最旧的
        while len(inst._buffer) > cls._max_rounds:
            inst._buffer.popitem(last=False)
    
    @classmethod
    def get_context(cls, query="", max_rounds=10):
        """获取最近的对话流上下文，格式化为提示文本"""
        inst = cls.init()
        if not inst._buffer:
            return ""
        items = list(inst._buffer.values())
        recent = items[-max_rounds:]
        
        # 如果传了 query，做关键词过滤——只返回相关的
        if query:
            query_lower = query.lower()
            keywords = set(query_lower.split())
            filtered = []
            for item in recent:
                content_lower = item["content"].lower()
                if any(kw in content_lower for kw in keywords) or                    any(kw in item.get("keyword","").lower() for kw in keywords):
                    filtered.append(item)
            recent = filtered[-max_rounds:] if filtered else recent[-3:]
        
        lines = ["【对话流上下文】"]
        for item in recent:
            role_tag = "用户" if item["role"] == "user" else ("知络" if item["role"] == "assistant" else "系统")
            lines.append(f"  [{role_tag}] {item['content'][:200]}")
        return "\n".join(lines)
    
    @classmethod
    def clear(cls):
        cls._buffer.clear()
    
    @classmethod
    def stat(cls):
        return {"rounds": len(cls._buffer), "max": cls._max_rounds}

# 全局单例
chat_log = ChatLog.init()
