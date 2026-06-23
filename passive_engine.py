# -*- coding: utf-8 -*-
"""被动响应引擎 v1.0
后台巡检 + 通知 chat_log / stderr
知识爆发 / 画像漂移 / 冲突积累 三类检测
"""
import time
import threading
import sys
from collections import deque

class PassiveEngine:
    """被动响应引擎

    三类检测:
      1. fresh_emergence - 知识爆发检测(新增>20% / 高频共现对)
      2. profile_shift - 用户画像漂移(新领域出现 / 风格变化)
      3. conflict_buildup - 冲突积累检测(持续矛盾未解决)
    """

    def __init__(self, lb, chat_log, interval=45):
        self.lb = lb
        self.chat_log = chat_log
        self.interval = interval
        self.running = False
        self._thread = None
        self._last_findings = set()
        self._baseline_nodes = 0
        self._baseline_profile = {}
        self._baseline_taken = False

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self.running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self.running = False

    def _loop(self):
        time.sleep(5)
        while self.running:
            try:
                findings = self._check()
                for f_text in findings:
                    if f_text not in self._last_findings:
                        self._notify(f_text)
                        self._last_findings.add(f_text)
                if len(self._last_findings) > 10:
                    self._last_findings = set(list(self._last_findings)[-10:])
            except Exception:
                pass
            time.sleep(self.interval)

    def _check(self):
        findings = []
        try:
            r = self._check_emergence()
            if r: findings.append(r)
        except Exception:
            pass
        try:
            r = self._check_profile_shift()
            if r: findings.append(r)
        except Exception:
            pass
        try:
            r = self._check_conflicts()
            if r: findings.append(r)
        except Exception:
            pass
        return findings

    def _take_baseline(self):
        if self._baseline_taken:
            return
        try:
            s = self.lb.s
            ws = self.lb.ws
            nodes = s.valid(ws) if hasattr(s, "valid") else []
            self._baseline_nodes = len(nodes)
            from user_profiler import user_profile
            self._baseline_profile = user_profile(s, ws).get("profile", {})
            self._baseline_taken = True
        except Exception:
            pass

    def _check_emergence(self):
        self._take_baseline()
        try:
            s = self.lb.s
            ws = self.lb.ws
            nodes = s.valid(ws) if hasattr(s, "valid") else []
            current = len(nodes)
            if current >= 5 and self._baseline_nodes > 0:
                ratio = current / max(self._baseline_nodes, 1)
                if ratio >= 1.2:
                    self._baseline_nodes = current
                    return f"[知识爆发] 新增{current - self._baseline_nodes}个节点(增长{int((ratio-1)*100)}%)，知识库扩展中"
        except Exception:
            pass
        try:
            if hasattr(self.lb, "brain") and self.lb.brain:
                co = self.lb.brain.bridge_stats()
                if co and co.get("total_edges", 0) > 5:
                    pairs = co.get("densest_pairs", [])
                    if len(pairs) >= 2:
                        top = pairs[0]
                        if isinstance(top, (list, tuple)) and len(top) >= 2:
                            return '[浮现关联] 高频共现: ' + str(top[0]) + ' 与 ' + str(top[1]) + ' 频繁出现，可能有隐含关联'
        except Exception:
            pass
        return None

    def _check_profile_shift(self):
        try:
            s = self.lb.s
            ws = self.lb.ws
            from user_profiler import user_profile
            current = user_profile(s, ws).get("profile", {})
            if not self._baseline_profile or not current:
                self._baseline_profile = current
                return None
            old_domains = self._baseline_profile.get("领域分布", {})
            new_domains = current.get("领域分布", {})
            if old_domains and new_domains:
                diff = set(new_domains.keys()) - set(old_domains.keys())
                if diff:
                    self._baseline_profile = current
                    return f"[画像漂移] 发现新领域: {', '.join(list(diff)[:3])}"
            old_style = self._baseline_profile.get("学习风格", "")
            new_style = current.get("学习风格", "")
            if old_style and new_style and old_style != new_style:
                self._baseline_profile = current
                return f"[画像漂移] 学习风格从{old_style}变为{new_style}"
        except Exception:
            pass
        return None

    def _check_conflicts(self):
        try:
            result = self.lb.run("检测冲突")
            if result and "冲突" in result and "无冲突" not in result:
                conflict_lines = [l for l in result.split("\n") if "冲突" in l]
                if conflict_lines:
                    return "[冲突积累] " + conflict_lines[0][:100]
        except Exception:
            pass
        return None

    def _notify(self, text):
        try:
            self.chat_log.record("system", text, "被动引擎")
        except Exception:
            pass
        try:
            print(text, file=sys.stderr, flush=True)
        except Exception:
            pass

    def poll_once(self):
        return self._check()

_engine_instance = None

def get_passive(lb, chat_log):
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = PassiveEngine(lb, chat_log)
    return _engine_instance

def start_passive(lb, chat_log):
    eng = get_passive(lb, chat_log)
    eng.start()
    return eng

def stop_passive():
    global _engine_instance
    if _engine_instance:
        _engine_instance.stop()
        _engine_instance = None
