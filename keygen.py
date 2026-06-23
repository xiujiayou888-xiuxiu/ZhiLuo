# -*- coding: utf-8 -*-
"""
知络 KeyGen — 激活码生成器（仅作者使用）
=========================================
用法：python keygen.py <机器码>
示例：python keygen.py a1b2c3d4e5f6
      → 激活码：1A2B3C4D5E6F7G8H
"""
import hashlib, hmac, sys, os

SECRET_KEY = b"zhiluo_2026_core_key"   # 与 license.py 保持一致

def generate(machine_code: str) -> str:
    """根据机器码生成激活码"""
    h = hmac.new(SECRET_KEY, machine_code.strip().encode(), hashlib.sha256)
    return h.hexdigest()[:16].upper()

def main():
    if len(sys.argv) < 2:
        print("用法：python keygen.py <机器码>")
        print("示例：python keygen.py a1b2c3d4e5f6")
        return

    mc = sys.argv[1].strip()
    if len(mc) < 8:
        print("❌ 机器码太短，请确认输入正确")
        return

    code = generate(mc)
    print(f"机器码：{mc}")
    print(f"激活码：{code}")
    print(f"\n客户指令：activate(activation_code='{code}')")

if __name__ == "__main__":
    main()
