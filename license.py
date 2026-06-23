# -*- coding: utf-8 -*-
"""
知络 许可证模块 — 7天试用 + 激活码永久解锁
==========================================
- 首次运行记录安装时间
- 每次工具调用前检查
- 7天后提示激活
- 激活码由 keygen.py 生成
"""
import os, sys, time, hashlib, hmac, base64, json, socket

# ========== 配置 ==========
TRIAL_DAYS = 7
SECRET_KEY = b"zhiluo_2026_core_key"   # 与 keygen.py 保持一致
SKILL_DIR = os.path.dirname(os.path.abspath(__file__))

# 数据目录：兼容 Windows 和 Linux
def _data_dir():
    home = os.path.expanduser("~")
    d = os.path.join(home, ".zhiluo")
    os.makedirs(d, exist_ok=True)
    return d

LICENSE_FILE = os.path.join(_data_dir(), "license.dat")

# ========== 工具函数 ==========

def _xor_cipher(data: bytes, key: bytes = b"ZhiLuo") -> bytes:
    """简单 XOR 混淆"""
    return bytes(d ^ key[i % len(key)] for i, d in enumerate(data))

def _get_machine_code() -> str:
    """生成机器特征码（主机名+IP哈希）"""
    try:
        host = socket.gethostname()
        ip = socket.gethostbyname(host)
    except Exception:
        host = "unknown"
        ip = "0.0.0.0"
    raw = f"{host}|{ip}|{os.name}"
    return hashlib.sha256(raw.encode()).hexdigest()[:12]

def _read_license():
    """读取许可证文件，返回 dict 或 None"""
    if not os.path.exists(LICENSE_FILE):
        return None
    try:
        with open(LICENSE_FILE, "rb") as f:
            enc = f.read()
        dec = _xor_cipher(enc)
        return json.loads(dec.decode("utf-8"))
    except Exception:
        return None

def _write_license(data: dict):
    """写入许可证文件"""
    raw = json.dumps(data).encode("utf-8")
    enc = _xor_cipher(raw)
    with open(LICENSE_FILE, "wb") as f:
        f.write(enc)

def _generate_activation_code(machine_code: str) -> str:
    """根据机器码生成激活码（与 keygen.py 使用相同算法）"""
    h = hmac.new(SECRET_KEY, machine_code.encode(), hashlib.sha256)
    return h.hexdigest()[:16].upper()


# ========== 对外接口 ==========

def get_status():
    """返回当前许可证状态
    Returns: {"status": "trial"|"expired"|"activated", "days_left": int, "machine_code": str}
    """
    mc = _get_machine_code()
    lic = _read_license()

    # 从未激活过 → 创建试用记录
    if lic is None:
        lic = {
            "install_ts": int(time.time()),
            "activated": False,
            "activation_code": "",
            "machine_code": mc,
        }
        _write_license(lic)

    # 已激活 → 直接放行
    if lic.get("activated") and lic.get("activation_code"):
        return {"status": "activated", "days_left": -1, "machine_code": mc}

    # 试用中 → 计算剩余天数
    install_ts = lic.get("install_ts", int(time.time()))
    elapsed = int(time.time()) - install_ts
    days_used = elapsed // 86400
    days_left = max(0, TRIAL_DAYS - days_used)

    if days_left <= 0:
        return {"status": "expired", "days_left": 0, "machine_code": mc}
    return {"status": "trial", "days_left": days_left, "machine_code": mc}


def activate(code: str):
    """输入激活码，成功返回 True"""
    mc = _get_machine_code()
    expected = _generate_activation_code(mc)
    if code.strip().upper() == expected:
        lic = _read_license() or {}
        lic["activated"] = True
        lic["activation_code"] = code.strip().upper()
        lic["machine_code"] = mc
        _write_license(lic)
        return True
    return False


def check_license(context="core"):
    """
    许可证检查，供工具调用前使用。
    返回 None = 放行
    返回 str = 错误提示（应该返回给用户）
    """
    status = get_status()
    if status["status"] == "activated":
        return None

    if status["status"] == "expired":
        mc = status["machine_code"]
        return (
            f"[知络] 7天试用已到期。\n"
            f"你的机器码：{mc}\n"
            f"购买激活码请联系作者（微信号：xxxxx）\n"
            f"激活命令：activate(activation_code='你的激活码')"
        )

    # 试用中 — 显示剩余天数（不用拦截）
    return None


def get_machine_code():
    """获取本机机器码"""
    return _get_machine_code()
