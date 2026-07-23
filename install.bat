@echo off
chcp 65001 >nul
echo ========================================
echo   知络基础版 — 安装脚本
echo ========================================
echo.

cd /d "%~dp0"

echo [1/3] 创建虚拟环境...
python -m venv venv
if %errorlevel% neq 0 (
    echo 错误: 创建虚拟环境失败，请确认已安装 Python 3.10+
    pause
    exit /b 1
)

echo [2/3] 安装依赖...
venv\Scripts\python -m pip install --upgrade pip -q
venv\Scripts\pip install -r requirements.txt -q
if %errorlevel% neq 0 (
    echo 错误: 依赖安装失败
    pause
    exit /b 1
)

echo [3/3] 编译检查...
venv\Scripts\python -m compileall -q .
if %errorlevel% neq 0 (
    echo 警告: 编译检查有错误，但不影响基本功能
)

echo.
echo ========================================
echo   安装完成！
echo ========================================
echo.
echo 下一步：
echo   1. 在你的Agent配置中添加MCP server
echo   2. 对AI说 setup(action="quick") 初始化
echo.
pause
