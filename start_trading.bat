@echo off
title AUD/USD Quant Trading System v9

echo ==========================================
echo  AUD/USD 量化交易系统 v9 — 启动中...
echo ==========================================

cd /d C:\Users\marti\Desktop\AUD_USD_quant

echo [1/2] 启动 Web 管理面板 (后台)...
start "Web Panel" pythonw webapp\app.py
echo       http://localhost:6060/

timeout /t 3 /nobreak >nul

echo [2/2] 启动自动交易引擎...
python live_trading.py --paper

echo.
echo 系统已停止。
pause