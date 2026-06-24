@echo off
chcp 65001 >nul
echo ============================================================
echo  重譯文本分析系統 AI Agent 啟動程式
echo ============================================================
echo.

cd /d "%~dp0"

echo [1/2] 安裝/確認相依套件...
pip install -r requirements_agent.txt --quiet
if %errorlevel% neq 0 (
    echo [錯誤] 套件安裝失敗，請手動執行：pip install -r requirements_agent.txt
    pause
    exit /b 1
)

echo.
echo [2/2] 啟動 Gradio 應用程式...
echo      網址：http://localhost:7860
echo      按 Ctrl+C 停止服務
echo.
python app_agent.py

pause
