@echo off
rem 一键启动 NanoVSR 的 Web 面板
setlocal
set HERE=%~dp0
cd /d "%HERE%"

if not exist "nanovsr.exe" (
  echo [1/2] 首次运行，正在编译...
  call build.bat || goto :fail
)

set PY=
where python >nul 2>nul && set PY=python
if "%PY%"=="" if exist "D:\ComfyUI_windows_portable\python_embeded\python.exe" set PY=D:\ComfyUI_windows_portable\python_embeded\python.exe
if "%PY%"=="" (
  echo 找不到 python，请安装或修改本脚本里的路径
  pause
  exit /b 1
)

echo 启动面板 http://127.0.0.1:7861
start "" "%PY%" "%HERE%webui.py" --port 7861
timeout /t 2 >nul
start "" http://127.0.0.1:7861
exit /b 0

:fail
echo 编译失败
pause
exit /b 1
