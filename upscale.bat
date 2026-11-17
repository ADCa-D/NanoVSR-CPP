@echo off
rem 把一个视频拖到这个 bat 上，输出同目录的 xxx_4x.mp4
setlocal
set HERE=%~dp0

if "%~1"=="" (
  echo 用法: 把视频文件拖到 upscale.bat 上
  echo       或  upscale.bat D:\video.mp4
  pause
  exit /b 1
)

set "IN=%~f1"
set "OUT=%~dpn1_4x.mp4"

echo 输入: %IN%
echo 输出: %OUT%
echo.
"%HERE%nanovsr.exe" --in "%IN%" --out "%OUT%" --crf 18

if errorlevel 1 (
  echo.
  echo 处理失败，看上面的报错
) else (
  echo.
  echo 完成: %OUT%
  start "" "%OUT%"
)
pause
