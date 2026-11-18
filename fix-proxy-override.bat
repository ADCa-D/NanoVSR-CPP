@echo off
rem Add 66.66.66.67 to the system proxy bypass list.
rem Proxy clients (ApexCore / Clash) rewrite ProxyOverride when they restart,
rem which makes http://66.66.66.67:7861 unreachable from the browser even
rem though the forwarder is fine (curl --noproxy works). Run this to repair.
setlocal enabledelayedexpansion
set ADDR=66.66.66.67
set KEY=HKCU\Software\Microsoft\Windows\CurrentVersion\Internet Settings

for /f "tokens=2,*" %%a in ('reg query "%KEY%" /v ProxyOverride 2^>nul ^| findstr /i ProxyOverride') do set "OLD=%%b"

echo !OLD! | findstr /C:"%ADDR%" >nul
if not errorlevel 1 (
  echo [ok] already in bypass list: %ADDR%
) else (
  if "!OLD!"=="" (
    set "NEW=%ADDR%"
  ) else (
    set "NEW=!OLD!;%ADDR%"
  )
  reg add "%KEY%" /v ProxyOverride /t REG_SZ /d "!NEW!" /f >nul
  echo [ok] added: %ADDR%
)

echo.
echo verify (direct, no proxy):
curl.exe -s -o NUL -w "  http_code=%%{http_code}\n" --noproxy "*" --max-time 5 http://66.66.66.67:7861/health
echo.
echo if not 200, the forwarder is down: cd C:\Users\MECHREVO\Desktop\server ^&^& .\start-proxy.ps1
pause
