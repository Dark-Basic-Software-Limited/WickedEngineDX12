@echo off
SET CONFIG=%1
IF "%CONFIG%"=="" SET CONFIG=Debug
cd /D "D:\max\WickedEngineDX12"
call "C:\Program Files\Microsoft Visual Studio\18\Community\Common7\Tools\VsDevCmd.bat" -arch=amd64 >nul 2>&1
echo Building WickedEngine_Windows %CONFIG% x64...
msbuild "D:\max\WickedEngineDX12\WickedEngine.sln" /p:Configuration=%CONFIG% /p:Platform=x64 /t:WickedEngine_Windows /m /verbosity:minimal
IF ERRORLEVEL 1 (
    echo BUILD FAILED
    exit /b 1
) ELSE (
    echo BUILD SUCCEEDED
    echo Refreshing stale engine shader .cso ...
    powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0refresh_shaders.ps1"
    exit /b 0
)
