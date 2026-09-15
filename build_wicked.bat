@echo off
SET CONFIG=%1
IF "%CONFIG%"=="" SET CONFIG=Debug
cd /D "D:\max\WickedEngineDX12"
REM --- Initialize VS Developer Environment ---
REM GGMAX 3.38: pick the install that EXISTS on this machine rather than hardcoding one - the
REM desktop has VS 2026 (18\Community) and the laptop only has VS 2022 Community.
REM WARNING: do NOT pin this to 2022\Community. On a box that also has a 2022 BuildTools install,
REM both provide toolset v143 at the SAME version, VsDevCmd sets INCLUDE from one while MSBuild
REM resolves the toolset to the other, and the CRT headers are then seen twice:
REM   excpt.h(22,14): error C2011: '_EXCEPTION_DISPOSITION': 'enum' type redefinition
REM That failure is LATENT - it only appears once a file in the affected sub-projects actually
REM recompiles, so an incremental build can look green for weeks.
SET "GG_VSDEV="
IF EXIST "C:\Program Files\Microsoft Visual Studio\18\Community\Common7\Tools\VsDevCmd.bat" SET "GG_VSDEV=C:\Program Files\Microsoft Visual Studio\18\Community\Common7\Tools\VsDevCmd.bat"
IF NOT DEFINED GG_VSDEV IF EXIST "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" SET "GG_VSDEV=C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat"
IF NOT DEFINED GG_VSDEV (
    ECHO ERROR: no supported Visual Studio install found ^(looked for 18\Community then 2022\Community^).
    EXIT /B 9
)
call "%GG_VSDEV%" -arch=amd64 >nul 2>&1
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
