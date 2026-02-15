@echo off
call "C:\Program Files\Microsoft Visual Studio\18\Community\Common7\Tools\VsDevCmd.bat" -arch=amd64 >nul 2>&1
cd /d "D:\max\WickedEngineDX12"
echo Building WickedEngine_Windows Release (forced rebuild)...
msbuild WickedEngine.sln /p:Configuration=Release /p:Platform=x64 /t:WickedEngine_Windows:Rebuild /m /verbosity:minimal
echo Exit code: %ERRORLEVEL%
