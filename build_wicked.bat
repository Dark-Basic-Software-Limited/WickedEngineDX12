@echo off
cd /D "D:\max\WickedEngineDX12"
call "C:\Program Files\Microsoft Visual Studio\18\Community\Common7\Tools\VsDevCmd.bat" -arch=amd64 >nul 2>&1
msbuild "D:\max\WickedEngineDX12\WickedEngine.sln" /p:Configuration=Release /p:Platform=x64 /t:WickedEngine_Windows /m /verbosity:minimal
