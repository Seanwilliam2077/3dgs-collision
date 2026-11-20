@echo off
rem Launcher: MSVC build env + venv on PATH (gsplat JIT-compiles its CUDA backend on first use).
rem chcp 936: torch decodes cl.exe output with the system OEM code page (936 here); if the
rem console is UTF-8 (65001) the localized compiler banner fails to decode and the build aborts.
call "C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
chcp 936 >nul
set "PATH=D:\3DGS_Research\.venv-gs\Scripts;%PATH%"
set DISTUTILS_USE_SDK=1
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
"D:\3DGS_Research\.venv-gs\Scripts\python.exe" %*
