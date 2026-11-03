@echo off
rem Run once, before the first push. Fills data\, models\ and requirements-lock.txt from the project folder.
cd /d "%~dp0.."
call conda activate tinyml 2>nul
python tools\export_release.py --project "%~dp0..\..\.."
pause
