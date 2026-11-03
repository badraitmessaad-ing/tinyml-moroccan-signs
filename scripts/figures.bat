@echo off
cd /d "%~dp0.."
rem Make all paper figures in the folder "figures" (a few minutes).
call conda activate tinyml 2>nul
python src\make_figures.py
echo.
echo Then compress the "figures" folder into figures.zip and tell me.
pause
