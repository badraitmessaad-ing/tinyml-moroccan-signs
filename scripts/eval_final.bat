@echo off
cd /d "%~dp0.."
REM 1) per-class metrics on the 696-crop DEVELOPMENT set (can be run any time)
python src\eval_final.py --test morocco_youtube --name dev
REM 2) the NEW final test set: run once, only after it is fully labelled. Remove REM below.
REM python src\eval_final.py --test morocco_final --name final
pause
