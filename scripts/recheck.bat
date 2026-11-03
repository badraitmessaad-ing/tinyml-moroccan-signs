cd /d "%~dp0.."
python src\eval_recheck.py --test morocco_final --name final
python src\eval_recheck.py --test morocco_youtube --name dev --steps
pause
