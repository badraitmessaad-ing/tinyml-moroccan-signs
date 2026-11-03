@echo off
cd /d "%~dp0.."
rem Test all models on the new Moroccan YouTube test set (no training, a few minutes).
call conda activate tinyml 2>nul
python src\eval_youtube.py --test morocco_youtube --data ts1m_ma_32_clean > runs\eval_youtube.log 2>&1
echo done. Results: youtube_eval.txt and youtube_errors.html
