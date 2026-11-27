@echo off
rem REVISION: two new RAIN test drives (Rabat, Tangier). Labelled blind, scored ONCE with the frozen final models.
rem Run only after BOTH folders are fully labelled with crop_tool --blind. Do not run twice.
cd /d "%~dp0"
call conda activate tinyml 2>nul
set PYTHONIOENCODING=utf-8
echo [1/5] novelty check, Rabat rain ...
python check_new_drive.py morocco_rain_rabat > runs\check_rain_rabat.log 2>&1
echo [2/5] novelty check, Tangier rain ...
python check_new_drive.py morocco_rain_tanger > runs\check_rain_tanger.log 2>&1
echo [3/5] scoring the frozen final models on Rabat rain (once) ...
python eval_final.py --test morocco_rain_rabat --name final_rain_rabat --runs runs\c48_relu40k_s42_maR runs\c48_relu40k_s1_maR runs\c48_relu40k_s2_maR > runs\eval_rain_rabat.log 2>&1
echo [4/5] scoring the frozen final models on Tangier rain (once) ...
python eval_final.py --test morocco_rain_tanger --name final_rain_tanger --runs runs\c48_relu40k_s42_maR runs\c48_relu40k_s1_maR runs\c48_relu40k_s2_maR > runs\eval_rain_tanger.log 2>&1
echo [5/5] saving per-crop outputs for the analyses ...
python dump_predictions.py rain_rabat rain_tanger > runs\dump_rain.log 2>&1
echo DONE. Tell Claude.
pause
