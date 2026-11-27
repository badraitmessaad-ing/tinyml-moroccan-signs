@echo off
rem REVISION: new independent test drive (own dashcam, Tangier). Scored ONCE with the frozen final models.
cd /d "%~dp0"
call conda activate tinyml 2>nul
echo [1/3] checking that no new crop matches an old one ...
python check_new_drive.py morocco_dashcam > runs\check_dashcam.log 2>&1
echo [2/3] scoring the frozen final models (once) ...
python eval_final.py --test morocco_dashcam --name final_dashcam --runs runs\c48_relu40k_s42_maR runs\c48_relu40k_s1_maR runs\c48_relu40k_s2_maR > runs\eval_dashcam.log 2>&1
echo [3/3] saving the per-crop outputs for the safety and calibration analyses ...
python dump_predictions.py dashcam > runs\dump_dashcam.log 2>&1
echo DONE. Tell Claude.
pause
