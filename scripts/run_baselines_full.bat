@echo off
rem REVISION (R1.4 / R2.4): evaluate the two matched baselines exactly like our model, from A to Z.
rem The baselines (runs\b48_ir_s*_maR, runs\b48_plain_s*_maR) were trained after our models were frozen; nothing is retrained here.
rem 1) save their int8 outputs on every set (development, final test, Tangier, both rain drives)
rem 2) compute the same statistics as for our model: per-drive accuracy with sign-level CIs, macro-F1, speed limits,
rem    pooled held-out accuracy and paired difference, error severity with/without tau = 0.3, calibration and risk-coverage.
cd /d "%~dp0"
call conda activate tinyml 2>nul
set PYTHONIOENCODING=utf-8
echo [1/3] inverted residual: predictions on all 5 sets ...
python dump_predictions.py dev final dashcam rain_rabat rain_tanger --model ir > runs\dump_ir.log 2>&1
echo [2/3] plain CNN: predictions on all 5 sets ...
python dump_predictions.py dev final dashcam rain_rabat rain_tanger --model plain > runs\dump_plain.log 2>&1
echo [3/3] same analysis as for our model ...
copy /Y revision_out\preds_*.npz revision_analysis\ >nul
copy /Y dataset_release\MRSC-V_v1.0_full\annotations\dev.csv revision_analysis\ >nul
copy /Y dataset_release\MRSC-V_v1.0_full\annotations\final_test.csv revision_analysis\ >nul
pushd revision_analysis
python baseline_compare.py > out_15_baselines_full.txt 2>&1
popd
echo DONE. Results in revision_analysis\out_15_baselines_full.txt. Tell Claude.
pause
