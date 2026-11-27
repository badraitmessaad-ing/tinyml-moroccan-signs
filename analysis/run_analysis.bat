@echo off
REM Re-creates all revision analyses (IC major revision) from the frozen models' saved predictions.
REM Run from anywhere:   revision_analysis\run_analysis.bat
REM Needs: revision_out\preds_dev.npz, preds_final.npz, preds_dashcam.npz, preds_rain_rabat.npz, preds_rain_tanger.npz (from dump_predictions.py)
REM        dataset_release\MRSC-V_v1.0_full\annotations\dev.csv and final_test.csv
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
copy /Y ..\revision_out\preds_dev.npz . >nul
copy /Y ..\revision_out\preds_final.npz . >nul
copy /Y ..\revision_out\preds_dashcam.npz . >nul
copy /Y ..\revision_out\preds_rain_rabat.npz . >nul
copy /Y ..\revision_out\preds_rain_tanger.npz . >nul
copy /Y ..\dataset_release\MRSC-V_v1.0_full\annotations\dev.csv . >nul
copy /Y ..\dataset_release\MRSC-V_v1.0_full\annotations\final_test.csv . >nul
echo 1/14 per-drive accuracy, calibration, selective prediction, safety rule sweep
python analyse.py > out_1_per_drive_calibration.txt
echo 2/14 error severity by driver action
python safety_groups.py > out_2_severity.txt
echo 3/14 severity with the safety rule tau = 0.3
python safety_tau.py prob > out_3_safety_rule.txt
echo 4/14 physical-sign grouping sensitivity (10-120 s)
python sign_groups.py > out_4_sign_grouping_sensitivity.txt
echo 5/14 sign IDs (2 per sign, 10 s) and sign-level CIs
python sign_rule.py > out_5_sign_ids_cis.txt
echo 6/14 macro-F1 / balanced accuracy CIs, voting on repeated screenshots, rho
python more_stats.py > out_6_macroF1_voting_rho.txt
echo 7/14 figure: reliability + risk-coverage
python fig_calib.py > out_7_figure.txt
echo 8/14 figure: GTSRB size-accuracy Pareto plot
python fig_pareto.py > out_8_pareto.txt
echo 9/14 camera-in-the-loop summary (ESP32-CAM)
python camera_summary.py > out_9_camera_test.txt
echo 10/14 ESP32 energy from the USB meter readings
python energy_calc.py > out_10_energy.txt
echo 11/14 digital twin at the measured rho = 0.29 (writes figures\fig9_twin_rev.png in the project folder)
pushd ..
python twin_sim_rev.py > revision_analysis\out_11_twin.txt
popd
echo 12/14 class coverage by regulation category, out-of-scope share per drive (R1.9)
python class_coverage.py > out_12_class_coverage.txt
echo 13/14 per-drive table of all development and held-out drives (manuscript Table)
python heldout_table.py > out_13_per_drive_table.txt
echo 14/14 figure: hardware tests (ESP32-CAM bench test, USB power meter, frames seen on the chip)
python fig_hardware.py > out_14_hardware_figure.txt
echo.
echo Done. Results: revision_analysis\out_*.txt, fig_calibration_risk_coverage.*, fig_gtsrb_pareto.* figures\fig9_twin_rev.* and figures\fig_hardware.*
pause
