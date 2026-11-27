@echo off
rem REVISION (R2.4): score the two matched baselines (inverted residual, plain CNN) ONCE on the Tangier drive,
rem so the comparison with our model covers all three held-out drives. Nothing is retrained.
cd /d "%~dp0"
call conda activate tinyml 2>nul
echo [1/2] inverted residual ...
python eval_final.py --test morocco_dashcam --name dashcam_ir --runs runs\b48_ir_s42_maR runs\b48_ir_s1_maR runs\b48_ir_s2_maR > runs\eval_dashcam_ir.log 2>&1
echo [2/2] plain CNN ...
python eval_final.py --test morocco_dashcam --name dashcam_plain --runs runs\b48_plain_s42_maR runs\b48_plain_s1_maR runs\b48_plain_s2_maR > runs\eval_dashcam_plain.log 2>&1
echo DONE. Tell Claude.
pause
