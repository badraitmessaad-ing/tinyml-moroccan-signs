@echo off
cd /d "%~dp0.."
rem Calibrate on-device prototype adaptation (no training, about 15-20 min).
call conda activate tinyml 2>nul
python src\proto_calib.py --data ts1m_ma_32_clean --morocco morocco_crops --runs runs\c32_mild40k_s42 runs\c32_mild40k_s1 runs\c32_mild40k_s2 > runs\proto_calib.log 2>&1
echo done. Result: proto_calib.txt
