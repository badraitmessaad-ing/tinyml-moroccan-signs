@echo off
cd /d "%~dp0.."
rem KAN table compression tests on the 3 trained KAN models (no training, about 1-1.5 hours)
python src\kan_compress.py --data ts1m_ma_32_clean runs\c32_kanact_late runs\c32_kanact_late_s1 runs\c32_kanact_late_s2 > kan_compress.log 2>&1
echo done. Send me kan_compress.txt
