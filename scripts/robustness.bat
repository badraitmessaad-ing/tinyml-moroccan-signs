@echo off
cd /d "%~dp0.."
rem Robustness test of the int8 models (no training, about 20-40 min on CPU).
rem Missing run folders are skipped automatically.
if not exist morocco_crops\morocco_map.csv (
  echo morocco_crops not found: run prepare_morocco_yolo.py first.
  exit /b 1
)
python src\robustness_test.py --data ts1m_ma_32_clean --morocco morocco_crops ^
  runs\c32_relu_wider ^
  runs\c32_relu40k_s42 runs\c32_relu40k_s1 runs\c32_relu40k_s2 ^
  runs\c32_relu52k_s42 runs\c32_relu52k_s1 runs\c32_relu52k_s2 ^
  runs\c32_relu40k_s42_maB runs\c32_relu40k_s1_maB runs\c32_relu40k_s2_maB ^
  runs\c32_gray40k_s42 runs\c32_gray40k_s1 runs\c32_gray40k_s2 ^
  runs\c32_gray40k_s42_ma runs\c32_gray40k_s1_ma runs\c32_gray40k_s2_ma > runs\robustness.log 2>&1
echo done. Send me robustness.txt (and runs\robustness.log if something failed).
