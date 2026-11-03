@echo off
cd /d "%~dp0.."
rem Robust augmentation: retrain 40k RGB with occlusion/noise/low light/blur/JPEG/distant augmentation
rem (3 seeds, about 4-5 h), fine-tune each on Moroccan crops (setting B), then rerun the robustness test.
if not exist morocco_crops\morocco_map.csv (
  echo morocco_crops not found: run prepare_morocco_yolo.py first.
  exit /b 1
)
for %%S in (42 1 2) do (
  echo seed %%S: training 40k with robust augmentation ...
  python src\train_tiny.py --data ts1m_ma_32_clean --out runs\c32_rob40k_s%%S --head linear --widths 16,32,64,96,160 --robust-aug --seed %%S > runs\c32_rob40k_s%%S.log 2>&1
  echo seed %%S: Moroccan fine-tuning, setting B ...
  python src\finetune_morocco.py --run runs\c32_rob40k_s%%S --data ts1m_ma_32_clean --morocco morocco_crops --out runs\c32_rob40k_s%%S_maB --morocco-frac 0.15 --epochs 20 > runs\c32_rob40k_s%%S_maB.log 2>&1
)
echo robustness test: old vs robust models ...
python src\robustness_test.py --data ts1m_ma_32_clean --morocco morocco_crops ^
  runs\c32_relu40k_s42 runs\c32_relu40k_s1 runs\c32_relu40k_s2 ^
  runs\c32_rob40k_s42 runs\c32_rob40k_s1 runs\c32_rob40k_s2 ^
  runs\c32_relu40k_s42_maB runs\c32_relu40k_s1_maB runs\c32_relu40k_s2_maB ^
  runs\c32_rob40k_s42_maB runs\c32_rob40k_s1_maB runs\c32_rob40k_s2_maB > runs\robustness2.log 2>&1
copy /y robustness.txt robustness2.txt > nul
echo done. Zip the results with:
echo   Compress-Archive -Path robustness2.txt, runs\c32_rob40k_*.log -DestinationPath robust.zip -Force
