@echo off
cd /d "%~dp0.."
rem Fine-tune with Moroccan crops: ReLU6 40k (3 seeds) + 52k (1 seed). About 1 hour.
rem Needs morocco_crops\ from prepare_morocco_yolo.py
if not exist morocco_crops\morocco_map.csv (
  echo morocco_crops not found: run prepare_morocco_yolo.py first.
  exit /b 1
)
for %%R in (c32_relu40k_s42 c32_relu40k_s1 c32_relu40k_s2 c32_relu52k_s42) do (
  echo %%R ...
  python src\finetune_morocco.py --run runs\%%R --data ts1m_ma_32_clean --morocco morocco_crops --out runs\%%R_ma > runs\%%R_ma.log 2>&1
)
echo done. Zip the results with:
echo   Compress-Archive -Path runs\*_ma.log -DestinationPath finetune.zip -Force
