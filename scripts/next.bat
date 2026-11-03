@echo off
cd /d "%~dp0.."
rem A) better fine-tuning settings (40k, 3 seeds x 2 settings, ~1 h)
rem B) grayscale 40k TS-1M models (3 seeds, ~3 h) + Moroccan fine-tuning of each (~15 min)
if not exist morocco_crops\morocco_map.csv (
  echo morocco_crops not found: run prepare_morocco_yolo.py first.
  exit /b 1
)
for %%S in (s42 s1 s2) do (
  echo A %%S: lr 1e-4, 20 epochs ...
  python src\finetune_morocco.py --run runs\c32_relu40k_%%S --data ts1m_ma_32_clean --morocco morocco_crops --out runs\c32_relu40k_%%S_maA --lr 1e-4 --epochs 20 > runs\c32_relu40k_%%S_maA.log 2>&1
  echo A %%S: 15%% Moroccan, 20 epochs ...
  python src\finetune_morocco.py --run runs\c32_relu40k_%%S --data ts1m_ma_32_clean --morocco morocco_crops --out runs\c32_relu40k_%%S_maB --morocco-frac 0.15 --epochs 20 > runs\c32_relu40k_%%S_maB.log 2>&1
)
for %%S in (42 1 2) do (
  echo B seed %%S: grayscale 40k on TS-1M ...
  python src\train_tiny.py --data ts1m_ma_32_clean --out runs\c32_gray40k_s%%S --head linear --widths 16,32,64,96,160 --gray --seed %%S > runs\c32_gray40k_s%%S.log 2>&1
  echo B seed %%S: Moroccan fine-tuning ...
  python src\finetune_morocco.py --run runs\c32_gray40k_s%%S --data ts1m_ma_32_clean --morocco morocco_crops --out runs\c32_gray40k_s%%S_ma > runs\c32_gray40k_s%%S_ma.log 2>&1
)
echo done. Zip the results with:
echo   Compress-Archive -Path runs\*_maA.log, runs\*_maB.log, runs\c32_gray40k_*.log -DestinationPath next.zip -Force
