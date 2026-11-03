@echo off
cd /d "%~dp0.."
rem Fine-tune with Kaggle crops + your own Moroccan speed-limit crops (morocco_extra\train\...),
rem then test everything on the YouTube test set. About 1 hour (6 fine-tunes).
call conda activate tinyml 2>nul
if not exist morocco_extra\train (
  echo morocco_extra\train not found - crop the new videos first with:
  echo python src\crop_tool.py --shots screenshots_train --out morocco_extra --split train
  exit /b 1
)
for %%S in (s42 s1 s2) do (
  python src\finetune_morocco.py --run runs\c32_mild40k_%%S --data ts1m_ma_32_clean --morocco morocco_crops --extra morocco_extra --no-dup morocco_youtube --out runs\c32_mild40k_%%S_maS --morocco-frac 0.15 --epochs 20 > runs\c32_mild40k_%%S_maS.log 2>&1
  python src\finetune_morocco.py --run runs\c32_relu40k_%%S --data ts1m_ma_32_clean --morocco morocco_crops --extra morocco_extra --no-dup morocco_youtube --out runs\c32_relu40k_%%S_maS --morocco-frac 0.15 --epochs 20 > runs\c32_relu40k_%%S_maS.log 2>&1
)
python src\eval_youtube.py --test morocco_youtube --data ts1m_ma_32_clean > runs\eval_youtube.log 2>&1
echo done. Results: youtube_eval.txt
