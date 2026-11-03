@echo off
cd /d "%~dp0.."
rem Novelty experiments with the mild 40k models (about 2 h). Finished steps are skipped,
rem so after a power cut just run this file again. Run it inside: conda activate tinyml
call conda activate tinyml 2>nul
if not exist morocco_crops\morocco_map.csv (
  echo morocco_crops not found: run prepare_morocco_yolo.py first.
  exit /b 1
)
rem 1) Few-shot curve with full fine-tuning: K Moroccan crops per class (K = all is c32_mild40k_sX_maB)
for %%S in (42 1 2) do (
  for %%K in (5 10 25 50) do (
    if not exist runs\c32_mild40k_s%%S_k%%K\finetune_results.txt (
      echo seed %%S, %%K crops per class: full fine-tuning ...
      python src\finetune_morocco.py --run runs\c32_mild40k_s%%S --data ts1m_ma_32_clean --morocco morocco_crops --out runs\c32_mild40k_s%%S_k%%K --morocco-frac 0.15 --epochs 20 --per-class %%K --seed %%S > runs\c32_mild40k_s%%S_k%%K.log 2>&1
    )
  )
)
rem 2) Last-layer (on-device) adaptation curve, unknown-sign rejection, multi-frame voting
echo novelty tests ...
python src\novelty_tests.py --data ts1m_ma_32_clean --morocco morocco_crops --runs runs\c32_mild40k_s42 runs\c32_mild40k_s1 runs\c32_mild40k_s2 > runs\novelty.log 2>&1
echo done. Results: novelty.txt and runs\c32_mild40k_s*_k*\finetune_results.txt
