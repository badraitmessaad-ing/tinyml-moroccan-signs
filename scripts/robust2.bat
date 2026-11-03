@echo off
cd /d "%~dp0.."
rem Keep clean accuracy, add robustness. About 5 h. Steps that already finished are skipped,
rem so after a power cut just run this file again.
rem  1) robust fine-tune: clean 40k models + 15 epochs of robust augmentation (lr 3e-4, last epoch kept)
rem  2) mild robust training from scratch: robust augmentation half as often (--robust-p 0.5)
rem  then Moroccan fine-tuning (setting B) of each, and the robustness test of everything.
if not exist morocco_crops\morocco_map.csv (
  echo morocco_crops not found: run prepare_morocco_yolo.py first.
  exit /b 1
)
for %%S in (42 1 2) do (
  if not exist runs\c32_robft40k_s%%S\model_int8.tflite (
    echo seed %%S: robust fine-tune of the clean 40k model ...
    python src\train_tiny.py --data ts1m_ma_32_clean --out runs\c32_robft40k_s%%S --head linear --widths 16,32,64,96,160 --robust-aug --init runs\c32_relu40k_s%%S --epochs 15 --lr 3e-4 --no-early-stop --seed %%S > runs\c32_robft40k_s%%S.log 2>&1
  )
  if not exist runs\c32_robft40k_s%%S_maB\finetune_results.txt (
    echo seed %%S: Moroccan fine-tuning of the robust fine-tuned model ...
    python src\finetune_morocco.py --run runs\c32_robft40k_s%%S --data ts1m_ma_32_clean --morocco morocco_crops --out runs\c32_robft40k_s%%S_maB --morocco-frac 0.15 --epochs 20 > runs\c32_robft40k_s%%S_maB.log 2>&1
  )
)
for %%S in (42 1 2) do (
  if not exist runs\c32_mild40k_s%%S\model_int8.tflite (
    echo seed %%S: mild robust training from scratch ...
    python src\train_tiny.py --data ts1m_ma_32_clean --out runs\c32_mild40k_s%%S --head linear --widths 16,32,64,96,160 --robust-aug --robust-p 0.5 --seed %%S > runs\c32_mild40k_s%%S.log 2>&1
  )
  if not exist runs\c32_mild40k_s%%S_maB\finetune_results.txt (
    echo seed %%S: Moroccan fine-tuning of the mild model ...
    python src\finetune_morocco.py --run runs\c32_mild40k_s%%S --data ts1m_ma_32_clean --morocco morocco_crops --out runs\c32_mild40k_s%%S_maB --morocco-frac 0.15 --epochs 20 > runs\c32_mild40k_s%%S_maB.log 2>&1
  )
)
echo robustness test: clean vs robust vs robust fine-tune vs mild ...
python src\robustness_test.py --data ts1m_ma_32_clean --morocco morocco_crops ^
  runs\c32_relu40k_s42 runs\c32_relu40k_s1 runs\c32_relu40k_s2 ^
  runs\c32_rob40k_s42 runs\c32_rob40k_s1 runs\c32_rob40k_s2 ^
  runs\c32_robft40k_s42 runs\c32_robft40k_s1 runs\c32_robft40k_s2 ^
  runs\c32_mild40k_s42 runs\c32_mild40k_s1 runs\c32_mild40k_s2 ^
  runs\c32_relu40k_s42_maB runs\c32_relu40k_s1_maB runs\c32_relu40k_s2_maB ^
  runs\c32_rob40k_s42_maB runs\c32_rob40k_s1_maB runs\c32_rob40k_s2_maB ^
  runs\c32_robft40k_s42_maB runs\c32_robft40k_s1_maB runs\c32_robft40k_s2_maB ^
  runs\c32_mild40k_s42_maB runs\c32_mild40k_s1_maB runs\c32_mild40k_s2_maB > runs\robustness3.log 2>&1
copy /y robustness.txt robustness3.txt > nul
echo done.
