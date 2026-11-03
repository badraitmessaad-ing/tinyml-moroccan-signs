@echo off
cd /d "%~dp0.."
rem Robust fine-tune of the clean 40k models (about 1.5-2 h). Finished steps are skipped,
rem so after a power cut just run this file again.
call conda activate tinyml 2>nul
python -c "import sys, keras; print(\"python:\", sys.prefix, \" keras:\", keras.__version__)"
if not exist morocco_crops\morocco_map.csv (
  echo morocco_crops not found: run prepare_morocco_yolo.py first.
  exit /b 1
)
for %%S in (42 1 2) do (
  if not exist runs\c32_robft40k_s%%S\model_int8.tflite (
    echo seed %%S: robust fine-tune of the clean 40k model ...
    python src\train_tiny.py --data ts1m_ma_32_clean --out runs\c32_robft40k_s%%S --head linear --widths 16,32,64,96,160 --robust-aug --init runs\c32_relu40k_s%%S --epochs 15 --lr 3e-4 --no-early-stop --seed %%S > runs\c32_robft40k_s%%S.log 2>&1
    if not exist runs\c32_robft40k_s%%S\model_int8.tflite echo    FAILED - see runs\c32_robft40k_s%%S.log
  )
  if not exist runs\c32_robft40k_s%%S_maB\finetune_results.txt (
    echo seed %%S: Moroccan fine-tuning ...
    python src\finetune_morocco.py --run runs\c32_robft40k_s%%S --data ts1m_ma_32_clean --morocco morocco_crops --out runs\c32_robft40k_s%%S_maB --morocco-frac 0.15 --epochs 20 > runs\c32_robft40k_s%%S_maB.log 2>&1
  )
)
echo robustness test: clean vs mild vs robust fine-tune ...
python src\robustness_test.py --data ts1m_ma_32_clean --morocco morocco_crops ^
  runs\c32_relu40k_s42 runs\c32_relu40k_s1 runs\c32_relu40k_s2 ^
  runs\c32_mild40k_s42 runs\c32_mild40k_s1 runs\c32_mild40k_s2 ^
  runs\c32_robft40k_s42 runs\c32_robft40k_s1 runs\c32_robft40k_s2 ^
  runs\c32_relu40k_s42_maB runs\c32_relu40k_s1_maB runs\c32_relu40k_s2_maB ^
  runs\c32_mild40k_s42_maB runs\c32_mild40k_s1_maB runs\c32_mild40k_s2_maB ^
  runs\c32_robft40k_s42_maB runs\c32_robft40k_s1_maB runs\c32_robft40k_s2_maB > runs\robustness4.log 2>&1
copy /y robustness.txt robustness4.txt > nul
echo done.
