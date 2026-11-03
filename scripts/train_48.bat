@echo off
cd /d "%~dp0.."
rem 48x48 experiment (long: about 8-9 hours, best overnight). Safe to restart: finished steps are skipped.
rem 1) ts1m_ma_48_clean = same images and labels as ts1m_ma_32_clean, in 48x48
rem 2) train 3 TS-1M models at 48x48 (same network as the 40k 32x32 model)
rem 3) fine-tune each with Kaggle + your crops + synthetic speed signs
rem 4) test everything on the YouTube set -> youtube_eval.txt
call conda activate tinyml 2>nul
if not exist ts1m_ma_48_clean\test (
  python src\make_48_clean.py --src48 ts1m_ma_48 --clean32 ts1m_ma_32_clean --out ts1m_ma_48_clean > runs\make_48_clean.log 2>&1
)
for %%S in (42 1 2) do (
  if not exist runs\c48_relu40k_s%%S\model_int8.tflite (
    echo training 48x48 seed %%S ...
    python src\train_tiny.py --data ts1m_ma_48_clean --size 48 --out runs\c48_relu40k_s%%S --head linear --widths 16,32,64,96,160 --seed %%S > runs\c48_relu40k_s%%S.log 2>&1
  )
  if not exist runs\c48_relu40k_s%%S_maT\model_int8.tflite (
    echo fine-tuning 48x48 seed %%S ...
    python src\finetune_morocco.py --run runs\c48_relu40k_s%%S --data ts1m_ma_48_clean --morocco morocco_crops --extra morocco_extra --synth morocco_synth --no-dup morocco_youtube --out runs\c48_relu40k_s%%S_maT --morocco-frac 0.15 --epochs 20 > runs\c48_relu40k_s%%S_maT.log 2>&1
  )
)
python src\eval_youtube.py --test morocco_youtube --data ts1m_ma_32_clean > runs\eval_youtube.log 2>&1
echo done. Results: youtube_eval.txt
