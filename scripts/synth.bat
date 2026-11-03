@echo off
cd /d "%~dp0.."
rem Synthetic speed signs test (about 40 minutes):
rem 1) draw synthetic speed signs + digit swaps from your own crops -> morocco_synth (check morocco_synth\preview.png)
rem 2) fine-tune the 3 standard models with Kaggle + your crops + synthetic speed signs (runs *_maT)
rem 3) test everything on the YouTube set -> youtube_eval.txt
call conda activate tinyml 2>nul
python src\make_synth_speed.py --real morocco_extra --data ts1m_ma_32_clean --out morocco_synth
for %%S in (s42 s1 s2) do (
  python src\finetune_morocco.py --run runs\c32_relu40k_%%S --data ts1m_ma_32_clean --morocco morocco_crops --extra morocco_extra --synth morocco_synth --no-dup morocco_youtube --out runs\c32_relu40k_%%S_maT --morocco-frac 0.15 --epochs 20 > runs\c32_relu40k_%%S_maT.log 2>&1
)
python src\eval_youtube.py --test morocco_youtube --data ts1m_ma_32_clean > runs\eval_youtube.log 2>&1
echo done. Results: youtube_eval.txt
