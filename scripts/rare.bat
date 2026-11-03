@echo off
cd /d "%~dp0.."
rem Synthetic RARE signs test (about 2-3 hours, 48x48 only):
rem 1) draw synthetic bus stop, no motorcycles, road narrows (both / left / right), turn left / right
rem    -> morocco_synth_rare (check morocco_synth_rare\preview.png)
rem 2) fine-tune the 3 48x48 models with Kaggle + own crops + synthetic speed + synthetic rare (runs *_maR)
rem 3) test everything on the YouTube set -> youtube_eval.txt (see "Rare classes" at the end)
python src\make_synth_rare.py --real morocco_extra --data ts1m_ma_32_clean --out morocco_synth_rare
for %%S in (42 1 2) do (
  if not exist runs\c48_relu40k_s%%S_maR\model_int8.tflite (
    echo fine-tuning c48 seed %%S ...
    python src\finetune_morocco.py --run runs\c48_relu40k_s%%S --data ts1m_ma_48_clean --morocco morocco_crops --extra morocco_extra --synth morocco_synth morocco_synth_rare --no-dup morocco_youtube --out runs\c48_relu40k_s%%S_maR --morocco-frac 0.15 --epochs 20 > runs\c48_relu40k_s%%S_maR.log 2>&1
  )
)
python src\eval_youtube.py --test morocco_youtube --data ts1m_ma_32_clean > runs\eval_youtube.log 2>&1
echo done. Results: youtube_eval.txt
