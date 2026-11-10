@echo off
rem MATCHED BASELINES (long: about 12-18 hours, best overnight). Safe to restart: finished steps are skipped.
rem Two baselines with about the same number of weights as our 40k model (39,942):
rem   plain = standard 3x3 convolutions          (widths 16,16,32,40,48 -> 40,182 weights)
rem   ir    = MobileNetV2 inverted residuals x4  (widths 16,16,24,40,64 -> 39,542 weights)
rem Same 48x48 data, same training settings, same Moroccan fine-tuning recipe (_maR) as the final model.
cd /d "%~dp0.."
call conda activate tinyml 2>nul
for %%A in (plain ir) do (
  for %%S in (42 1 2) do (
    if not exist runs\b48_%%A_s%%S\model_int8.tflite (
      echo training %%A seed %%S ...
      if "%%A"=="plain" python src\train_baselines.py --arch plain --widths 16,16,32,40,48 --data ts1m_ma_48_clean --size 48 --head linear --out runs\b48_%%A_s%%S --seed %%S > runs\b48_%%A_s%%S.log 2>&1
      if "%%A"=="ir" python src\train_baselines.py --arch ir --expand 4 --widths 16,16,24,40,64 --data ts1m_ma_48_clean --size 48 --head linear --out runs\b48_%%A_s%%S --seed %%S > runs\b48_%%A_s%%S.log 2>&1
    )
    if not exist runs\b48_%%A_s%%S_maR\model_int8.tflite (
      echo fine-tuning %%A seed %%S ...
      python src\finetune_morocco.py --run runs\b48_%%A_s%%S --data ts1m_ma_48_clean --morocco morocco_crops --extra morocco_extra --synth morocco_synth morocco_synth_rare --no-dup morocco_youtube --out runs\b48_%%A_s%%S_maR --morocco-frac 0.15 --epochs 20 > runs\b48_%%A_s%%S_maR.log 2>&1
    )
  )
)
echo evaluating on the development and final sets ...
python src\eval_final.py --test morocco_youtube --name dev_plain --runs runs\b48_plain_s42_maR runs\b48_plain_s1_maR runs\b48_plain_s2_maR > runs\eval_dev_plain.log 2>&1
python src\eval_final.py --test morocco_youtube --name dev_ir --runs runs\b48_ir_s42_maR runs\b48_ir_s1_maR runs\b48_ir_s2_maR > runs\eval_dev_ir.log 2>&1
python src\eval_final.py --test morocco_final --name final_plain --runs runs\b48_plain_s42_maR runs\b48_plain_s1_maR runs\b48_plain_s2_maR > runs\eval_final_plain.log 2>&1
python src\eval_final.py --test morocco_final --name final_ir --runs runs\b48_ir_s42_maR runs\b48_ir_s1_maR runs\b48_ir_s2_maR > runs\eval_final_ir.log 2>&1
echo checking RAM with TensorFlow Lite Micro ...
python src\tflm_check.py --data ts1m_ma_48_clean runs\b48_plain_s42_maR runs\b48_ir_s42_maR runs\c48_relu40k_s42_maR > runs\tflm_baselines.log 2>&1
echo.
echo DONE. Results: eval_dev_plain_report.txt, eval_dev_ir_report.txt, eval_final_plain_report.txt, eval_final_ir_report.txt, tflm_check.txt
pause
