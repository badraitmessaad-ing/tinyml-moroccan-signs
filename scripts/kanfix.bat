@echo off
cd /d "%~dp0.."
rem Make the KAN activation survive int8: 3 KAN seeds + 3 same-size ReLU6 controls (same fine-tune length).
rem Uses the models already trained in runs\ (no full retraining). About 2-3 hours.
if not exist runs mkdir runs
for %%R in (c32_kanact_late c32_kanact_late_s1 c32_kanact_late_s2 c32_relu_wider c32_relu_wider_s1 c32_relu_wider_s2) do (
  echo %%R ...
  python src\kan_quant_fix.py --run runs\%%R --data ts1m_ma_32_clean --out runs\%%R_q > runs\%%R_q.log 2>&1
)
echo done. Send me the 6 quant_fix.txt files:  runs\*_q\quant_fix.txt
