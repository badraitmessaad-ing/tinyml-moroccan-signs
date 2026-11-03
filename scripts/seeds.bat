@echo off
cd /d "%~dp0.."
rem Two more seeds of the KAN activation model and its same-size ReLU6 control.
if not exist runs mkdir runs
for %%S in (1 2) do (
  echo seed %%S: KAN activation ...
  python src\train_tiny.py --data ts1m_ma_32_clean --out runs\c32_kanact_late_s%%S --head linear --act kan --act-grid 4 --act-from 3 --seed %%S > runs\c32_kanact_late_s%%S.log 2>&1
  echo seed %%S: ReLU6 wider ...
  python src\train_tiny.py --data ts1m_ma_32_clean --out runs\c32_relu_wider_s%%S --head linear --widths 16,32,64,80,128 --seed %%S > runs\c32_relu_wider_s%%S.log 2>&1
)
echo done. Send me the 4 .log files.
