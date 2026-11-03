@echo off
cd /d "%~dp0.."
rem KAN-activation test on the cleaned 70-class data. Run from the project folder
rem in the Anaconda Prompt after:  conda activate tinyml
if not exist runs mkdir runs
echo [1/3] ReLU6 reference ...
python src\train_tiny.py --data ts1m_ma_32_clean --out runs\c32_relu --head linear > runs\c32_relu.log 2>&1
echo [2/3] KAN activation in the last two blocks ...
python src\train_tiny.py --data ts1m_ma_32_clean --out runs\c32_kanact_late --head linear --act kan --act-grid 4 --act-from 3 > runs\c32_kanact_late.log 2>&1
echo [3/3] ReLU6, same size as the KAN model ...
python src\train_tiny.py --data ts1m_ma_32_clean --out runs\c32_relu_wider --head linear --widths 16,32,64,80,128 > runs\c32_relu_wider.log 2>&1
echo done. Send me the runs\c32_* folders (results.txt) and the 3 .log files.
