@echo off
cd /d "%~dp0.."
rem GTSRB official split, for the AykoNet comparison. First run once:
rem    python src\prepare_gtsrb.py --src "PATH\TO\EXTRACTED\GTSRB" --out gtsrb_32
if not exist gtsrb_32\train (
  echo gtsrb_32 not found: run prepare_gtsrb.py first ^(see the lines above in this file^).
  exit /b 1
)
if not exist runs mkdir runs
echo [1/3] train_baseline width 24 ...
python src\train_baseline.py --data gtsrb_32 --out runs\g32_w24 --width 24 --epochs 100 > runs\g32_w24.log 2>&1
echo [2/3] best tiny model (ReLU6, widths 16,32,64,80,128) ...
python src\train_tiny.py --data gtsrb_32 --out runs\g32_wider --head linear --widths 16,32,64,80,128 > runs\g32_wider.log 2>&1
echo [3/3] same model, grayscale input like AykoNet ...
python src\train_tiny.py --data gtsrb_32 --out runs\g32_wider_gray --head linear --widths 16,32,64,80,128 --gray > runs\g32_wider_gray.log 2>&1
echo done. Send me the 3 .log files.
