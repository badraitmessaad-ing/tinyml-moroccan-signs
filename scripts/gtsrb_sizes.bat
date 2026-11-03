@echo off
cd /d "%~dp0.."
rem GTSRB official split: ReLU6 at 3 sizes x color/grayscale x 3 seeds = 18 runs (about 2.5-3 hours).
rem   31k = 16,32,64,80,128   40k = 16,32,64,96,160   52k = 16,32,80,112,192
if not exist gtsrb_32\train (
  echo gtsrb_32 not found: run prepare_gtsrb.py first.
  exit /b 1
)
if not exist runs mkdir runs
for %%S in (42 1 2) do (
  for %%M in (31k 40k 52k) do (
    if "%%M"=="31k" set W=16,32,64,80,128
    if "%%M"=="40k" set W=16,32,64,96,160
    if "%%M"=="52k" set W=16,32,80,112,192
    call :one %%M %%S
  )
)
echo done. Zip the logs with:
echo   Compress-Archive -Path runs\gs_*.log -DestinationPath gtsrb_sizes.zip -Force
exit /b 0

:one
echo %1 seed %2 color ...
python src\train_tiny.py --data gtsrb_32 --out runs\gs_%1_rgb_s%2 --head linear --widths %W% --patience 15 --seed %2 > runs\gs_%1_rgb_s%2.log 2>&1
echo %1 seed %2 gray ...
python src\train_tiny.py --data gtsrb_32 --out runs\gs_%1_gray_s%2 --head linear --widths %W% --gray --patience 15 --seed %2 > runs\gs_%1_gray_s%2.log 2>&1
exit /b 0
