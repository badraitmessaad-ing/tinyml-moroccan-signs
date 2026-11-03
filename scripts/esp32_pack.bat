@echo off
cd /d "%~dp0.."
rem Make the ESP32 header files. Optional argument: the model run folder.
rem   run_esp32_pack.bat                         -> runs\c32_mild40k_s42      (TS-1M only: shows on-chip learning)
rem   run_esp32_pack.bat runs\c32_mild40k_s42_maB  (already fine-tuned on Kaggle photos)
call conda activate tinyml 2>nul
set RUN=%1
if "%RUN%"=="" set RUN=runs\c32_mild40k_s42
python esp32\esp32_pack.py --run %RUN%
echo.
echo Now open esp32\ESP32_SignTest\ESP32_SignTest.ino in the Arduino IDE and upload.
pause
