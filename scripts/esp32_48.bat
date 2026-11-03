@echo off
cd /d "%~dp0.."
rem ESP32 test of the 48x48 model: speed, RAM, accuracy, and on-chip learning from YOUR crops (morocco_extra).
rem Optional argument: the model run (default: the 48x48 TS-1M-only model, so on-chip learning is visible)
call conda activate tinyml 2>nul
set RUN=%1
if "%RUN%"=="" set RUN=runs\c48_relu40k_s42
python esp32\esp32_pack.py --run %RUN%
echo.
echo Now close and reopen esp32\ESP32_SignTest\ESP32_SignTest.ino in the Arduino IDE and upload.
pause
