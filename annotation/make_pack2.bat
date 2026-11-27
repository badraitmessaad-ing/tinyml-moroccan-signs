@echo off
rem REVISION: build pack 2 for the second annotator (all remaining test crops) and zip it for sending.
cd /d "%~dp0"
call conda activate tinyml 2>nul
python make_annotator_pack2.py
powershell -NoProfile -Command "Compress-Archive -Path annotator_pack2\* -DestinationPath annotator_pack2.zip -Force"
echo.
echo Send annotator_pack2.zip to the annotator. KEEP annotator_key_part2.csv private (do not send it).
pause
