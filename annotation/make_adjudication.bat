@echo off
rem REVISION: blind adjudication pack for the crops on which the two annotators disagree (send to a third person, e.g. Saad).
cd /d "%~dp0"
call conda activate tinyml 2>nul
python make_adjudication_pack.py annotator_key.csv labels_annotator2.csv annotator_key_part2.csv labels_annotator2_part2.csv
powershell -NoProfile -Command "Compress-Archive -Path adjudication_pack\* -DestinationPath adjudication_pack.zip -Force"
echo Send adjudication_pack.zip. KEEP adjudication_key.csv private.
pause
