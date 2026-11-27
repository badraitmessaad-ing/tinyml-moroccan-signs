@echo off
rem REVISION: sensitivity analysis with corrected labels: the 21 disputed crops take the second annotator's label
rem (needs annotator_key.csv, labels_annotator2.csv, annotator_key_part2.csv, labels_annotator2_part2.csv).
cd /d "%~dp0"
call conda activate tinyml 2>nul
set PYTHONIOENCODING=utf-8
pushd revision_analysis
python adjudicated_eval.py > out_16_adjudicated.txt 2>&1
popd
type revision_analysis\out_16_adjudicated.txt
pause
