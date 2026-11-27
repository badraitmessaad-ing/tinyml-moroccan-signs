@echo off
rem REVISION: build MRSC-V v2.0 (three held-out dashcam drives, sign ids, second-annotator and corrected labels).
rem Gives dataset_release\MRSC-V_v2.0_public.zip (open, with the dashcam crop images) and
rem       dataset_release\MRSC-V_v2.0_full.zip   (restricted, all crop images).
cd /d "%~dp0"
call conda activate tinyml 2>nul
set PYTHONIOENCODING=utf-8
copy /Y revision_out\preds_*.npz revision_analysis\ >nul
pushd revision_analysis
python export_sign_ids.py
popd
python build_dataset_v2.py
echo DONE. Tell Claude.
pause
