# Revision code (v1.2)

Code added for the journal revision of *Microcontroller traffic-sign classification for Moroccan roads under
cross-country domain shift* (Intelligent Computing, under revision). Nothing here retrains the frozen models
(`c48_relu40k_s{42,1,2}_maR`); every script reads their saved predictions or runs them once on a held-out set.
All folders are meant to be copied into the project folder (the one that holds `runs/`, `morocco_*/`, `revision_out/`).

| Folder | What it does |
|---|---|
| `scripts/` | Score the frozen models once on the new held-out drives (`run_dashcam.bat`, `run_rain.bat`), save per-crop outputs (`dump_predictions.py`, also `--model ir/plain` for the matched baselines), duplicate check of a new drive (`check_new_drive.py`), ESP32 = PC batch test (`esp32_batch_test.py`), baselines on every set (`run_baselines_full.bat`), sensitivity with corrected labels (`run_adjudicated.bat`). |
| `analysis/` | `run_analysis.bat` runs every analysis and figure of the revision: per-drive accuracy with sign-level CIs, calibration and risk-coverage, error severity and the tau = 0.3 rule, physical-sign grouping, macro-F1 CIs and voting, GTSRB Pareto plot, camera test summary, measured ESP32 energy, digital twin at the measured rho (`twin_sim_rev.py`), class coverage, the per-drive table, the hardware figure; `baseline_compare.py` (matched baselines), `adjudicated_eval.py` (corrected labels), `export_sign_ids.py`. |
| `esp32/` | Arduino sketches: `ESP32_CamSign` (camera-in-the-loop on an AI-Thinker ESP32-CAM), `ESP32_Energy_INFER` / `ESP32_Energy_IDLE` (30-minute energy runs with a USB power meter). |
| `camera_test/` | PC side of the camera test (`camera_test.py`) and the keys of the printed signs. |
| `annotation/` | Blind second-annotator tool (packs 1 and 2), agreement scoring (Cohen's kappa), blind adjudication pack. The answer keys are private and not included. |
| `dataset/` | Builds MRSC-V v2.0/v2.1 (open: https://doi.org/10.5281/zenodo.24592672; restricted: https://doi.org/10.5281/zenodo.24593515). |

Python 3.11, TensorFlow 2.x (tflite), numpy, scikit-learn, matplotlib, Pillow; Arduino IDE with the esp32 core and
TensorFlow Lite Micro for the sketches.
