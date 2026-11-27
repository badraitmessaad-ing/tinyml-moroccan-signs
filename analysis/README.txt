Revision analyses (Intelligent Computing, ICOMPUTING-D-26-01696)
Every number and figure used in the revision is produced ON THIS PC by run_analysis.bat.

Double-click run_analysis.bat (or run it from the tinyml conda terminal). Inputs: the saved int8 outputs of the
3 frozen final models (revision_out\preds_*.npz, from dump_predictions.py), the annotations
(dataset_release\...\annotations), camera_test\results_nearcm.csv and the USB-meter readings typed in energy_calc.py.
Nothing is retrained.

  out_1_per_drive_calibration.txt   per-drive accuracy + bootstrap CIs, ECE, selective prediction, safety rule sweep (R1.1, R1.3, R1.7)
  out_2_severity.txt                errors by driver consequence: dangerous / safe side / same action / other (R1.7)
  out_3_safety_rule.txt             the same with the speed-limit safety rule tau = 0.3 (R1.7)
  out_4_sign_grouping_sensitivity.txt  CIs when crops are grouped into physical signs, 10 to 120 s (R1.2)
  out_5_sign_ids_cis.txt            sign IDs (same drive and class, <= 10 s, <= 2 per sign) and sign-level CIs (R1.2)
  out_6_macroF1_voting_rho.txt      macro-F1 / balanced accuracy CIs, voting on repeated screenshots, empirical rho (R1.3, R1.7, R1.8)
  fig_calibration_risk_coverage.*   reliability diagrams and risk-coverage curves (R1.7)
  fig_gtsrb_pareto.*                GTSRB accuracy vs on-device size, ours vs published models (R1.4)
  out_9_camera_test.txt             camera-in-the-loop test on the ESP32-CAM (R1.5, R2)
  out_10_energy.txt                 measured ESP32 power and energy per image (R1.6)
  out_11_twin.txt + figures\fig9_twin_rev.*   digital twin at the measured rho = 0.29 (R1.8)
  out_12_class_coverage.txt         70 classes by Moroccan regulation category, signs outside the 70 classes per drive, accuracy per category (R1.9)

Results produced by other scripts in the project folder (also on this PC):
  esp32_batch_final_report.txt, esp32_batch_tangier_report.txt   ESP32 vs PC on every final/Tangier crop (esp32_batch_test.py)
  eval_dashcam_ir_report.txt, eval_dashcam_plain_report.txt      matched baselines on Tangier (run_baselines_tangier.bat)
  eval_budget_*_report.txt                                       matched-compute controls (run_budget.bat)
  agreement_report.txt                                           second annotator, Cohen's kappa (score_agreement.py)
  audit_report.txt                                               TS-1M cleaning audit (make_audit_pack.py --score)
Python packages: numpy, scikit-learn, matplotlib, pillow (all in the tinyml environment).


Update 2026-11-26: the two rain drives (morocco_rain_rabat, morocco_rain_tanger) are included in steps 2-7 and 12;
step 13 (heldout_table.py) writes the per-drive table with sign-level and screenshot-level intervals for every drive.
