# Results behind the paper's tables

| File | Paper |
|---|---|
| `youtube_eval.txt` | Development-set accuracy of every model, step by step (Table 14) |
| `eval_dev_report.txt`, `eval_dev_per_class.csv`, `eval_dev_confusion.csv` | Development set, per class (supplementary E) |
| `eval_final_report.txt`, `eval_final_per_class.csv`, `eval_final_confusion.csv` | Final test (run once) |
| `recheck_*_report.txt`, `recheck_*_predictions.csv` | Intervals per model, per drive, paired McNemar tests; every prediction |
| `expert_jitter.txt` | Detector proxy (box shift, scale, resolution) |
| `expert_proto.txt` | Prototypes with a grouped split by drive |
| `proto_calib.txt` | Calibrated on-device prototypes |
| `novelty.txt` | Unknown-sign rejection (AUROC, energy score) |
| `robustness.csv` | Accuracy under 17 corruptions (supplementary C) |
| `kan_compress.txt` | KAN head compression |
| `latency_measurements.csv`, `esp32_batch_report.txt`, `esp32_batch_results.csv` | Latency on the boards; ESP32 vs PC prediction for every crop |
| `twin_results.json` | Digital-twin simulation with uncertainty |
