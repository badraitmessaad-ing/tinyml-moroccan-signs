# Microcontroller traffic-sign classification for Moroccan roads

Code, frozen models and labels for the paper

> B. Ait Messaad, S. Motahhir. *Microcontroller traffic-sign classification for Moroccan roads under cross-country domain shift.* Submitted to *Intelligent Computing* (special issue "Sustainable AI and AI for Sustainability"), 2026.

LSATI Laboratory, Sidi Mohamed Ben Abdellah University, Fez, Morocco.

The networks classify **cropped** traffic signs. Detection is out of scope. They are small enough to run on a low-cost microcontroller: 40 K weights, a 58.7 KB int8 file (60.4 KB of flash and 24.3 KB of RAM with the ST runtime), 42.8 ms on a 170 MHz Cortex-M4 and 189 ms on an ESP32.

## Main results (frozen models, mean of 3 seeds)

| Test set | Crops | Accuracy |
|---|---|---|
| GTSRB official test (27 KB model) | 12,630 | 97.05% |
| Moroccan driving videos, development set (4 drives) | 696 | 87.5% |
| Moroccan driving videos, **final test** (2 new drives, labelled blind, evaluated once) | 423 known + 24 unknown | **95.1%** |

The final test comes from two dry daytime drives, so 95.1% is a two-drive held-out estimate (per drive: Fes–Ifrane 96.0%, Marrakech 94.6%; `results/per_drive_report.txt`).

### Matched baselines (same weights, data and recipe; added in v1.1)

| | Ours (depthwise-separable) | Plain CNN | Inverted residual (MobileNetV2-style) |
|---|---|---|---|
| Weights | 39,942 | 40,182 | 39,542 |
| Time on NUCLEO-G474RE | **42.8 ms** | 56.2 ms | 86.4 ms |
| RAM (activations + runtime) | 24.3 KB | 17.7 KB | 56.5 KB |
| Development set | 87.5% | 83.8% | 89.2% |
| Final test | **95.1%** | 92.7% | 95.0% |

All reports are in [`results/`](results/).

## Quick check (2 minutes, no images, no TensorFlow)

```bash
python tools/check_results.py
```

It checks the SHA-256 hashes of the frozen models and recomputes from the saved predictions the accuracy of each model, the Wilson intervals, the accuracy per drive and the crops on which the seeds disagree, for the development and final test sets, and compares them with the paper (`ALL CHECKS PASSED`).

## Datasets

- **MRSC-V labels and boxes** (open, CC BY 4.0): [doi:10.5281/zenodo.23922220](https://doi.org/10.5281/zenodo.23922220)
- **MRSC-V crop images** (restricted, research use, access on request; reviewers are granted access): [doi:10.5281/zenodo.23922536](https://doi.org/10.5281/zenodo.23922536)
- TS-1M (Kaggle, MIT), GTSRB and the Kaggle Moroccan set are public; see [`data/README.md`](data/README.md).

## Repository layout

```
src/            Python code (data preparation, training, evaluation, figures)
scripts/        Windows .bat files that ran each experiment, exactly as in the paper
esp32/          ESP32 Arduino sketches and the scripts that pack a model into C headers
models/         frozen int8 models used in the paper + SHA256SUMS.txt
data/           labels, boxes and file lists (no images); see data/README.md
results/        text/CSV reports behind every table of the paper
tools/          check_results.py (quick check), setup_crops.py (crops -> script folders),
                build_dataset.py and export_release.py (how the release was made)
```

## Setup

```bash
conda create -n tinyml python=3.10
conda activate tinyml
pip install -r requirements.txt          # or requirements-lock.txt for the exact versions
```

Run every command **from the repository root** (the `.bat` files change to it themselves). Paths such as `runs\...`, `ts1m_ma_48_clean\...` and `morocco_*\...` are relative to the root.

## Data

No image is distributed here. See [`data/README.md`](data/README.md) for how to get each source:
TS-1M, GTSRB, the Kaggle Moroccan set, and the crops from YouTube driving videos (labels and boxes only).

## Reproducing the paper

| Paper | What | Command |
|---|---|---|
| 2.2.1 | 70-class Morocco-relevant TS-1M subset (32 and 48 px) | `python src\build_ts1m_subset.py --zip ts-1m-datset.zip --out ts1m_ma_32 --size 32` |
| 2.2.2–2.2.3 | Label-noise cleaning (Confident Learning) | `python src\find_label_issues.py ... --make-clean ts1m_ma_32_clean`, then `src\apply_test_review.py` |
| 2.2.2 | Same cleaning at 48 px | `python src\make_48_clean.py --src48 ts1m_ma_48 --clean32 ts1m_ma_32_clean --out ts1m_ma_48_clean` |
| 3.2 | GTSRB with official split | `python src\prepare_gtsrb.py --src <gtsrb> --out gtsrb_32`, `scripts\gtsrb.bat`, `scripts\gtsrb_sizes.bat` |
| 2.2.4 | Kaggle Moroccan crops + class map | `python src\prepare_morocco_yolo.py --src <kaggle>\data --out morocco_crops` |
| 2.2.5–2.2.7 | Crops from driving videos (blind mode for the final test) | `python src\crop_tool.py [--blind] --shots <screenshots> --out <set>` |
| 2.3.1 | Network (and the KAN heads and activations that were tried) | `scripts\seeds.bat`, `scripts\kanact.bat`, `scripts\kanfix.bat`, `scripts\kancompress.bat` |
| 3.4 | Kaggle fine-tuning | `scripts\finetune.bat` |
| 2.2.6, 3.8 | Synthetic speed signs / rare signs | `scripts\synth.bat`, `scripts\rare.bat` |
| 3.8 | 48 x 48 models (final recipe) | `scripts\train_48.bat`, then `scripts\rare.bat` (runs `*_maR`) |
| 3.7, 3.9 | Development and final test | `scripts\eval_youtube.bat`, `scripts\eval_final.bat` |
| 3.8, 3.9 | Intervals per model, paired effect sizes | `scripts\recheck.bat`, `scripts\recheck_dev.bat`, `python src\sign_bootstrap.py` |
| 3.9, Table E.3 | Results per drive | `python src\per_drive.py` (reads `results/recheck_*_predictions.csv`) |
| 3.9, Table S11 | Matched baselines (plain CNN, inverted residual) | `scripts\baselines.bat` (about 12–18 h on CPU) |
| 3.5 | Robustness to corruptions | `scripts\robustness.bat`, `scripts\robust_aug.bat`, `scripts\robust2.bat`, `scripts\robft.bat` |
| 3.6 | On-device prototypes (calibrated, grouped split) | `scripts\proto_calib.bat`, `scripts\expert_tests.bat` (`proto`) |
| 3.6 | Unknown signs | `scripts\novelty.bat`, `scripts\proto_other.bat` |
| 3.9 | Detector proxy (box jitter, low resolution) | `scripts\expert_tests.bat` (`jitter`) |
| 3.3 | ESP32 and STM32 deployment | `scripts\esp32_48.bat`, `python src\esp32_batch_test.py --run runs\c48_relu40k_s42_maR --port COM5`, `src\pack_for_tflm.py`, `src\tflm_check.py` |
| 3.10, Note F | Digital-twin simulation (conditional) | `python src\twin_sim.py`, `python src\twin_uncertainty.py` |
| all | Figures | `scripts\figures.bat` (data figures, serif font); `python src\fig_pipeline.py`, `python src\fig_network.py` (Figs. 1–2) |

Training is seeded (seeds 42, 1, 2). GPU results can still differ slightly between machines. The frozen models in `models/` are the ones scored in the paper; check them against `models/SHA256SUMS.txt`.

### Re-running the evaluation on the crop images

1. Request access to the MRSC-V images record and unzip `MRSC-V_v1.0_full.zip`.
2. `python tools/setup_crops.py --mrsc <unzipped folder>` puts the crops into `morocco_extra/`, `morocco_youtube/` and `morocco_final/` and checks every image hash.
3. Copy the frozen models to the run folders and evaluate:

```bash
copy models\c48_relu40k_s42_maR\*  runs\c48_relu40k_s42_maR\      (same for s1 and s2)
python src\eval_final.py   --test morocco_final   --name final     -> 95.1% (development set: --test morocco_youtube --name dev -> 87.5%)
python src\eval_recheck.py --test morocco_final   --name final     -> intervals per model and per drive
```

int8 inference is deterministic, so these numbers are identical on any computer. The screenshots are not distributed, so the detector-proxy test (`expert_tests.py jitter`) cannot be re-run.

### Notes on exactness

- **Training** ran on CPU (Intel Core i7-14700, 32 GB RAM, Windows 11; Python 3.11, TensorFlow 2.21, Keras 3.15, see `requirements-lock.txt`); no GPU was used. One 48 x 48 model takes about 2 h, a Moroccan fine-tuning about 13 min. CPU training with oneDNN is not bit-exact across machines: retraining gives results within the seed spread reported in the paper, not identical numbers. All reported results come from the frozen models in `models/`.
- **Synthetic signs** (`make_synth_speed.py`, `make_synth_rare.py`) draw digits and text with TrueType fonts found on the computer (Windows: Bahnschrift, Arial Bold, Calibri Bold, Verdana Bold, Tahoma Bold, Segoe UI Bold, Trebuchet Bold; Linux: DejaVu Sans, Liberation Sans). With other fonts the synthetic images differ slightly. They also reuse the own training crops (backgrounds, digit swaps), so they need the MRSC-V images.
- **Label cleaning:** `data/ts1m_test_review.csv` holds the human decisions on the flagged TS-1M test images (`label_right`, `model_right`, `unclear`) used to build the human-verified test set (`src/apply_test_review.py`).
- **STM32:** `models/*/model_int8_mcu.tflite` was uploaded to the ST Edge AI Developer Cloud (ST Edge AI Core 4.1; settings in Supplementary Note B of the paper), which compiles it and times it on real boards (3 repeats); the results are in `results/latency_measurements.csv` (baselines: `results/baselines_latency_measurements.csv`, memory reports `results/baselines_stedgeai_analyze_*.txt`). No custom operators are used.
- **ESP32:** `scripts\esp32_48.bat` writes the C headers; the sketches use the TensorFlow Lite Micro library for Arduino (ESP32 core).

## ESP32

1. `scripts\esp32_48.bat` writes the C headers into `esp32\ESP32_SignTest\`.
2. Open `ESP32_SignTest.ino` in the Arduino IDE (TensorFlow Lite for Microcontrollers library) and upload it.
3. For the live demo and batch test, use `esp32\live_pack.py`, the `ESP32_SignLive` sketch and `src\esp32_batch_test.py`.

The generated `model_data.h`, `test_data.h` and `adapt_data.h` are not in the repository. They contain crop pixels.

## Licence

- Code: MIT (see `LICENSE`).
- Models (`models/`): MIT. They are trained on TS-1M (MIT) and fine-tuned on the Kaggle Moroccan set (MIT); see `models/README.md`.
- Labels and boxes in `data/`: CC BY 4.0.
- The video frames belong to the video authors and are not distributed.

## Citation

See `CITATION.cff`. Please cite the paper once it is published.

## Contact

Badr Ait Messaad, badr.aitmessaad@usmba.ac.ma
