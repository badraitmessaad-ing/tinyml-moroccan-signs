# TinyML traffic-sign classification for Moroccan roads

Code, frozen models and labels for the paper

> B. Ait Messaad, S. Motahhir. *TinyML traffic-sign classification for Moroccan roads under cross-country domain shift: local adaptation and microcontroller deployment.* Submitted to *Digital Engineering* (Elsevier), 2026.

LSATI Laboratory, Sidi Mohamed Ben Abdellah University, Fez, Morocco.

The networks classify **cropped** traffic signs. Detection is out of scope. They are small enough to run on a low-cost microcontroller: 40 K weights, a 65 KB int8 file, 42.8 ms on a 170 MHz Cortex-M4 and 189 ms on an ESP32.

## Main results (frozen models, mean of 3 seeds)

| Test set | Crops | Accuracy |
|---|---|---|
| GTSRB official test (27 KB model) | 12,630 | 97.05% |
| Moroccan driving videos, development set (4 drives) | 696 | 87.5% |
| Moroccan driving videos, **final test** (2 new drives, labelled blind, evaluated once) | 423 known + 24 unknown | **95.1%** |

All reports are in [`results/`](results/).

## Repository layout

```
src/            Python code (data preparation, training, evaluation, figures)
scripts/        Windows .bat files that ran each experiment, exactly as in the paper
esp32/          ESP32 Arduino sketches and the scripts that pack a model into C headers
models/         frozen int8 models used in the paper + SHA256SUMS.txt
data/           labels, boxes and file lists (no images); see data/README.md
results/        text/CSV reports behind every table of the paper
tools/          export_release.py (fills data/ and models/ from the working folder)
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
| 3.1 | 70-class Morocco-relevant TS-1M subset (32 and 48 px) | `python src\build_ts1m_subset.py --zip ts-1m-datset.zip --out ts1m_ma_32 --size 32` |
| 3.1 | Label-noise cleaning (Confident Learning) | `python src\find_label_issues.py ... --make-clean ts1m_ma_32_clean`, then `src\apply_test_review.py` |
| 3.1 | Same cleaning at 48 px | `python src\make_48_clean.py --src48 ts1m_ma_48 --clean32 ts1m_ma_32_clean --out ts1m_ma_48_clean` |
| 3.2 | GTSRB with official split | `python src\prepare_gtsrb.py --src <gtsrb> --out gtsrb_32`, `scripts\gtsrb.bat`, `scripts\gtsrb_sizes.bat` |
| 3.3 | Kaggle Moroccan crops + class map | `python src\prepare_morocco_yolo.py --src <kaggle>\data --out morocco_crops` |
| 3.4 | Crops from driving videos (blind mode for the final test) | `python src\crop_tool.py [--blind] --shots <screenshots> --out <set>` |
| 4 | Network, KAN heads and activations | `scripts\seeds.bat`, `scripts\kanact.bat`, `scripts\kanfix.bat`, `scripts\kancompress.bat` |
| 5.3 | Kaggle fine-tuning | `scripts\finetune.bat` |
| 5.4 | Synthetic speed signs / rare signs | `scripts\synth.bat`, `scripts\rare.bat` |
| 5.4 | 48 x 48 models (final recipe) | `scripts\train_48.bat`, then `scripts\rare.bat` (runs `*_maR`) |
| 5.5 | Development and final test | `scripts\eval_youtube.bat`, `scripts\eval_final.bat` |
| 5.5 | Intervals per model, per drive, paired tests | `scripts\recheck.bat`, `scripts\recheck_dev.bat`, `python src\sign_bootstrap.py` |
| 5.6 | Robustness to corruptions | `scripts\robustness.bat`, `scripts\robust_aug.bat`, `scripts\robust2.bat`, `scripts\robft.bat` |
| 5.7 | On-device prototypes (calibrated, grouped split) | `scripts\proto_calib.bat`, `scripts\expert_tests.bat` (`proto`) |
| 5.8 | Unknown signs | `scripts\novelty.bat`, `scripts\proto_other.bat` |
| 5.9 | Detector proxy (box jitter, low resolution) | `scripts\expert_tests.bat` (`jitter`) |
| 5.10 | ESP32 and STM32 deployment | `scripts\esp32_48.bat`, `python src\esp32_batch_test.py --run runs\c48_relu40k_s42_maR --port COM5`, `src\pack_for_tflm.py`, `src\tflm_check.py` |
| 5.11 | Digital-twin simulation | `python src\twin_sim.py`, `python src\twin_uncertainty.py` |
| all | Figures | `scripts\figures.bat` |

Training is seeded (seeds 42, 1, 2). GPU results can still differ slightly between machines. The frozen models in `models/` are the ones scored in the paper; check them against `models/SHA256SUMS.txt`.

### Only evaluating the frozen models

```bash
copy models\c48_relu40k_s42_maR\*  runs\c48_relu40k_s42_maR\      (same for s1 and s2)
python src\eval_final.py --test morocco_final --name final
```

This needs the crops (see `data/README.md`).

## ESP32

1. `scripts\esp32_48.bat` writes the C headers into `esp32\ESP32_SignTest\`.
2. Open `ESP32_SignTest.ino` in the Arduino IDE (TensorFlow Lite for Microcontrollers library) and upload it.
3. For the live demo and batch test, use `esp32\live_pack.py`, the `ESP32_SignLive` sketch and `src\esp32_batch_test.py`.

The generated `model_data.h`, `test_data.h` and `adapt_data.h` are not in the repository. They contain crop pixels.

## Licence

- Code: MIT (see `LICENSE`).
- Models (`models/`) are trained on TS-1M and GTSRB and fine-tuned on the Kaggle Moroccan set. Their use is subject to the terms of those datasets; see `models/README.md`.
- Labels and boxes in `data/`: CC BY 4.0.
- The video frames belong to the video authors and are not distributed.

## Citation

See `CITATION.cff`. Please cite the paper once it is published.

## Contact

Badr Ait Messaad, badr.aitmessaad@usmba.ac.ma
