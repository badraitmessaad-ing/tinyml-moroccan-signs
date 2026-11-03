# Data

This repository contains **no images**. It contains what is needed to rebuild or request every set used in the paper.

| Set | Use in the paper | Where to get it | In this repo |
|---|---|---|---|
| TS-1M [33] | Pre-training (70 Morocco-relevant classes) | From the TS-1M authors (arXiv:2603.23034) | `ts1m_subsets/*.csv.gz`: every image in each split/class of our cleaned 32 and 48 px subsets; `ts1m_classes.csv` |
| GTSRB [30] | Comparison with published MCU models | https://benchmark.ini.rub.de (or the Kaggle copy) | rebuilt by `src/prepare_gtsrb.py` |
| Kaggle Moroccan Traffic-sign detection [43] (v1, MIT) | Fine-tuning, Moroccan photo test | https://www.kaggle.com/datasets/abdelouahabbella/moroccan-traffic-sign-detection | `kaggle_class_map.csv`: its 74 classes mapped to our 70; crops rebuilt by `src/prepare_morocco_yolo.py` |
| Own crops from 3 drives | Fine-tuning | YouTube videos in `videos.csv` | `annotations/own_train_crops.csv` |
| Development set, 4 drives (696 known + unknown) | Model selection | YouTube videos in `videos.csv` | `annotations/dev_set.csv` |
| Final test, 2 drives (423 known + 24 unknown) | Final test, run once | YouTube videos in `videos.csv` | `annotations/final_test.csv` |
| Synthetic speed and rare signs | Fine-tuning | generated | `src/make_synth_speed.py`, `src/make_synth_rare.py` |

## Annotation files

`annotations/*.csv` columns:

| Column | Meaning |
|---|---|
| `crop_file` | Crop file name |
| `class` | Final label (TS-1M class id and name; `9999_other_sign` = sign outside the 70 classes) |
| `drive` | Drive name |
| `youtube_id` | YouTube video id |
| `screenshot_file` | Screenshot file name, with the capture time |
| `x0, y0, x1, y1` | Box of the sign in that screenshot, in pixels (the crop tool adds a 10% margin) |

The screenshots were captured by hand from the videos while they played, so the box coordinates refer to the screenshot, not to a video frame. The crops therefore cannot be re-cut from the videos automatically. **The crops are available from the corresponding author on reasonable request, for research use.** The YouTube videos remain the property of their authors and are cited in the paper [54–62].
