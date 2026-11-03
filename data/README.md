# Data

This repository contains **no images**. It contains what is needed to rebuild or request every set used in the paper.

| Set | Use in the paper | Where to get it | In this repo |
|---|---|---|---|
| TS-1M [33] (MIT) | Pre-training (70 Morocco-relevant classes) | https://www.kaggle.com/datasets/guoyangzhao/ts-1m-datset | `ts1m_subsets/*.csv.gz`: every image in each split/class of our cleaned 32 and 48 px subsets; `ts1m_classes.csv` |
| GTSRB [30] | Comparison with published MCU models only | https://benchmark.ini.rub.de (or the Kaggle copy) | rebuilt by `src/prepare_gtsrb.py` |
| Kaggle Moroccan Traffic-sign detection [43] (v1, MIT) | Fine-tuning, Moroccan photo test | https://www.kaggle.com/datasets/abdelouahabbella/moroccan-traffic-sign-detection | `kaggle_class_map.csv`: its 74 classes mapped to our 70; crops rebuilt by `src/prepare_morocco_yolo.py` |
| Own crops, 3 drives (868 crops) | Fine-tuning | YouTube videos in `videos.csv`; crops on request | `annotations/own_train_crops.csv` |
| Development set, 4 drives (696 known + 70 unknown) | Model selection | YouTube videos in `videos.csv`; crops on request | `annotations/dev_set.csv` |
| Final test, 2 drives (423 known + 24 unknown) | Final test, run once | YouTube videos in `videos.csv`; crops on request | `annotations/final_test.csv` |
| Synthetic speed and rare signs | Fine-tuning | generated | `src/make_synth_speed.py`, `src/make_synth_rare.py` |

## Annotation files

`annotations/*.csv` columns:

| Column | Meaning |
|---|---|
| `crop_file` | Crop file name |
| `class` | Final label (TS-1M class id and name; `9999_other_sign` = sign outside the 70 classes) |
| `drive` | Drive name (`unknown` for one development crop whose screenshot was not saved in a drive folder) |
| `youtube_id` | YouTube video id |
| `screenshot_file` | Screenshot file name, with the capture time |
| `x0, y0, x1, y1` | Box of the sign in that screenshot, in pixels (the crop tool adds a 10% margin) |

The labels, boxes and statistics are archived as **MRSC-V** ([doi:10.5281/zenodo.23922220](https://doi.org/10.5281/zenodo.23922220), CC BY 4.0). The crop images are in a restricted record ([doi:10.5281/zenodo.23922536](https://doi.org/10.5281/zenodo.23922536)): access on request for research use; reviewers are granted access. `tools/setup_crops.py` puts them back into the folders the scripts expect.

`ts1m_test_review.csv` holds the human review of the flagged TS-1M test images (Section 3.3 of the paper).

The screenshots were captured by hand from the videos while they played, so the box coordinates refer to the screenshot, not to a video frame. The crops therefore cannot be re-cut from the videos automatically. **The crops are available from the corresponding author on reasonable request, for research use** (badr.aitmessaad@usmba.ac.ma). The YouTube videos remain the property of their authors and are cited in the paper [54–62].
