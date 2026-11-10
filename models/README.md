# Frozen models

Each run folder has its int8 TFLite file and, when available, its Keras model and `config.json`. `SHA256SUMS.txt` holds the hashes reported in the paper.

| Folder | What | Used in |
|---|---|---|
| `c48_relu40k_s{42,1,2}_maR` | Final 48 x 48 models (TS-1M, then Kaggle + own crops + synthetic speed + synthetic rare signs) | Development and final test, ESP32 |
| `c48_relu40k_s{42,1,2}` | Same network trained on TS-1M only | On-device prototype tests |
| `c32_relu_wider` | 31 K-weight 32 x 32 model (70 TS-1M classes, RGB) | Latency on 16 STM32 boards |
| `b48_plain_s{42,1,2}_maR` | Matched baseline: plain CNN with standard 3 x 3 convolutions (40,182 weights), same data and recipe as the final models | Table S11 |
| `b48_ir_s{42,1,2}_maR` | Matched baseline: MobileNetV2-style inverted residual blocks, expansion 4 (39,542 weights), same data and recipe | Table S11 |

`model_int8_mcu.tflite` is the same network with tensor names stripped for the microcontroller. If ST Edge AI Core reports `Unable to parse name`, upload `model_int8.tflite` instead (same weights, tensor names kept); the baseline benchmarks used `model_int8.tflite`.

The models are released under the MIT licence, like the code. They were trained on TS-1M (Kaggle release, MIT licence) and fine-tuned on the Kaggle Moroccan Traffic-sign detection set (MIT licence) and on crops from YouTube driving videos (not redistributed). The 27 KB GTSRB comparison model is not included.
