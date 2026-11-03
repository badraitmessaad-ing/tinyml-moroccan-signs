# Frozen models

Each run folder has its int8 TFLite file and, when available, its Keras model and `config.json`. `SHA256SUMS.txt` holds the hashes reported in the paper.

| Folder | What | Used in |
|---|---|---|
| `c48_relu40k_s{42,1,2}_maR` | Final 48 x 48 models (TS-1M, then Kaggle + own crops + synthetic speed + synthetic rare signs) | Development and final test, ESP32 |
| `c48_relu40k_s{42,1,2}` | Same network trained on TS-1M only | On-device prototype tests |
| `c32_relu_wider` | 31 K-weight 32 x 32 model (70 TS-1M classes, RGB) | Latency on 16 STM32 boards |

`model_int8_mcu.tflite` is the same network with the final softmax removed for TFLite Micro.

The models were trained on TS-1M and GTSRB and fine-tuned on the Kaggle Moroccan set (MIT). Check the TS-1M terms of use before any commercial use.
