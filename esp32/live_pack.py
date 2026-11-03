"""
Make the header for the live ESP32 demo (ESP32_SignLive): the int8 model of a run, as a C array.
The PC app (sign_live.py) uses the SAME model_int8.tflite, so both sides compute the same thing.

Usage (from the project folder, inside `conda activate tinyml`):
    python esp32\\live_pack.py --run runs\\c48_relu40k_s42_maT
then open esp32\\ESP32_SignLive\\ESP32_SignLive.ino in the Arduino IDE and upload.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf


def c_bytes(a, per_line=24):
    a = np.asarray(a).ravel()
    return ",\n".join("  " + ",".join(str(int(v)) for v in a[i:i + per_line]) for i in range(0, len(a), per_line))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=r"runs\c48_relu40k_s42_maT")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "ESP32_SignLive"))
    args = ap.parse_args()
    run = Path(args.run)
    mb = (run / "model_int8.tflite").read_bytes()
    it = tf.lite.Interpreter(model_content=mb)
    it.allocate_tensors()
    inp, outp = it.get_input_details()[0], it.get_output_details()[0]
    S = int(inp["shape"][1])
    n = int(outp["shape"][-1])
    if inp["dtype"] != np.int8 or inp["shape"][-1] != 3:
        raise SystemExit("expected an int8 RGB model")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "live_model.h").write_text(
        f"// {run.name}/model_int8.tflite for the live demo, made by live_pack.py\n#pragma once\n#include <stdint.h>\n"
        f"#define IMG_SIZE {S}\n#define IMG_BYTES {S * S * 3}\n#define N_CLASSES {n}\n"
        f"const char RUN_NAME[] = \"{run.name}\";\n"
        f"const unsigned int g_model_len = {len(mb)};\n"
        f"alignas(16) const unsigned char g_model[] = {{\n{c_bytes(np.frombuffer(mb, np.uint8))}\n}};\n")
    cfg = json.loads((run / "config.json").read_text()) if (run / "config.json").exists() else {}
    print(f"{run.name}: input {S}x{S}x3 int8, {n} classes, {len(mb) / 1024:.1f} KB "
          f"(input scale {inp['quantization'][0]:.6f}, zero point {inp['quantization'][1]}); data {cfg.get('data', '?')}")
    print(f"written {out / 'live_model.h'}; now upload {out / 'ESP32_SignLive.ino'} with the Arduino IDE")


if __name__ == "__main__":
    main()
