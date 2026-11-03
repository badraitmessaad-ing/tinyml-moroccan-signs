"""
Pack everything the ESP32 test needs into C header files (no camera needed):

  * the int8 model with TWO outputs: the 160 features (for prototypes) and the 70 class scores
  * test images: your YouTube Moroccan crops (never used for training), already quantized to int8
  * "local photos" for on-chip learning: K crops per class, from morocco_extra\\train (your own
    crops, folder = our class name) if it exists, otherwise from the Kaggle training crops
  * prototype settings: feature mean (mu), logit scale s, alpha, tau
  * what the PC expects (accuracy and every prediction), so the chip's result can be checked

Usage (inside `conda activate tinyml`, from the project folder):
    python esp32\\esp32_pack.py --run runs\\c48_relu40k_s42
Writes esp32\\ESP32_SignTest\\*.h   -> then open esp32\\ESP32_SignTest\\ESP32_SignTest.ino in the Arduino IDE.
"""

import argparse
import csv
import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))   # project folder: train_tiny, train_baseline

import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow import keras

import train_tiny  # noqa: F401  (registers the custom layers)
from train_baseline import convert, load_split, to_numpy

SPEED_IDS = {11, 17, 23, 28, 36, 41, 46}
OTHER = "9999_other_sign"


def load_img(f, size=32):
    return np.asarray(Image.open(f).convert("RGB").resize((size, size), Image.BILINEAR), np.float32) / 255.0


def sample_dir(root, classes, per_class, rng, size=32):
    xs, ys = [], []
    for c, name in enumerate(classes):
        files = sorted(p for p in (root / name).iterdir() if p.suffix.lower() in (".png", ".jpg"))
        for k in sorted(rng.choice(len(files), min(per_class, len(files)), replace=False)):
            xs.append(load_img(files[k], size))
            ys.append(c)
    return np.stack(xs), np.array(ys)


class Dual:
    """int8 TFLite model with two outputs, run one image at a time (like the chip)."""

    def __init__(self, model_bytes, n_classes):
        self.it = tf.lite.Interpreter(model_content=model_bytes)
        self.it.allocate_tensors()
        self.inp = self.it.get_input_details()[0]
        outs = self.it.get_output_details()
        self.feat = next(o for o in outs if o["shape"][-1] != n_classes)
        self.logit = next(o for o in outs if o["shape"][-1] == n_classes)

    def quant(self, x):
        s, z = self.inp["quantization"]
        return np.clip(np.round(x / s + z), -128, 127).astype(np.int8)

    def run(self, xq):
        F, Z = [], []
        for k in range(len(xq)):
            self.it.set_tensor(self.inp["index"], xq[k:k + 1])
            self.it.invoke()
            for o, out in ((self.feat, F), (self.logit, Z)):
                s, z = o["quantization"]
                out.append((self.it.get_tensor(o["index"])[0].astype(np.float32) - z) * s)
        return np.array(F, np.float32), np.array(Z, np.float32)


def norm(f, mu):
    g = f - mu
    return g / (np.linalg.norm(g, axis=1, keepdims=True) + 1e-8)


def c_bytes(a, per_line=24):
    a = np.asarray(a).ravel()
    return ",\n".join("  " + ",".join(str(int(v)) for v in a[i:i + per_line]) for i in range(0, len(a), per_line))


def c_floats(a, per_line=8):
    a = np.asarray(a, np.float32).ravel()
    return ",\n".join("  " + ", ".join(f"{float(v):.8e}f" for v in a[i:i + per_line]) for i in range(0, len(a), per_line))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=r"runs\c32_mild40k_s42_maB")
    ap.add_argument("--data", default=None, help="default: ts1m_ma_32_clean or ts1m_ma_48_clean, from the run's size")
    ap.add_argument("--test", default="morocco_youtube")
    ap.add_argument("--adapt", default="morocco_extra", help="your own crops: <folder>/train/<our class name>/")
    ap.add_argument("--kaggle", default="morocco_crops", help="used when --adapt does not exist")
    ap.add_argument("--k", type=int, default=10, help="local photos per class for on-chip learning")
    ap.add_argument("--max-test", type=int, default=None, help="test crops in flash (default 300 at 32x32, 150 at 48x48)")
    ap.add_argument("--max-adapt", type=int, default=None, help="max local photos in flash (default 400 / 150)")
    ap.add_argument("--alpha", type=float, default=0.5, help="from proto_calib.txt")
    ap.add_argument("--tau", type=float, default=0.6, help="from proto_calib.txt")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "ESP32_SignTest"))
    args = ap.parse_args()

    import json
    run = Path(args.run)
    cfg = json.loads((run / "config.json").read_text())
    S = int(cfg.get("size", 32))
    if cfg.get("gray"):
        raise SystemExit("grayscale models are not supported by this test")
    data = Path(args.data or f"ts1m_ma_{S}_clean")
    max_test = args.max_test or (300 if S <= 32 else 150)
    max_adapt = args.max_adapt or (400 if S <= 32 else 150)
    out = Path(args.out)
    print(f"model {run.name}: input {S}x{S}, data {data}")
    out.mkdir(parents=True, exist_ok=True)
    classes = sorted(d.name for d in (data / "train").iterdir() if d.is_dir())
    n = len(classes)
    cidx = {c: i for i, c in enumerate(classes)}
    by_id = {int(c[:4]): i for i, c in enumerate(classes)}
    rng = np.random.default_rng(0)

    # 1) two-output int8 model: features (gap) + class scores
    model = keras.models.load_model(run / "model.keras", compile=False)
    dual = keras.Model(model.input, [model.get_layer("gap").output, model.output])
    val_ds, _ = load_split(data, "val", S, classes, 128, False)
    x_rep, _ = to_numpy(val_ds.take(5))
    mb = convert(dual, "int8", x_rep)
    m = Dual(mb, n)
    print(f"two-output int8 model: {len(mb) / 1024:.1f} KB")

    # 2) prototype constants from TS-1M (as in proto_calib.py): feature mean and logit scale
    x_p, _ = sample_dir(data / "train", classes, 30, rng, S)
    f_p, _ = m.run(m.quant(x_p))
    mu = f_p.mean(0)
    x_s, _ = sample_dir(data / "test", classes, 30, rng, S)
    _, z_s = m.run(m.quant(x_s))
    s_imp = float(np.median(z_s.max(1)))

    # 3) test crops (YouTube, known classes only)
    test = []
    for d in sorted(p for p in (Path(args.test) / "test").iterdir() if p.is_dir()):
        if d.name in cidx:
            test += [(f, cidx[d.name]) for f in sorted(d.glob("*.png"))]
    pick = sorted(rng.choice(len(test), min(max_test, len(test)), replace=False))
    test = [test[k] for k in pick]
    x_te = m.quant(np.stack([load_img(f, S) for f, _ in test]))
    y_te = np.array([c for _, c in test])
    alt = np.full(len(y_te), 255)
    for a, b in ((218, 221), (221, 218)):
        if a in by_id and b in by_id:
            alt[y_te == by_id[a]] = by_id[b]

    # 4) local photos for on-chip learning
    adapt, src = [], ""
    if (Path(args.adapt) / "train").exists():
        src = f"{args.adapt} (your own crops)"
        for d in sorted(p for p in (Path(args.adapt) / "train").iterdir() if p.is_dir()):
            if d.name in cidx:
                fs = sorted(d.glob("*.png"))
                adapt += [(fs[k], cidx[d.name]) for k in sorted(rng.choice(len(fs), min(args.k, len(fs)), replace=False))]
    else:
        src = f"{args.kaggle} (Kaggle train crops, classes that map to exactly one of ours)"
        for row in csv.DictReader(open(Path(args.kaggle) / "morocco_map.csv", encoding="utf-8")):
            ids = [by_id[int(x)] for x in row["ts1m_ids"].split() if int(x) in by_id]
            d = Path(args.kaggle) / "train" / row["folder"]
            if len(ids) == 1 and d.exists():
                fs = sorted(p for p in d.iterdir() if p.suffix.lower() in (".png", ".jpg"))
                if fs:
                    adapt += [(fs[k], ids[0]) for k in sorted(rng.choice(len(fs), min(args.k, len(fs)), replace=False))]
    if len(adapt) > max_adapt:
        keep = sorted(rng.choice(len(adapt), max_adapt, replace=False))
        adapt = [adapt[k] for k in keep]
    x_ad = m.quant(np.stack([load_img(f, S) for f, _ in adapt]))
    y_ad = np.array([c for _, c in adapt])
    print(f"test crops {len(test)}, local photos {len(adapt)} for {len(set(y_ad))} classes from {src}")

    # 5) what the PC expects (same float maths as the sketch)
    f_te, z_te = m.run(x_te)
    f_ad, _ = m.run(x_ad)
    g_te, g_ad = norm(f_te, mu), norm(f_ad, mu)
    P = np.zeros((n, f_te.shape[1]), np.float32)
    for c in set(y_ad):
        v = g_ad[y_ad == c].sum(0)
        P[c] = v / np.linalg.norm(v)
    has = np.array([c in set(y_ad) for c in range(n)])
    p_head = z_te.argmax(1)
    cs = g_te @ P.T
    z2 = z_te.copy()
    boost = np.where(has[None, :] & (cs > args.tau), args.alpha * s_imp * cs, -np.inf)
    z2 = np.maximum(z2, boost)
    p_proto = z2.argmax(1)
    ok = lambda p: (p == y_te) | (p == alt)
    speed = np.array([int(classes[c][:4]) in SPEED_IDS for c in y_te])
    acc = lambda p, msk=None: 100 * ok(p)[msk if msk is not None else slice(None)].mean()
    print(f"PC expects: head only {acc(p_head):.1f}% (speed limits {acc(p_head, speed):.1f}%), "
          f"with prototypes {acc(p_proto):.1f}% (speed limits {acc(p_proto, speed):.1f}%)")
    from collections import Counter
    fixed = Counter(classes[y_te[k]] for k in range(len(y_te)) if ok(p_proto)[k] and not ok(p_head)[k])
    broken = Counter((classes[y_te[k]], classes[p_proto[k]]) for k in range(len(y_te))
                     if ok(p_head)[k] and not ok(p_proto)[k])
    print(f"prototypes fixed {sum(fixed.values())} test crops and broke {sum(broken.values())}")
    for c, v in fixed.most_common(8):
        print(f"   fixed  {v:3d}  {c}")
    for (a, b), v in broken.most_common(8):
        print(f"   broken {v:3d}  {a} -> now {b}")
    print("local photos per class:", dict(Counter(classes[c][:4] for c in y_ad)))

    # 6) headers
    (out / "model_data.h").write_text(
        "// two-output int8 model (features + class scores), made by esp32_pack.py\n#pragma once\n#include <stdint.h>\n"
        f"const unsigned int g_model_len = {len(mb)};\n"
        f"alignas(16) const unsigned char g_model[] = {{\n{c_bytes(np.frombuffer(mb, np.uint8))}\n}};\n")
    (out / "test_data.h").write_text(
        "// YouTube Moroccan test crops (int8, 32x32x3), made by esp32_pack.py\n#pragma once\n#include <stdint.h>\n"
        f"#define N_TEST {len(test)}\n#define IMG_BYTES {S * S * 3}\n"
        f"const int8_t test_images[N_TEST][IMG_BYTES] = {{\n{c_bytes(x_te)}\n}};\n"
        f"const uint8_t test_labels[N_TEST] = {{\n{c_bytes(y_te)}\n}};\n"
        f"const uint8_t test_alt[N_TEST] = {{\n{c_bytes(alt)}\n}};      // 255 = none (bump: 218 or 221 both correct)\n"
        f"const uint8_t pc_pred_head[N_TEST] = {{\n{c_bytes(p_head)}\n}};\n"
        f"const uint8_t pc_pred_proto[N_TEST] = {{\n{c_bytes(p_proto)}\n}};\n")
    (out / "adapt_data.h").write_text(
        f"// local photos for on-chip learning: {src}\n#pragma once\n#include <stdint.h>\n"
        f"#define N_ADAPT {len(adapt)}\n#ifndef IMG_BYTES\n#define IMG_BYTES {S * S * 3}\n#endif\n"
        f"const int8_t adapt_images[N_ADAPT][IMG_BYTES] = {{\n{c_bytes(x_ad)}\n}};\n"
        f"const uint8_t adapt_labels[N_ADAPT] = {{\n{c_bytes(y_ad)}\n}};\n")
    (out / "proto_params.h").write_text(
        "// prototype settings and PC reference results, made by esp32_pack.py\n#pragma once\n#include <stdint.h>\n"
        f"#define N_CLASSES {n}\n#define N_FEAT {f_te.shape[1]}\n"
        f"const float PROTO_ALPHA = {args.alpha}f;\nconst float PROTO_TAU = {args.tau}f;\n"
        f"const float LOGIT_SCALE = {s_imp:.6f}f;\n"
        f"const float feat_mean[N_FEAT] = {{\n{c_floats(mu)}\n}};\n"
        f"const int16_t class_ids[N_CLASSES] = {{\n{c_bytes([int(c[:4]) for c in classes])}\n}};\n"
        f"const float PC_ACC_HEAD = {acc(p_head):.2f}f, PC_ACC_PROTO = {acc(p_proto):.2f}f;\n"
        f"const char RUN_NAME[] = \"{run.name}\";\n")
    total = (len(test) + len(adapt)) * S * S * 3 + len(mb)
    print(f"headers written to {out}  (data in flash: {total / 1e6:.2f} MB; use the 'Huge APP (3MB No OTA)' partition)")
    if total > 2.3e6:
        print("WARNING: may not fit in 3 MB with the program; run again with a smaller --max-test or --k")


if __name__ == "__main__":
    main()
