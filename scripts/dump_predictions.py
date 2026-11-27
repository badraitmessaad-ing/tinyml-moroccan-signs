"""REVISION: save the per-crop outputs of the FROZEN final models (c48_relu40k_s{42,1,2}_maR) on the
development and final sets, so the new analyses asked by the reviewers (per-drive and per-sign
uncertainty, calibration/ECE, risk-coverage of rejection, safety-weighted confusions) can be computed
without touching the models. Nothing is trained or tuned.

Usage (project folder):  python dump_predictions.py [dev] [final] [dashcam] [rain_rabat] [rain_tanger] [--model ir|plain]
Writes: revision_out/preds_dev.npz, revision_out/preds_final.npz
"""
import csv, hashlib, json, os
from pathlib import Path
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import numpy as np
import tensorflow as tf
from PIL import Image
from eval_final import int8_logits, OTHER

import sys
# --model ir / plain: the matched baselines (same data, recipe and int8 conversion); default: our frozen models
MODEL = ""
if "--model" in sys.argv:
    k = sys.argv.index("--model"); MODEL = sys.argv[k + 1]; del sys.argv[k:k + 2]
RUNS = ["runs/c48_relu40k_s%s_maR" % s for s in (42, 1, 2)] if not MODEL else ["runs/b48_%s_s%s_maR" % (MODEL, s) for s in (42, 1, 2)]
SUFFIX = "_" + MODEL if MODEL else ""
SETS = {"dev": "morocco_youtube", "final": "morocco_final", "dashcam": "morocco_dashcam",
        "rain_rabat": "morocco_rain_rabat", "rain_tanger": "morocco_rain_tanger"}
if len(sys.argv) > 1: SETS = {k: v for k, v in SETS.items() if k in sys.argv[1:]}
classes = sorted(d.name for d in Path("ts1m_ma_48_clean/train").iterdir() if d.is_dir())
out = Path("revision_out"); out.mkdir(exist_ok=True)

for name, folder in SETS.items():
    test = Path(folder)
    shot_of = {}
    log = test / "crops_log.csv"
    if log.exists():
        for r in csv.DictReader(open(log, encoding="utf-8")):
            shot_of[Path(r["crop"].replace("\\", "/")).name] = r["screenshot"]
    items = []
    for d in sorted(p for p in (test / "test").iterdir() if p.is_dir()):
        for f in sorted(d.glob("*.png")):
            items.append((f, d.name, shot_of.get(f.name, "")))
    x = np.stack([np.asarray(Image.open(p).convert("RGB").resize((48, 48), Image.BILINEAR), np.float32) / 255.0
                  for p, _, _ in items])
    logits, hashes = [], []
    for r in RUNS:
        blob = (Path(r) / "model_int8.tflite").read_bytes()
        hashes.append(hashlib.sha256(blob).hexdigest())
        logits.append(int8_logits(blob, x)[:, :len(classes)])
        print(name, "done", r)
    np.savez_compressed(out / f"preds_{name}{SUFFIX}.npz",
                        files=np.array([str(p.name) for p, _, _ in items]),
                        folder=np.array([c for _, c, _ in items]),
                        screenshot=np.array([s for _, _, s in items]),
                        logits=np.stack(logits), runs=np.array(RUNS), sha256=np.array(hashes),
                        classes=np.array(classes))
    print(f"written {out / f'preds_{name}{SUFFIX}.npz'}: {len(items)} crops ({sum(c == OTHER for _, c, _ in items)} other signs)")
