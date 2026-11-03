"""
Test trained models on Moroccan sign crops, without retraining.

Two input layouts:
  * crops in folders named like our classes (made with crop_signs.py from your own photos):
        python src\\eval_morocco.py --test morocco_test --data ts1m_ma_32_clean runs\\c32_relu40k_s42
  * crops of a Moroccan dataset with its own classes (made with prepare_morocco_yolo.py) plus a
    mapping file "folder -> accepted TS-1M ids" (several ids = any of them counts as correct):
        python src\\eval_morocco.py --test morocco_crops --map morocco_crops\\morocco_map.csv --split test ^
            --data ts1m_ma_32_clean runs\\c32_relu40k_s42 runs\\c32_relu52k_s42
    --split all = train + val + test (fine while we do not train on these photos).

Reports, for the int8 model that goes on the MCU: accuracy, balanced accuracy over classes,
top-3, per class, by photo source (file-name prefix) and by sign size in the photo, the most
common mistakes, and the model's confidence on signs that are not in our 70 classes.
Writes morocco_eval.txt and morocco_errors.html (pictures of the mistakes).
"""

import argparse
import base64
import csv
import io
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf
from PIL import Image

import train_tiny  # noqa: F401

OTHER = "9999_other_sign"


def source_of(name):
    m = re.match(r"([A-Za-z]+)", name)
    return m.group(1).lower() if m else "?"


def int8_probs(model_bytes, x):
    it = tf.lite.Interpreter(model_content=model_bytes)
    it.allocate_tensors()
    i, o = it.get_input_details()[0], it.get_output_details()[0]
    si, zi = i["quantization"]
    so, zo = o["quantization"]
    out = []
    for k in range(len(x)):
        q = np.clip(np.round(x[k:k + 1] / si + zi), -128, 127).astype(np.int8)
        it.set_tensor(i["index"], q)
        it.invoke()
        logits = (it.get_tensor(o["index"])[0].astype(np.float32) - zo) * so
        e = np.exp(logits - logits.max())
        out.append(e / e.sum())
    return np.array(out)


def thumb(path):
    im = Image.open(path).convert("RGB")
    im.thumbnail((96, 96))
    b = io.BytesIO()
    im.save(b, format="PNG")
    return "data:image/png;base64," + base64.b64encode(b.getvalue()).decode()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--test", default="morocco_test")
    ap.add_argument("--data", required=True, help="training dataset (gives the model's class order)")
    ap.add_argument("--map", default=None, help="morocco_map.csv from prepare_morocco_yolo.py")
    ap.add_argument("--split", default="test", help="test, val, train or all")
    args = ap.parse_args()
    classes = sorted(d.name for d in (Path(args.data) / "train").iterdir() if d.is_dir())
    by_id = {int(c[:4]): i for i, c in enumerate(classes)}

    splits = ["train", "val", "test"] if args.split == "all" else [args.split]
    accept = {}                                   # folder -> set of model indices (empty = other)
    if args.map:
        for row in csv.DictReader(open(args.map, encoding="utf-8")):
            accept[row["folder"]] = {by_id[int(x)] for x in row["ts1m_ids"].split() if int(x) in by_id}
    items = []                                    # (path, folder)
    for s in splits:
        root = Path(args.test) / s
        if not root.exists():
            continue
        for d in sorted(p for p in root.iterdir() if p.is_dir()):
            if not args.map:
                accept.setdefault(d.name, {classes.index(d.name)} if d.name in classes else set())
            for f in sorted(d.iterdir()):
                if f.suffix.lower() in (".png", ".jpg", ".jpeg"):
                    items.append((f, d.name))
    if not items:
        raise SystemExit(f"no crops found in {args.test} for split {args.split}")
    sizes = np.array([min(Image.open(f).size) for f, _ in items])
    srcs = [source_of(f.name) for f, _ in items]

    L, html = [f"Moroccan test: {args.test}, split {args.split}, {len(items)} crops", ""], [
        "<!doctype html><meta charset=utf-8><title>Moroccan test errors</title>",
        "<style>body{font:14px system-ui;margin:16px}img{width:96px;height:96px;border:1px solid #ccc}"
        "div.e{display:inline-block;width:160px;vertical-align:top;font-size:11px;margin:4px}</style>"]
    for r in args.runs:
        run = Path(r)
        cfg = json.loads((run / "config.json").read_text()) if (run / "config.json").exists() else {}
        size, gray = int(cfg.get("size", 32)), bool(cfg.get("gray", False))
        x = np.stack([np.asarray(Image.open(f).convert("RGB").resize((size, size), Image.BILINEAR),
                                 np.float32) / 255.0 for f, _ in items])
        if gray:
            x = np.asarray(tf.image.rgb_to_grayscale(x))
        probs = int8_probs((run / "model_int8.tflite").read_bytes(), x)
        pred, top3 = probs.argmax(1), np.argsort(-probs, 1)[:, :3]
        known = [k for k, (_, d) in enumerate(items) if accept.get(d)]
        other = [k for k, (_, d) in enumerate(items) if not accept.get(d)]
        ok = {k: pred[k] in accept[items[k][1]] for k in known}
        ok3 = {k: any(t in accept[items[k][1]] for t in top3[k]) for k in known}
        per = defaultdict(list)
        for k in known:
            per[items[k][1]].append(ok[k])
        acc = 100 * np.mean(list(ok.values()))
        bal = 100 * np.mean([np.mean(v) for v in per.values()])
        L += [f"{run.name} ({'gray' if gray else 'RGB'}, int8): accuracy {acc:.1f}%   balanced {bal:.1f}%   "
              f"top-3 {100*np.mean(list(ok3.values())):.1f}%   on {len(known)} crops of {len(per)} classes"]
        for s in sorted(set(srcs)):
            ks = [k for k in known if srcs[k] == s]
            if ks:
                L.append(f"   source '{s}': {100*np.mean([ok[k] for k in ks]):5.1f}%  (n={len(ks)})")
        for lo, hi in ((0, 24), (24, 48), (48, 96), (96, 10**6)):
            ks = [k for k in known if lo <= sizes[k] < hi]
            if ks:
                L.append(f"   crop {lo:>3}-{hi if hi < 10**6 else '':<3} px: {100*np.mean([ok[k] for k in ks]):5.1f}%  (n={len(ks)})")
        L.append("   per class:")
        for d, v in sorted(per.items(), key=lambda kv: (np.mean(kv[1]), -len(kv[1]))):
            L.append(f"   {100*np.mean(v):5.1f}%  ({sum(v):3d}/{len(v):3d})  {d}")
        conf = Counter((items[k][1], classes[pred[k]]) for k in known if not ok[k])
        if conf:
            L += ["   most common mistakes (true -> predicted):"]
            L += [f"   {n:3d}  {a} -> {b}" for (a, b), n in conf.most_common(12)]
        if other:
            L.append(f"   {len(other)} crops of signs NOT in our classes: mean top confidence "
                     f"{100*probs[other].max(1).mean():.0f}% (signs in our classes: {100*probs[known].max(1).mean():.0f}%)")
        L.append("")
        html.append(f"<h2>{run.name}: mistakes ({len(conf and [k for k in known if not ok[k]])})</h2>")
        for k in known:
            if not ok[k]:
                html.append(f"<div class=e><img src='{thumb(items[k][0])}'><br><b>true</b> {items[k][1]}"
                            f"<br><b>model</b> {classes[pred[k]]} ({100*probs[k].max():.0f}%)</div>")
    text = "\n".join(L)
    print(text)
    Path("morocco_eval.txt").write_text(text + "\n", encoding="utf-8")
    Path("morocco_errors.html").write_text("\n".join(html), encoding="utf-8")


if __name__ == "__main__":
    main()
