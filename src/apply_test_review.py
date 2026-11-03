"""
Use your review answers (test_review.csv from review.html) to build a human-verified test set,
and score a trained model on the original test and on the verified test.

Verified test = original test, except the reviewed images:
    label_right -> kept with its label
    model_right -> moved to the suggested class
    unclear     -> removed

Usage (from the project folder):
    python src\\apply_test_review.py --review test_review.csv --data ts1m_ma_32_clean
        -> writes ts1m_ma_32_clean\\test_verified\\<class>\\...
    python src\\apply_test_review.py --data ts1m_ma_32_clean --eval runs\\clean_w24=ts1m_ma_32_clean runs\\ts1m_32_w24=ts1m_ma_32
        -> balanced and plain accuracy on test and test_verified, overall and for direction signs
"""

import argparse
import csv
import json
import os
import shutil
import sys
from collections import Counter
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

DIRECTION_IDS = {114, 116, 203, 204, 205, 206, 237, 238, 239, 326, 327, 328, 330,
                 331, 333, 336, 337, 339, 340}


def cid(name):
    head = name.split("_")[0]
    return int(head) if head.isdigit() else -1


def build(review_csv, data):
    rows = list(csv.DictReader(open(review_csv, encoding="utf-8")))
    decision = {(r["class_folder"], r["file"]): r for r in rows}
    src, dst = data / "test", data / "test_verified"
    if dst.exists():
        shutil.rmtree(dst)
    counts = Counter()
    for c in sorted(d.name for d in src.iterdir() if d.is_dir()):
        (dst / c).mkdir(parents=True, exist_ok=True)
    for c in sorted(d.name for d in src.iterdir() if d.is_dir()):
        for p in (src / c).iterdir():
            r = decision.get((c, p.name))
            if r is None or r["decision"] == "label_right":
                shutil.copy2(p, dst / c / p.name)
                counts["kept" if r is None else "checked, label right"] += 1
            elif r["decision"] == "model_right":
                shutil.copy2(p, dst / r["suggested"] / f"fixed_from_{cid(c):04d}_{p.name}")
                counts["checked, moved to model's class"] += 1
            else:
                counts["checked, unclear -> removed"] += 1
    print(f"verified test written to {dst}")
    for k, v in counts.items():
        print(f"  {k}: {v}")


def evaluate(run, train_data, data, splits):
    """Score the model in `run` (trained on `train_data`) on data/<split> for each split."""
    import numpy as np
    import tensorflow as tf
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import train_tiny  # noqa: F401  registers custom layers

    cfg = json.loads((run / "config.json").read_text()) if (run / "config.json").exists() else {}
    size, gray = int(cfg.get("size", 32)), bool(cfg.get("gray", False))
    model = tf.keras.models.load_model(run / "model.keras")
    model_classes = sorted(d.name for d in (train_data / "train").iterdir() if d.is_dir())
    assert len(model_classes) == model.output_shape[-1], \
        f"{run}: model has {model.output_shape[-1]} outputs but {train_data} has {len(model_classes)} classes"
    midx = {c: i for i, c in enumerate(model_classes)}
    out = {}
    for split in splits:
        classes = sorted(d.name for d in (data / split).iterdir() if d.is_dir())
        ds = tf.keras.utils.image_dataset_from_directory(
            data / split, labels="inferred", label_mode="int", class_names=classes,
            image_size=(size, size), batch_size=512, shuffle=False)
        ds = ds.map(lambda x, y: (x / 255.0, y))
        if gray:
            ds = ds.map(lambda x, y: (tf.image.rgb_to_grayscale(x), y))
        ys, ps = [], []
        for x, y in ds:
            ps.append(np.argmax(model(x, training=False).numpy(), axis=1))
            ys.append(y.numpy())
        y, p = np.concatenate(ys), np.concatenate(ps)
        y_m = np.array([midx.get(classes[k], -1) for k in y])     # data label -> model index
        res = {}
        for subset in ("all", "direction"):
            accs, correct, total = [], 0, 0
            for k, c in enumerate(classes):
                if subset == "direction" and cid(c) not in DIRECTION_IDS:
                    continue
                m = y == k
                if m.any():
                    hit = p[m] == y_m[m]
                    accs.append(float(hit.mean()))
                    correct += int(hit.sum())
                    total += int(m.sum())
            res[subset] = (100 * float(np.mean(accs)), 100 * correct / max(total, 1), total)
        out[split] = res
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--review", help="test_review.csv exported from review.html")
    ap.add_argument("--data", required=True)
    ap.add_argument("--eval", nargs="*", default=[],
                    help="runs to score, each as RUN_FOLDER=TRAINING_DATA_FOLDER "
                         "(e.g. runs\\clean_w24=ts1m_ma_32_clean runs\\ts1m_32_w24=ts1m_ma_32); "
                         "without '=' the --data folder is assumed")
    args = ap.parse_args()
    data = Path(args.data)
    if args.review:
        build(args.review, data)
    if args.eval:
        splits = ["test"] + (["test_verified"] if (data / "test_verified").exists() else [])
        lines = [f"{'run':28} {'split':14} {'balanced':>9} {'accuracy':>9} {'dir. bal.':>10} {'dir. acc':>9}"]
        for spec in args.eval:
            r, _, td = spec.partition("=")
            res = evaluate(Path(r), Path(td) if td else data, data, splits)
            for s in splits:
                a, d = res[s]["all"], res[s]["direction"]
                lines.append(f"{Path(r).name:28} {s:14} {a[0]:8.2f}% {a[1]:8.2f}% {d[0]:9.2f}% {d[1]:8.2f}%")
        report = "\n".join(lines)
        print("\n" + report)
        (data / "verified_eval.txt").write_text(report + "\n")


if __name__ == "__main__":
    main()
