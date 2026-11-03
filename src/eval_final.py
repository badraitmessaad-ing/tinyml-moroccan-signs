"""
Final evaluation with per-class metrics, for the paper's review comments 1 and 2.

Use it on two sets:
  1. the 696-crop YouTube set (now called the DEVELOPMENT set, because its errors guided our changes):
        python src\\eval_final.py --test morocco_youtube --name dev
  2. the NEW final test set cut in blind mode from drives never seen before (crop_tool.py --blind):
        python src\\eval_final.py --test morocco_final --name final
     Run this only ONCE, after every model choice is frozen. Do not change anything after seeing it.

For the final models (default: c48_relu40k_s{42,1,2}_maR, 48x48 int8) it writes:
  eval_<name>_report.txt      accuracy, balanced accuracy, macro-F1 (mean +/- std over seeds),
                              95% bootstrap interval (resampling whole screenshots), speed limits,
                              per-class precision / recall / F1 / support, most common confusions,
                              and the SHA-256 of every model file (proves the models were frozen)
  eval_<name>_per_class.csv   the per-class table
  eval_<name>_confusion.csv   confusion matrix, summed over the seeds (rows = true, columns = predicted)
  eval_<name>_confusion.png   the same as a picture (classes present in the set only)
Signs in 9999_other_sign are left out of accuracy; their rejection is reported as an AUROC.
"""

import argparse
import csv
import datetime
import hashlib
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf
from PIL import Image

OTHER = "9999_other_sign"
SAME = [{218, 221}]                    # road bump / uneven road: either answer counts as correct
SPEED = {11, 17, 23, 28, 36, 41, 46}
DEFAULT_RUNS = ["runs/c48_relu40k_s%s_maR" % s for s in (42, 1, 2)]


def int8_logits(model_bytes, x):
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
        out.append((it.get_tensor(o["index"])[0].astype(np.float32) - zo) * so)
    return np.array(out)


def auroc(pos, neg):
    s = np.concatenate([pos, neg])
    r = s.argsort().argsort().astype(np.float64) + 1
    return (r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", required=True, help="folder made by crop_tool.py (has test/<class>/*.png)")
    ap.add_argument("--name", required=True, help="dev or final (used in the output file names)")
    ap.add_argument("--data", default="ts1m_ma_48_clean", help="dataset whose train folders give the class order")
    ap.add_argument("--runs", nargs="*", default=DEFAULT_RUNS)
    ap.add_argument("--boot", type=int, default=2000)
    args = ap.parse_args()

    classes = sorted(d.name for d in (Path(args.data) / "train").iterdir() if d.is_dir())
    cidx = {c: i for i, c in enumerate(classes)}
    ids = np.array([int(c[:4]) for c in classes])
    test = Path(args.test)

    shot_of = {}                                       # crop file -> screenshot (for the bootstrap)
    log = test / "crops_log.csv"
    if log.exists():
        for r in csv.DictReader(open(log, encoding="utf-8")):
            shot_of[Path(r["crop"].replace("\\", "/")).name] = r["screenshot"]
    items, skipped = [], Counter()
    for d in sorted(p for p in (test / "test").iterdir() if p.is_dir()):
        for f in sorted(d.glob("*.png")):
            if d.name == OTHER or d.name in cidx:
                items.append((f, d.name, shot_of.get(f.name, f.name)))
            else:
                skipped[d.name] += 1
    if skipped:
        print("folders not in our classes (skipped):", dict(skipped))
    known = np.array([k for k, it in enumerate(items) if it[1] != OTHER])
    unk = np.array([k for k, it in enumerate(items) if it[1] == OTHER])
    y = np.array([cidx.get(it[1], -1) for it in items])
    shots = np.array([it[2] for it in items])
    present = sorted(set(y[known]))
    print(f"{len(known)} crops of {len(present)} of our classes, {len(unk)} other signs, "
          f"{len(set(shots[known]))} screenshots")

    def correct(p, t):
        return p == t or any(ids[p] in g and ids[t] in g for g in SAME) if p < len(classes) else False

    runs, preds, energies, hashes = [], [], [], {}
    xcache = {}
    for r in args.runs:
        run = Path(r)
        f = run / "model_int8.tflite"
        if not f.exists():
            print("skip (not found):", f)
            continue
        cfg = json.loads((run / "config.json").read_text()) if (run / "config.json").exists() else {}
        size, gray = int(cfg.get("size", 48)), bool(cfg.get("gray", False))
        if (size, gray) not in xcache:
            x = np.stack([np.asarray(Image.open(p).convert("RGB").resize((size, size), Image.BILINEAR),
                                     np.float32) / 255.0 for p, _, _ in items])
            if gray:
                x = (x @ np.array([0.2989, 0.5870, 0.1140], np.float32))[..., None]
            xcache[(size, gray)] = x
        blob = f.read_bytes()
        hashes[run.name] = hashlib.sha256(blob).hexdigest()
        z = int8_logits(blob, xcache[(size, gray)])
        s = z[:, :len(classes)]
        energies.append(np.log(np.exp(s - s.max(1, keepdims=True)).sum(1)) + s.max(1))
        preds.append(z.argmax(1))
        runs.append(run.name)
        print("evaluated", run.name)
    if not runs:
        raise SystemExit("no model found")

    # ---------------------------------------------------------------- overall numbers per seed
    def scores(pred, idx):
        ok = np.array([correct(pred[k], y[k]) for k in idx])
        per = defaultdict(list)
        for k, o in zip(idx, ok):
            per[y[k]].append(o)
        sp = [o for k, o in zip(idx, ok) if ids[y[k]] in SPEED]
        return 100 * ok.mean(), 100 * np.mean([np.mean(v) for v in per.values()]), (100 * np.mean(sp) if sp else np.nan)

    def prf(pred, idx):
        tp, fp, fn = Counter(), Counter(), Counter()
        for k in idx:
            t, p = y[k], pred[k]
            if correct(p, t):
                tp[t] += 1
            else:
                fn[t] += 1
                if p < len(classes):
                    fp[p] += 1
        out = {}
        for c in present:
            pr = tp[c] / (tp[c] + fp[c]) if tp[c] + fp[c] else 0.0
            rc = tp[c] / (tp[c] + fn[c]) if tp[c] + fn[c] else 0.0
            out[c] = (pr, rc, 2 * pr * rc / (pr + rc) if pr + rc else 0.0, tp[c] + fn[c])
        return out

    acc, bal, spd, mf1 = [], [], [], []
    per_seed = []
    for pred in preds:
        a, b, s = scores(pred, known)
        t = prf(pred, known)
        acc.append(a), bal.append(b), spd.append(s), mf1.append(100 * np.mean([v[2] for v in t.values()]))
        per_seed.append(t)

    # 95% bootstrap interval, resampling whole screenshots (crops of one screenshot are correlated)
    rng = np.random.default_rng(0)
    groups = defaultdict(list)
    for k in known:
        groups[shots[k]].append(k)
    keys = list(groups)
    oks = [np.array([correct(p[k], y[k]) for k in range(len(items))]) for p in preds]
    boot = []
    for _ in range(args.boot):
        pick = rng.choice(len(keys), len(keys), replace=True)
        idx = np.concatenate([groups[keys[j]] for j in pick])
        boot.append(100 * np.mean([o[idx].mean() for o in oks]))
    lo, hi = np.percentile(boot, [2.5, 97.5])

    au = [100 * auroc(e[known], e[unk]) for e in energies] if len(unk) else []

    # ---------------------------------------------------------------- per class (mean over seeds)
    rows = []
    for c in present:
        v = np.array([t[c][:3] for t in per_seed])
        rows.append((classes[c], int(per_seed[0][c][3]), *(100 * v.mean(0))))
    rows.sort(key=lambda r: -r[1])
    conf = np.zeros((len(classes) + 1, len(classes) + 1), int)       # last row/col = other
    for pred in preds:
        for k in known:
            conf[y[k], min(pred[k], len(classes))] += 1
    mistakes = Counter()
    for pred in preds:
        for k in known:
            if not correct(pred[k], y[k]):
                mistakes[(classes[y[k]], classes[pred[k]] if pred[k] < len(classes) else "other")] += 1

    # ---------------------------------------------------------------- write
    ms = lambda v: (f"{np.nanmean(v):.1f} ± {np.nanstd(v):.1f}" if np.isfinite(v).any() else "-")
    L = [f"Evaluation '{args.name}' on {test}  ({datetime.date.today()})",
         f"{len(known)} crops of {len(present)} of our classes from {len(set(shots[known]))} screenshots; "
         f"{len(unk)} signs outside our classes",
         f"models ({len(runs)} seeds): " + ", ".join(runs), "",
         f"accuracy           {ms(acc)}   (95% bootstrap interval over screenshots: {lo:.1f}–{hi:.1f})",
         f"balanced accuracy  {ms(bal)}",
         f"macro F1           {ms(mf1)}",
         f"speed limits       {ms(spd)}",
         (f"unknown-sign AUROC {ms(au)}" if au else "unknown-sign AUROC  -  (no 'other' crops)"), "",
         "Per class, mean of seeds (precision = of the crops predicted as this class, how many are right;",
         "recall = of the crops of this class, how many are found):",
         f"{'class':<46}{'support':>8}{'precision':>11}{'recall':>9}{'F1':>8}"]
    L += [f"{c:<46}{n:>8}{p:>10.1f}%{r:>8.1f}%{f:>7.1f}%" for c, n, p, r, f in rows]
    L += ["", f"Most common confusions (summed over {len(runs)} seeds):"]
    L += [f"  {n:>4}  {t} -> {p}" for (t, p), n in mistakes.most_common(15)]
    L += ["", "SHA-256 of the model files (the same files must be used for every reported result):"]
    L += [f"  {k}  {v}" for k, v in hashes.items()]
    text = "\n".join(L)
    print("\n" + text)
    Path(f"eval_{args.name}_report.txt").write_text(text + "\n", encoding="utf-8")
    with open(f"eval_{args.name}_per_class.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["class", "support", "precision_%", "recall_%", "f1_%"])
        w.writerows([[c, n, f"{p:.1f}", f"{r:.1f}", f"{f:.1f}"] for c, n, p, r, f in rows])
    names = classes + ["other"]
    with open(f"eval_{args.name}_confusion.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["true \\ predicted"] + names)
        for i, row in enumerate(conf):
            if row.sum():
                w.writerow([names[i]] + list(row))

    try:                                                              # picture of the confusion matrix
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        cols = sorted(set(present) | {j for j in range(conf.shape[1]) if conf[present][:, j].sum()})
        m = conf[np.ix_(present, cols)].astype(float)
        m = 100 * m / m.sum(1, keepdims=True)
        lab = lambda i: re.sub(r"^\d{4}_", "", names[i]).replace("_", " ")[:26]
        fig, ax = plt.subplots(figsize=(0.28 * len(cols) + 3, 0.28 * len(present) + 2.5))
        ax.imshow(m, cmap="Blues", vmin=0, vmax=100)
        ax.set_xticks(range(len(cols)), [lab(i) for i in cols], rotation=90, fontsize=6)
        ax.set_yticks(range(len(present)), [lab(i) for i in present], fontsize=6)
        ax.set_xlabel("predicted class")
        ax.set_ylabel("true class")
        for a in range(len(present)):
            for b in range(len(cols)):
                if m[a, b] >= 5:
                    ax.text(b, a, f"{m[a, b]:.0f}", ha="center", va="center", fontsize=4.5,
                            color="white" if m[a, b] > 50 else "black")
        fig.tight_layout()
        fig.savefig(f"eval_{args.name}_confusion.png", dpi=300)
    except Exception as e:                                            # the numbers above are the result
        print("confusion picture not drawn:", e)
    print(f"\nwritten eval_{args.name}_report.txt, _per_class.csv, _confusion.csv, _confusion.png")


if __name__ == "__main__":
    main()
