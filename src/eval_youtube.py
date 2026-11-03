"""
Test the int8 models on the new Moroccan test set cut from YouTube videos (crop_tool.py output).
None of these photos was used for training or fine-tuning.

Reports for each model: accuracy and balanced accuracy on signs of our classes, per video, by
crop size, per class (main model), and rejection of signs outside our classes (energy score,
threshold keeping 95% of known signs; for models with an "other" class also argmax = other).

Usage (inside `conda activate tinyml`, from the project folder):
    python src\\eval_youtube.py --test morocco_youtube --data ts1m_ma_32_clean
(the list of runs is built in; missing runs are skipped; add --runs ... to choose)
Writes youtube_eval.txt
"""

import argparse
import csv
import json
import os
import re
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf
from PIL import Image

OTHER = "9999_other_sign"
SAME = [{218, 221}]          # bump / uneven road: either class counts as correct
DEFAULT_RUNS = (["c32_relu40k_s%s" % s for s in (42, 1, 2)] +
                ["c32_relu40k_s%s_maB" % s for s in (42, 1, 2)] +
                ["c32_mild40k_s%s" % s for s in (42, 1, 2)] +
                ["c32_mild40k_s%s_maB" % s for s in (42, 1, 2)] +
                ["c32_mild40k_s%s_other" % s for s in (42, 1, 2)] +
                ["c32_relu40k_s%s_maS" % s for s in (42, 1, 2)] +
                ["c32_mild40k_s%s_maS" % s for s in (42, 1, 2)] +
                ["c32_relu40k_s%s_maT" % s for s in (42, 1, 2)] +
                ["c48_relu40k_s%s" % s for s in (42, 1, 2)] +
                ["c48_relu40k_s%s_maT" % s for s in (42, 1, 2)] +
                ["c48_relu40k_s%s_maR" % s for s in (42, 1, 2)])
GROUPS = [("TS-1M only, standard", "c32_relu40k_s{}"),
          ("TS-1M + Kaggle Moroccan fine-tune, standard", "c32_relu40k_s{}_maB"),
          ("TS-1M only, mild robust", "c32_mild40k_s{}"),
          ("TS-1M + Kaggle Moroccan fine-tune, mild robust", "c32_mild40k_s{}_maB"),
          ("+ explicit other class, mild robust", "c32_mild40k_s{}_other"),
          ("+ Kaggle + own speed crops, standard", "c32_relu40k_s{}_maS"),
          ("+ Kaggle + own speed crops, mild robust", "c32_mild40k_s{}_maS"),
          ("+ Kaggle + own crops + synthetic speed, standard", "c32_relu40k_s{}_maT"),
          ("48x48: TS-1M only, standard", "c48_relu40k_s{}"),
          ("48x48: + Kaggle + own crops + synthetic, standard", "c48_relu40k_s{}_maT"),
          ("48x48: + synthetic rare signs too", "c48_relu40k_s{}_maR")]
RARE = {364, 178, 237, 238, 239, 326, 327}
SPEED = {11, 17, 23, 28, 36, 41, 46}


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
    ap.add_argument("--test", default="morocco_youtube")
    ap.add_argument("--data", default="ts1m_ma_32_clean")
    ap.add_argument("--runs", nargs="*", default=None)
    args = ap.parse_args()

    classes = sorted(d.name for d in (Path(args.data) / "train").iterdir() if d.is_dir())
    cidx = {c: i for i, c in enumerate(classes)}
    test = Path(args.test)
    video = {}
    log = test / "crops_log.csv"
    if log.exists():
        for r in csv.DictReader(open(log, encoding="utf-8")):
            parts = re.split(r"[\\/]", r["screenshot"])
            video[Path(r["crop"].replace("\\", "/")).name] = parts[-2] if len(parts) > 1 else "?"
    items = []
    for d in sorted(p for p in (test / "test").iterdir() if p.is_dir()):
        for f in sorted(d.glob("*.png")):
            if d.name == OTHER or d.name in cidx:
                items.append((f, d.name, video.get(f.name, "?")))
    known = [k for k, it in enumerate(items) if it[1] != OTHER]
    unk = [k for k, it in enumerate(items) if it[1] == OTHER]
    y = np.array([cidx.get(it[1], -1) for it in items])
    sizes = np.array([min(Image.open(f).size) for f, _, _ in items])
    vids = sorted(set(it[2] for it in items))
    print(f"{len(known)} known-sign crops ({len(set(y[known]))} classes), {len(unk)} other-sign crops, videos: {vids}")

    xcache = {}
    res = {}
    runs = args.runs or DEFAULT_RUNS
    for r in runs:
        run = Path("runs") / r if not Path(r).exists() else Path(r)
        if not (run / "model_int8.tflite").exists():
            print("skip (not found):", run)
            continue
        cfg = json.loads((run / "config.json").read_text()) if (run / "config.json").exists() else {}
        size, gray = int(cfg.get("size", 32)), bool(cfg.get("gray", False))
        key = (size, gray)
        if key not in xcache:
            x = np.stack([np.asarray(Image.open(f).convert("RGB").resize((size, size), Image.BILINEAR),
                                     np.float32) / 255.0 for f, _, _ in items])
            if gray:
                x = (x @ np.array([0.2989, 0.5870, 0.1140], np.float32))[..., None]
            xcache[key] = x
        z = int8_logits((run / "model_int8.tflite").read_bytes(), xcache[key])
        n = len(classes)
        pred = z.argmax(1)
        ids = np.array([int(c[:4]) for c in classes] + [9999] * (z.shape[1] - len(classes)))
        ok = np.array([pred[k] == y[k] or any(ids[pred[k]] in g and ids[y[k]] in g for g in SAME) for k in known])
        per = defaultdict(list)
        for k, o in zip(known, ok):
            per[items[k][1]].append(o)
        s = z[:, :n]
        energy = np.log(np.exp(s - s.max(1, keepdims=True)).sum(1)) + s.max(1)
        thr = np.percentile(energy[known], 5)
        out = dict(acc=100 * ok.mean(), bal=100 * np.mean([np.mean(v) for v in per.values()]), per=per,
                   vid={v: 100 * np.mean([o for k, o in zip(known, ok) if items[k][2] == v])
                        for v in vids if any(items[k][2] == v for k in known)},
                   size={b: 100 * np.mean([o for k, o in zip(known, ok) if lo <= sizes[k] < hi])
                         for b, (lo, hi) in {"<48": (0, 48), "48-96": (48, 96), "96-192": (96, 192),
                                             ">=192": (192, 10**6)}.items()
                         if any(lo <= sizes[k] < hi for k in known)},
                   speed=100 * np.mean([o for k, o in zip(known, ok) if int(items[k][1][:4]) in SPEED]),
                   nonspeed=100 * np.mean([o for k, o in zip(known, ok) if int(items[k][1][:4]) not in SPEED]),
                   auroc=100 * auroc(energy[known], energy[unk]) if unk else float("nan"),
                   rej=100 * (energy[unk] < thr).mean() if unk else float("nan"),
                   rej_other=(100 * (pred[unk] == n).mean() if z.shape[1] > n and unk else None),
                   known_other=(100 * (pred[known] == n).mean() if z.shape[1] > n else None),
                   pred=pred, okmap=dict(zip(known, ok)))
        res[run.name] = out
        print(f"{run.name:<24} acc {out['acc']:5.1f}  balanced {out['bal']:5.1f}  "
              + "  ".join(f"{v} {a:5.1f}" for v, a in out["vid"].items())
              + f"  | unknown AUROC {out['auroc']:5.1f}, rejected {out['rej']:5.1f}%")

    def ms(vals):
        vals = [v for v in vals if v is not None and not np.isnan(v)]
        return f"{np.mean(vals):5.1f} ± {np.std(vals):3.1f}" if len(vals) > 1 else (f"{vals[0]:5.1f}      " if vals else "  -        ")

    nk = len(known)
    L = [f"YouTube Moroccan test: {nk} crops of {len(set(y[known]))} of our classes + {len(unk)} signs outside "
         f"our classes, from {len(vids)} videos ({', '.join(f'{v}: {sum(1 for k in known if items[k][2] == v)}' for v in vids)}).",
         f"None used for training. int8 models. 95% CI on accuracy about ±{100*1.96*np.sqrt(0.25/max(nk,1)):.1f} pt.", "",
         f"{'model (3 seeds)':<48}{'accuracy':>14}{'balanced':>14}" + "".join(f"{v[:14]:>16}" for v in vids)
         + f"{'speed limits':>14}{'other signs':>14}{'unk. AUROC':>14}{'unk. rejected':>15}"]
    for name, pat in GROUPS:
        rs = [res[pat.format(s)] for s in (42, 1, 2) if pat.format(s) in res]
        if not rs:
            continue
        row = f"{name:<48}{ms([r['acc'] for r in rs]):>14}{ms([r['bal'] for r in rs]):>14}"
        row += "".join(f"{ms([r['vid'].get(v) for r in rs]):>16}" for v in vids)
        row += f"{ms([r['speed'] for r in rs]):>14}{ms([r['nonspeed'] for r in rs]):>14}"
        row += f"{ms([r['auroc'] for r in rs]):>14}{ms([r['rej'] for r in rs]):>15}"
        L.append(row)
        if rs[0]["rej_other"] is not None:
            L.append(f"{'   (argmax = other: unknown rejected / known rejected)':<48}"
                     f"{ms([r['rej_other'] for r in rs]):>14}{ms([r['known_other'] for r in rs]):>14}")
    L += ["", "Accuracy by crop size (px of the shorter side):"]
    for name, pat in GROUPS:
        rs = [res[pat.format(s)] for s in (42, 1, 2) if pat.format(s) in res]
        if rs:
            L.append(f"{name:<48}" + "".join(f"  {b}: {ms([r['size'].get(b) for r in rs])}" for b in rs[0]["size"]))
    # rare classes: before / after the synthetic rare signs
    cmp_pats = [p for p in ("c48_relu40k_s{}_maT", "c48_relu40k_s{}_maR") if p.format(42) in res]
    if len(cmp_pats) == 2:
        L += ["", "Rare classes (mean of seeds): " + "  vs  ".join(p.format("*") for p in cmp_pats)]
        r0 = res[cmp_pats[0].format(42)]["per"]
        for c in sorted(r0):
            if c[:4].isdigit() and int(c[:4]) in RARE:
                vals = []
                for p in cmp_pats:
                    rs = [res[p.format(s)] for s in (42, 1, 2) if p.format(s) in res]
                    vals.append(np.mean([100 * np.mean(r["per"][c]) for r in rs if c in r["per"]]))
                L.append(f"  n={len(r0[c]):3d}  {c:<36}" + "".join(f"{v:7.1f}%" for v in vals))
    main_pat = next((p for p in ("c48_relu40k_s{}_maR", "c48_relu40k_s{}_maT", "c32_relu40k_s{}_maT", "c32_relu40k_s{}_maS", "c32_mild40k_s{}_maS", "c32_relu40k_s{}_maS", "c32_mild40k_s{}_maB")
                     if p.format(42) in res), None)
    main = [res[main_pat.format(s)] for s in (42, 1, 2) if main_pat and main_pat.format(s) in res]
    if main:
        L += ["", f"Per class, {main_pat.format('*')} (mean of seeds), n = crops:"]
        rows = []
        for c in sorted(main[0]["per"], key=lambda c: -len(main[0]["per"][c])):
            rows.append(f"  {np.mean([100*np.mean(m['per'][c]) for m in main]):5.1f}%  n={len(main[0]['per'][c]):3d}  {c}")
        L += rows
        conf = defaultdict(int)
        for m in main:
            for k in known:
                if not m["okmap"][k]:
                    conf[(items[k][1], classes[m["pred"][k]] if m["pred"][k] < len(classes) else OTHER)] += 1
        L += ["", "Most common mistakes (summed over seeds):"]
        L += [f"  {v:3d}  {a} -> {b}" for (a, b), v in sorted(conf.items(), key=lambda kv: -kv[1])[:15]]
    text = "\n".join(L)
    print("\n" + text)
    Path("youtube_eval.txt").write_text(text + "\n", encoding="utf-8")
    if main:                                   # page of the mistakes of the main model (seed 42), for checking
        import base64
        m = main[0]
        cards = []
        for k in known:
            if m["okmap"][k]:
                continue
            p = m["pred"][k]
            b64 = base64.b64encode(items[k][0].read_bytes()).decode()
            cards.append(f'<div class=c><img src="data:image/png;base64,{b64}"><b>{items[k][1]}</b><br>'
                         f'predicted: {classes[p] if p < len(classes) else OTHER}<br><small>{items[k][2]} / '
                         f'{items[k][0].name}</small></div>')
        Path("youtube_errors.html").write_text(
            "<meta charset=utf-8><style>body{font-family:sans-serif}.c{display:inline-block;width:180px;margin:6px;"
            "vertical-align:top;font-size:12px}.c img{width:170px;height:170px;object-fit:contain;background:#eee}</style>"
            f"<h3>{len(cards)} mistakes, {main_pat.format(42)}</h3>" + "".join(cards), encoding="utf-8")


if __name__ == "__main__":
    main()
