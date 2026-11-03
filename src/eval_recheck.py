"""Re-analysis of the frozen models for the reviewer (no model is changed or retrained).

Same images, same int8 models and same pre-processing as eval_final.py, so the
predictions are identical; only the statistics are new:
  1. 95% bootstrap interval PER MODEL (each seed separately), resampling whole
     screenshots, plus the Wilson binomial interval for comparison.
  2. Accuracy per drive (screenshot folder) for the final and development sets.
  3. Paired comparison of the Table 14 steps on the development set: McNemar test
     (seeds pooled) and a paired screenshot bootstrap of the accuracy difference.
  4. Every prediction saved to recheck_<name>_predictions.csv.

Usage (from the project folder):
  python src\\eval_recheck.py --test morocco_final   --name final
  python src\\eval_recheck.py --test morocco_youtube --name dev --steps
Writes recheck_<name>_report.txt
"""
import argparse, csv, json, hashlib, math, os
from collections import Counter, defaultdict
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import numpy as np
from PIL import Image
from eval_final import int8_logits, OTHER, SAME, SPEED

SEEDS = (42, 1, 2)
FINAL = "c48_relu40k_s%s_maR"
STEPS = [  # Table 14, in order (missing runs are skipped)
    ("TS-1M only, 32x32", "c32_relu40k_s%s"),
    ("+ Kaggle fine-tune, 32x32", "c32_relu40k_s%s_maB"),
    ("+ own crops, 32x32", "c32_relu40k_s%s_maS"),
    ("+ synthetic speed signs, 32x32", "c32_relu40k_s%s_maT"),
    ("TS-1M only, 48x48", "c48_relu40k_s%s"),
    ("+ Kaggle + own + synthetic, 48x48", "c48_relu40k_s%s_maT"),
    ("+ synthetic rare signs (final), 48x48", FINAL),
]


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * (c - h), 100 * (c + h)


def mcnemar_p(b, c):
    """Exact two-sided McNemar p-value (binomial on the discordant pairs)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--data", default="ts1m_ma_48_clean")
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--steps", action="store_true", help="also compare the Table 14 steps (paired tests)")
    ap.add_argument("--boot", type=int, default=2000)
    a = ap.parse_args()

    classes = sorted(d.name for d in (Path(a.data) / "train").iterdir() if d.is_dir())
    cidx = {c: i for i, c in enumerate(classes)}
    ids = np.array([int(c[:4]) for c in classes])
    test = Path(a.test)
    shot_of = {}
    for r in csv.DictReader(open(test / "crops_log.csv", encoding="utf-8")):
        shot_of[Path(r["crop"].replace("\\", "/")).name] = r["screenshot"].replace("\\", "/")
    items = []
    for d in sorted(p for p in (test / "test").iterdir() if p.is_dir()):
        if d.name == OTHER or d.name in cidx:
            for f in sorted(d.glob("*.png")):
                items.append((f, d.name, shot_of.get(f.name, f.name)))
    known = np.array([k for k, it in enumerate(items) if it[1] != OTHER])
    y = np.array([cidx.get(it[1], -1) for it in items])
    shots = np.array([it[2] for it in items])
    drive = np.array([s.split("/")[-2] if "/" in s else "?" for s in shots])

    def ok_vec(pred):
        return np.array([(pred[k] == y[k]) or any(ids[pred[k]] in g and ids[y[k]] in g for g in SAME)
                         if pred[k] < len(classes) else False for k in known])

    xcache, out = {}, []

    def run_group(pattern):
        res = []
        for s in SEEDS:
            run = Path(a.runs_dir) / (pattern % s)
            f = run / "model_int8.tflite"
            if not f.exists():
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
            z = int8_logits(blob, xcache[(size, gray)])
            res.append((run.name, hashlib.sha256(blob).hexdigest()[:12], z.argmax(1)))
        return res

    groups = defaultdict(list)
    for j, k in enumerate(known):
        groups[shots[k]].append(j)
    keys = list(groups)
    rng = np.random.default_rng(0)
    picks = [np.concatenate([groups[keys[i]] for i in rng.choice(len(keys), len(keys), replace=True)])
             for _ in range(a.boot)]

    final = run_group(FINAL)
    if not final:
        raise SystemExit("final models not found in " + a.runs_dir)
    oks = [ok_vec(p) for _, _, p in final]
    n = len(known)
    out.append(f"Re-analysis '{a.name}' on {a.test}: {n} known-class crops, {len(keys)} screenshots, "
               f"{len(items) - n} other signs. Predictions identical to eval_final.py (same models, same input).")
    out.append(f"\n1. Accuracy with 95% intervals, per model (screenshot bootstrap, {a.boot} resamples) and Wilson:")
    for (name, h, _), o in zip(final, oks):
        bs = [100 * o[p].mean() for p in picks]
        lo, hi = np.percentile(bs, [2.5, 97.5])
        wl, wh = wilson(int(o.sum()), n)
        out.append(f"   {name:24s} sha {h}  {100 * o.mean():5.1f}%  bootstrap {lo:5.1f}-{hi:5.1f}  Wilson {wl:5.1f}-{wh:5.1f}")
    m = np.mean(oks, 0)
    bs = [100 * m[p].mean() for p in picks]
    lo, hi = np.percentile(bs, [2.5, 97.5])
    out.append(f"   mean of the 3 models: {100 * m.mean():.1f}%, bootstrap of the seed-mean {lo:.1f}-{hi:.1f} "
               f"(this is the interval eval_final.py reported; it is narrower because averaging three models "
               f"smooths crops on which the seeds disagree)")
    pooled = np.concatenate(oks)
    out.append(f"   crops on which the 3 models disagree: {int(((m > 0) & (m < 1)).sum())} of {n}")

    out.append("\n2. Accuracy per drive (mean of the 3 models; Wilson interval of the seed-42 model):")
    for dname in sorted(set(drive[known])):
        sel = np.array([j for j, k in enumerate(known) if drive[k] == dname])
        sp = np.array([j for j in sel if ids[y[known[j]]] in SPEED])
        per = [100 * o[sel].mean() for o in oks]
        wl, wh = wilson(int(oks[0][sel].sum()), len(sel))
        spd = f"speed limits {np.mean([100 * o[sp].mean() for o in oks]):.1f}% (n={len(sp)})" if len(sp) else ""
        out.append(f"   {dname:22s} n={len(sel):4d}  {np.mean(per):5.1f}% ± {np.std(per):.1f}  "
                   f"(Wilson {wl:.1f}-{wh:.1f})  {spd}")

    with open(f"recheck_{a.name}_predictions.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["crop", "screenshot", "drive", "true"] + [nm for nm, _, _ in final])
        for k, it in enumerate(items):
            w.writerow([it[0].name, it[2], drive[k], it[1]] +
                       [classes[p[k]] if p[k] < len(classes) else OTHER for _, _, p in final])

    if a.steps:
        out.append("\n3. Paired comparison of consecutive steps (same crops). McNemar: seeds pooled, "
                   "exact two-sided. Difference interval: paired screenshot bootstrap of the seed-mean accuracy.")
        spd_mask = np.array([ids[y[k]] in SPEED for k in known])

        def compare(o0, o1, mask=None):
            if mask is None:
                mask = np.ones(o0.shape[1], bool)
            A, B = o0[:, mask], o1[:, mask]
            b = int(((A == 1) & (B == 0)).sum())   # was right, now wrong
            c = int(((A == 0) & (B == 1)).sum())   # was wrong, now right
            d = np.zeros(o0.shape[1])
            d[mask] = B.mean(0) - A.mean(0)
            ds = [100 * d[p][mask[p]].mean() for p in picks]
            dl, dh = np.percentile(ds, [2.5, 97.5])
            return (f"{100 * (B.mean() - A.mean()):+5.1f} pts (95% {dl:+.1f} to {dh:+.1f}), "
                    f"fixed {c}, broken {b}, McNemar p = {mcnemar_p(b, c):.3g}")

        prev, got = None, {}
        for label, pat in STEPS:
            g = final if pat == FINAL else run_group(pat)
            if not g:
                out.append(f"   {label}: models not found, skipped")
                continue
            o = np.array([ok_vec(p) for _, _, p in g])
            got[pat] = o
            line = f"   {label:40s} {100 * o.mean():5.1f}%  ({len(g)} seeds)"
            if prev is not None and prev.shape == o.shape:
                line += "   vs previous: " + compare(prev, o)
            out.append(line)
            prev = o
        out.append("\n4. Other paired comparisons:")
        pairs = [("input size, TS-1M only: 32x32 -> 48x48", "c32_relu40k_s%s", "c48_relu40k_s%s", None),
                 ("input size, full recipe: 32x32 -> 48x48", "c32_relu40k_s%s_maT", "c48_relu40k_s%s_maT", None),
                 ("synthetic speed signs, speed limits only", "c32_relu40k_s%s_maS", "c32_relu40k_s%s_maT", spd_mask),
                 ("own crops, speed limits only", "c32_relu40k_s%s_maB", "c32_relu40k_s%s_maS", spd_mask),
                 ("whole recipe at 48x48: TS-1M only -> final", "c48_relu40k_s%s", FINAL, None)]
        for label, p0, p1, mask in pairs:
            if p0 in got and p1 in got and got[p0].shape == got[p1].shape:
                out.append(f"   {label:46s} " + compare(got[p0], got[p1], mask))

    txt = "\n".join(out)
    Path(f"recheck_{a.name}_report.txt").write_text(txt, encoding="utf-8")
    print(txt)


if __name__ == "__main__":
    main()
