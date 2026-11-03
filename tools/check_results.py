"""
Quick check of the paper's main numbers, without images, GPU or TensorFlow (Python standard library only).

  1. the frozen models in models/ have the SHA-256 hashes given in the paper;
  2. from the saved predictions (results/recheck_*_predictions.csv) it recomputes, for the
     development and final test sets: crops per set, accuracy of each of the 3 final models and their mean,
     Wilson 95% intervals, accuracy and speed-limit accuracy per drive, and crops on which the seeds disagree;
  3. it compares them with the values reported in the paper and prints PASS / FAIL.

Usage (from the repository root):  python tools/check_results.py
"""
import csv, hashlib, math, sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = ["c48_relu40k_s42_maR", "c48_relu40k_s1_maR", "c48_relu40k_s2_maR"]
HASH_PREFIX = {"c48_relu40k_s42_maR": "a86ea97f", "c48_relu40k_s1_maR": "babd4ee3", "c48_relu40k_s2_maR": "4b0769a4"}
SAME = [{"0218", "0221"}]                      # road bump / uneven road: either answer counts as correct
SPEED = {"0011", "0017", "0023", "0028", "0036", "0041", "0046"}
OTHER = "9999_other_sign"
# values reported in the paper (accuracy in %, one decimal)
PAPER = {
    "dev":   {"n": 696, "unknown": 70, "models": [87.6, 87.1, 87.8], "mean": 87.5, "disagree": 133,
              "drives": {"autorout fes-rabat": 76.4, "fes_video1": 85.8, "fes_video2": 92.7, "rabat_video1": 91.5}},
    "final": {"n": 423, "unknown": 24, "models": [94.8, 95.3, 95.3], "mean": 95.1, "disagree": 34,
              "drives": {"marrakech": 94.6, "fes to ifran": 96.0}},
}
fails = 0


def check(label, got, want, tol=0.051):
    global fails
    ok = abs(got - want) <= tol
    fails += not ok
    print(f"  {'PASS' if ok else 'FAIL'}  {label:<38} {got:7.1f}   (paper {want})")


def correct(p, t):
    return p == t or any(p[:4] in g and t[:4] in g for g in SAME)


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * (c - h), 100 * (c + h)


print("1. Model hashes")
for m in MODELS:
    f = ROOT / "models" / m / "model_int8.tflite"
    if not f.exists():
        print(f"  SKIP  {m}: {f} not found")
        continue
    h = hashlib.sha256(f.read_bytes()).hexdigest()
    ok = h.startswith(HASH_PREFIX[m])
    fails += not ok
    print(f"  {'PASS' if ok else 'FAIL'}  {m:<38} {h[:12]}   (paper {HASH_PREFIX[m]}…)")

for name in ("dev", "final"):
    rows = list(csv.DictReader(open(ROOT / "results" / f"recheck_{name}_predictions.csv", encoding="utf-8")))
    known = [r for r in rows if r["true"] != OTHER]
    P = PAPER[name]
    print(f"\n2. {'Development' if name == 'dev' else 'Final test'} set")
    check("known-class crops", len(known), P["n"], 0)
    check("signs outside the 70 classes", len(rows) - len(known), P["unknown"], 0)
    accs = []
    for m, want in zip(MODELS, P["models"]):
        k = sum(correct(r[m], r["true"]) for r in known)
        a = 100 * k / len(known)
        accs.append(a)
        lo, hi = wilson(k, len(known))
        check(f"{m} (Wilson {lo:.1f}-{hi:.1f})", a, want)
    check("mean of the 3 models", sum(accs) / 3, P["mean"])
    dis = sum(len({correct(r[m], r["true"]) for m in MODELS}) > 1 for r in known)
    check("crops right for some seeds, wrong for others", dis, P["disagree"], 0)
    by = defaultdict(list)
    for r in known:
        by[r["drive"]].append(r)
    for d, want in P["drives"].items():
        rs = by[d]
        a = sum(100 * sum(correct(r[m], r["true"]) for r in rs) / len(rs) for m in MODELS) / 3
        sp = [r for r in rs if r["true"][:4] in SPEED]
        s = sum(100 * sum(correct(r[m], r["true"]) for r in sp) / len(sp) for m in MODELS) / 3 if sp else float("nan")
        check(f"drive {d} (n={len(rs)}, speed {s:.1f}%)", a, want)

print(f"\n{'ALL CHECKS PASSED' if fails == 0 else f'{fails} CHECK(S) FAILED'}")
sys.exit(1 if fails else 0)
