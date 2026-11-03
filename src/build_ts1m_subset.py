"""
Build our TinyML dataset from TS-1M, reading directly from the zip (no extraction).

It keeps only sign classes that exist on Moroccan roads (Vienna-Convention signs,
listed in MOROCCO_CLASSES below), drops classes with too few training images,
caps very large classes, resizes everything to SIZE x SIZE and writes:

    <out>/train/<class>/*.png
    <out>/val/<class>/*.png      (random part of TS-1M train, for early stopping)
    <out>/test/<class>/*.png     (official TS-1M test split, untouched)
    <out>/classes.csv            id, name, group, counts per split
    <out>/all_class_counts.csv   train/test counts for ALL 454 TS-1M classes
    <out>/summary.txt

Usage (from the project folder):
    # 1) see what would be selected, without writing images (fast)
    python src\\build_ts1m_subset.py --zip "ts-1m-datset.zip" --dry-run
    # 2) build it
    python src\\build_ts1m_subset.py --zip "ts-1m-datset.zip" --out ts1m_ma_32 --size 32

Options:
    --min-train 500     drop classes with fewer training images
    --max-train 3000    cap training images per class (random sample)
    --max-test 0        cap test images per class (0 = keep all)
    --val-frac 0.1      fraction of the kept training images moved to val
    --include-other     also keep class 453 "other-sign" as a reject class
    --workers 8         parallel processes for decoding/resizing
"""

import argparse
import csv
import io
import os
import random
import re
import zipfile
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path, PurePosixPath

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".ppm", ".webp"}
TRAIN_DIR = "Core-Dataset-unify-TRAIN/"
TEST_DIR = "Core-Dataset-unify-TEST/"

# Unified TS-1M class ids of signs used on Moroccan roads, grouped by type.
MOROCCO_CLASSES = {
    "speed limit": [6, 11, 17, 23, 28, 32, 36, 39, 41, 44, 46],
    "prohibitory": [113, 114, 116, 123, 130, 131, 135, 142, 148, 150, 176, 178,
                    179, 183, 184, 190, 191],
    "priority": [132, 133, 134, 140, 141],
    "danger": [202, 203, 204, 205, 206, 208, 218, 221, 223, 231, 237, 238, 239,
               241, 242, 243, 244, 245, 252, 255, 298, 301, 302, 323],
    "mandatory": [326, 327, 328, 330, 331, 336, 337, 339, 340, 342],
    "information": [333, 364, 406, 436, 444],
}
OTHER_ID = 453

FOLDER_RE = re.compile(r"^(\d{4})-(\d+)-(.+)$")


def scan_zip(zip_path):
    """Return {split: {class_id: (folder_name, [member names])}} for core train/test."""
    zf = zipfile.ZipFile(zip_path)
    out = {"train": defaultdict(lambda: [None, []]), "test": defaultdict(lambda: [None, []])}
    for name in zf.namelist():
        if name.endswith("/"):
            continue
        if name.startswith(TRAIN_DIR):
            split = "train"
        elif name.startswith(TEST_DIR):
            split = "test"
        else:
            continue
        p = PurePosixPath(name)
        if p.suffix.lower() not in IMG_EXT or len(p.parts) < 4:
            continue
        m = FOLDER_RE.match(p.parts[2])
        if not m:
            continue
        cid = int(m.group(1))
        entry = out[split][cid]
        entry[0] = p.parts[2]
        entry[1].append(name)
    return out


def safe_name(folder):
    """'0142-142-Stop' -> '0142_Stop' (Windows-safe, sorts by id)."""
    m = FOLDER_RE.match(folder)
    cid, label = m.group(1), m.group(3)
    label = re.sub(r"[^A-Za-z0-9]+", "_", label).strip("_")
    return f"{cid}_{label}"[:60]


# ---------------------------------------------------------------- worker side
_ZF = None


def _init(zip_path):
    global _ZF
    _ZF = zipfile.ZipFile(zip_path)


def _process(job):
    member, dst, size = job
    from PIL import Image
    try:
        with Image.open(io.BytesIO(_ZF.read(member))) as im:
            im = im.convert("RGB").resize((size, size), Image.BILINEAR)
            im.save(dst)
        return True
    except Exception:
        return False


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", required=True)
    ap.add_argument("--out", default="ts1m_ma_32")
    ap.add_argument("--size", type=int, default=32)
    ap.add_argument("--min-train", type=int, default=500)
    ap.add_argument("--max-train", type=int, default=3000)
    ap.add_argument("--max-test", type=int, default=0)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--include-other", action="store_true")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    rng = random.Random(args.seed)

    print("reading zip index ...")
    data = scan_zip(args.zip)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # counts for every class (useful later)
    all_ids = sorted(set(data["train"]) | set(data["test"]))
    with open(out / "all_class_counts.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "folder", "train", "test"])
        for cid in all_ids:
            folder = (data["train"].get(cid) or data["test"].get(cid))[0]
            w.writerow([cid, folder, len(data["train"].get(cid, [None, []])[1]),
                        len(data["test"].get(cid, [None, []])[1])])

    group_of = {cid: g for g, ids in MOROCCO_CLASSES.items() for cid in ids}
    if args.include_other:
        group_of[OTHER_ID] = "other (reject)"

    kept, dropped, missing = [], [], []
    for cid in sorted(group_of):
        if cid not in data["train"]:
            missing.append(cid)
            continue
        n = len(data["train"][cid][1])
        (kept if n >= args.min_train else dropped).append(cid)

    # plan jobs
    plan = []          # (class_id, split, member, dst)
    rows = []
    for cid in kept:
        folder, train_members = data["train"][cid]
        cname = safe_name(folder)
        tr = sorted(train_members)
        rng.shuffle(tr)
        tr = tr[:args.max_train]
        n_val = int(round(len(tr) * args.val_frac))
        val, tr = tr[:n_val], tr[n_val:]
        te = sorted(data["test"].get(cid, [None, []])[1])
        if args.max_test and len(te) > args.max_test:
            rng.shuffle(te)
            te = te[:args.max_test]
        for split, members in (("train", tr), ("val", val), ("test", te)):
            for i, m in enumerate(members):
                plan.append((cid, split, m, out / split / cname / f"{i:05d}.png"))
        rows.append(dict(id=cid, name=cname, group=group_of[cid],
                         ts1m_train=len(train_members), train=len(tr), val=len(val), test=len(te)))

    # report
    L = [f"TS-1M subset for Moroccan roads   size={args.size}   min_train={args.min_train}   "
         f"max_train={args.max_train}   val_frac={args.val_frac}",
         f"candidate classes: {len(group_of)}   kept: {len(kept)}   "
         f"dropped (< {args.min_train} train): {len(dropped)}   not found: {len(missing)}",
         ""]
    if dropped:
        L.append("dropped: " + ", ".join(f"{c}({len(data['train'][c][1])})" for c in dropped))
    if missing:
        L.append("not found in zip: " + ", ".join(map(str, missing)))
    L.append("")
    L.append(f"{'id':>4}  {'class':<45} {'group':<12} {'ts1m':>7} {'train':>6} {'val':>5} {'test':>6}")
    for r in rows:
        L.append(f"{r['id']:>4}  {r['name']:<45} {r['group']:<12} {r['ts1m_train']:>7} "
                 f"{r['train']:>6} {r['val']:>5} {r['test']:>6}")
    tot = {k: sum(r[k] for r in rows) for k in ("train", "val", "test")}
    L.append("")
    L.append(f"TOTAL images  train={tot['train']}  val={tot['val']}  test={tot['test']}   "
             f"classes={len(rows)}")
    by_group = defaultdict(int)
    for r in rows:
        by_group[r["group"]] += 1
    L.append("classes per group: " + ", ".join(f"{g}={n}" for g, n in by_group.items()))
    report = "\n".join(L)
    print(report)

    with open(out / "classes.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["id"])
        w.writeheader()
        w.writerows(rows)
    (out / "summary.txt").write_text(report, encoding="utf-8")

    if args.dry_run:
        print("\n(dry run: no images written)")
        return

    # write images in parallel
    for _, split, _, dst in plan:
        dst.parent.mkdir(parents=True, exist_ok=True)
    jobs = [(m, str(dst), args.size) for _, _, m, dst in plan]
    print(f"\nwriting {len(jobs)} images with {args.workers} workers ...")
    ok = 0
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init,
                             initargs=(args.zip,)) as ex:
        for i, r in enumerate(ex.map(_process, jobs, chunksize=256), 1):
            ok += r
            if i % 10000 == 0:
                print(f"  {i}/{len(jobs)}")
    failed = len(jobs) - ok
    msg = f"done: {ok} images written, {failed} failed"
    print(msg)
    with open(out / "summary.txt", "a", encoding="utf-8") as f:
        f.write("\n" + msg + "\n")


if __name__ == "__main__":
    main()
