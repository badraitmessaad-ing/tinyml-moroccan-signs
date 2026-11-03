"""
Build the dataset release of the Moroccan video crops (MRSC-V), from the project folder.

Two packages are written to <out>:
  MRSC-V_v1.0_public/   labels, boxes, video list, statistics, NO images  -> GitHub / open Zenodo record
  MRSC-V_v1.0_full/     the same + every crop image                       -> Zenodo RESTRICTED access
                                                                            (shared on request, research use)
and a .zip of each. Nothing in the project folder is changed.

Usage (inside `conda activate tinyml`, from the project folder):
    python build_dataset.py
    python build_dataset.py --project "C:\\...\\Trafic Signe recognition" --out dataset_release
"""
import argparse, csv, hashlib, re, shutil, zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

VERSION = "1.0"
NAME = "MRSC-V"
TITLE = "MRSC-V: Moroccan Road-Sign Crops from driving Videos"
OTHER = "9999_other_sign"
SPEED = {11, 17, 23, 28, 36, 41, 46}
DUP_BITS = 6          # same threshold as finetune_morocco.py

# screenshot folder -> (drive, YouTube id, paper reference, conditions)
DRIVES = {
    "autorout fes-rabat": ("fes_rabat_motorway", "yK7JmVRO3sM", "[55]", "motorway, rain"),
    "fes_video1": ("fes_1", "tVI4_HK3c1A", "[54]", "city, day"),
    "fes_video2": ("fes_2", "vD_5qURwOiY", "[57]", "city, day"),
    "rabat_video1": ("rabat", "ETjahDd78_o", "[56]", "city, day"),
    "casa_video1": ("casablanca", "R215CDuS3A0", "[58]", "city, day"),
    "casa marrakech autorout": ("casa_marrakech_motorway", "FUjXMhdKw4Y", "[59]", "motorway, day"),
    "midelt-fes": ("midelt_fes", "mrgWXXPw5uY", "[60]", "mountain road, day"),
    "marrakech": ("marrakech", "KkdaPz2aOc0", "[61]", "city, dry day"),
    "fes to ifran": ("fes_ifrane", "pKoqc-MnVkg", "[62]", "rural/mountain road, dry day"),
}
# (project subfolder, its split folder, release split name, role)
SETS = [("morocco_extra", "train", "train", "training (3 drives)"),
        ("morocco_youtube", "test", "dev", "development / model selection (4 drives)"),
        ("morocco_final", "test", "final_test", "final test, labelled blind, evaluated once (2 drives)")]


def dhash(path):
    g = np.asarray(Image.open(path).convert("L").resize((9, 8), Image.BILINEAR), np.int16)
    return np.unpackbits(np.packbits((g[:, 1:] > g[:, :-1]).ravel()))


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def collect(project):
    rows = []
    for folder, sub, split, _ in SETS:
        log = {}
        for r in csv.DictReader(open(project / folder / "crops_log.csv", encoding="utf-8")):
            log[Path(r["crop"].replace("\\", "/")).name] = r
        for d in sorted(p for p in (project / folder / sub).iterdir() if p.is_dir()):
            for f in sorted(d.glob("*.png")):
                r = log.get(f.name, {})
                shot = Path(r.get("screenshot", "").replace("\\", "/"))
                drive, vid, ref, cond = DRIVES.get(shot.parent.name, ("unknown", "", "", ""))
                m = re.search(r"(\d{4})-(\d\d)-(\d\d) (\d{6})", f.name)
                k = re.search(r"_(\d+)\.png$", f.name)
                stamp = f"{m.group(1)}{m.group(2)}{m.group(3)}_{m.group(4)}" if m else f.stem
                new = f"{drive}_{stamp}_{k.group(1) if k else 0}.png"
                w, h = Image.open(f).size
                cid = int(d.name[:4]) if d.name[:4].isdigit() else -1
                rows.append(dict(split=split, src=f, file=f"images/{split}/{d.name}/{new}", original_name=f.name,
                                 class_folder=d.name, class_id=cid, class_name=d.name[5:],
                                 in_scope=int(d.name != OTHER), speed_limit=int(cid in SPEED), drive=drive,
                                 youtube_id=vid, paper_ref=ref, screenshot_file=shot.name,
                                 x0=r.get("x0", ""), y0=r.get("y0", ""), x1=r.get("x1", ""), y1=r.get("y1", ""),
                                 crop_w=w, crop_h=h))
    # name clashes (should not happen) -> add a counter
    seen = Counter()
    for r in rows:
        seen[r["file"]] += 1
        if seen[r["file"]] > 1:
            r["file"] = r["file"].replace(".png", f"_{seen[r['file']]}.png")
    # near-duplicates of an evaluation crop inside the training split (removed before training in the paper)
    ev = [r for r in rows if r["split"] != "train"]
    H = np.stack([dhash(r["src"]) for r in ev])
    for r in rows:
        r["near_dup_of_eval"] = 0
        if r["split"] == "train":
            r["near_dup_of_eval"] = int(int((H != dhash(r["src"])).sum(1).min()) <= DUP_BITS)
        r["sha256"] = sha(r["src"])
    return rows


COLS = ["file", "split", "class_id", "class_name", "class_folder", "in_scope", "speed_limit", "drive", "youtube_id",
        "screenshot_file", "x0", "y0", "x1", "y1", "crop_w", "crop_h", "near_dup_of_eval", "original_name", "sha256"]


def write_csv(path, rows, cols):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def stats(rows):
    splits = [s for _, _, s, _ in SETS]
    per = defaultdict(Counter)
    for r in rows:
        per[r["class_folder"]][r["split"]] += 1
    cls_rows = [dict(class_folder=c, class_id=int(c[:4]) if c[:4].isdigit() else -1, class_name=c[5:],
                     **{s: per[c][s] for s in splits}, total=sum(per[c].values())) for c in sorted(per)]
    drv = defaultdict(Counter)
    for r in rows:
        drv[(r["split"], r["drive"])]["in_scope" if r["in_scope"] else "out_of_scope"] += 1
        drv[(r["split"], r["drive"])]["screenshots"] += 0
    shots = defaultdict(set)
    for r in rows:
        shots[(r["split"], r["drive"])].add(r["screenshot_file"])
    drv_rows = [dict(split=s, drive=d, in_scope=c["in_scope"], out_of_scope=c["out_of_scope"],
                     screenshots=len(shots[(s, d)])) for (s, d), c in sorted(drv.items())]
    return cls_rows, drv_rows, splits


def readme(rows, cls_rows, drv_rows, splits, full):
    n = Counter(r["split"] for r in rows)
    ins = Counter(r["split"] for r in rows if r["in_scope"])
    ncls = Counter(r["split"] for r in rows if r["in_scope"] and r["class_folder"])
    classes_per = {s: len({r["class_folder"] for r in rows if r["split"] == s and r["in_scope"]}) for s in splits}
    dups = sum(r["near_dup_of_eval"] for r in rows)
    roles = {s: role for _, _, s, role in SETS}
    t = [f"# {TITLE} (v{VERSION})", "",
         "Crops of traffic signs cut from public driving videos filmed on Moroccan roads, labelled with the "
         "70 Morocco-relevant TS-1M classes (plus `9999_other_sign` for signs outside these classes). "
         "Made for the paper *TinyML traffic-sign classification for Moroccan roads under cross-country domain "
         "shift: local adaptation and microcontroller deployment* (B. Ait Messaad, S. Motahhir, LSATI, USMBA Fez).",
         "", "This package " + ("**contains the crop images** (restricted access, research use only)."
                               if full else "contains labels, boxes and statistics only (**no images**)."), "",
         "## Splits", "", "| Split | Role | Crops | In scope | Out of scope | Classes (in scope) |", "|---|---|---|---|---|---|"]
    for s in splits:
        t.append(f"| `{s}` | {roles[s]} | {n[s]} | {ins[s]} | {n[s] - ins[s]} | {classes_per[s]} |")
    t += ["", f"Total: {len(rows)} crops from {len(DRIVES)} drives. The three splits come from **different drives** "
          "(no drive is shared between splits).", "",
          "## Drives", "", "| Split | Drive | Screenshots | In scope | Out of scope |", "|---|---|---|---|---|"]
    for d in drv_rows:
        t.append(f"| {d['split']} | {d['drive']} | {d['screenshots']} | {d['in_scope']} | {d['out_of_scope']} |")
    t += ["", "Video links, channels and conditions: `videos.csv`. Per-class counts: `stats/class_counts.csv`.", "",
          "## How the crops were made", "",
          "1. The videos were played in a browser and screenshots were taken by hand whenever a sign was readable; "
          "for the final test every visible sign was taken, without selection.",
          "2. Each sign was boxed by hand in `crop_tool.py`, which adds a 10% margin and saves the crop at its real "
          "size (no enlarging).",
          "3. Training and development crops were labelled with the model's top-3 guesses shown; the **final test** "
          "was labelled in blind mode (no model guess shown) and scored once with frozen models.",
          "4. Labels were checked by the first author; road bump / uneven road (0218 / 0221) are treated as one "
          "answer in the paper.", "",
          "## Files", "",
          "| File | Content |", "|---|---|",
          "| `annotations/all.csv`, `annotations/<split>.csv` | one line per crop (columns below) |",
          "| `classes.csv` | the classes present in the crops (TS-1M id and name) |",
          "| `kaggle_class_map.csv` | Kaggle Moroccan set classes mapped to these classes |",
          "| `videos.csv` | the 9 drives: YouTube id, link, reference in the paper, split, conditions |",
          "| `stats/class_counts.csv`, `stats/drive_counts.csv` | counts |",
          "| `SHA256SUMS.txt` | hash of every image (checks a received copy) |"]
    if full:
        t.append("| `images/<split>/<class>/<drive>_<date>_<time>_<k>.png` | the crops |")
    t += ["", "Columns of the annotation files:", "",
          "| Column | Meaning |", "|---|---|",
          "| `file` | path of the image in the full package |",
          "| `split` | train / dev / final_test |",
          "| `class_id`, `class_name`, `class_folder` | TS-1M class (9999 = sign outside the 70 classes) |",
          "| `in_scope` | 1 if the sign is one of the 70 classes |",
          "| `speed_limit` | 1 for the 7 speed-limit classes |",
          "| `drive`, `youtube_id` | source video |",
          "| `screenshot_file` | screenshot name (contains the capture date and time) |",
          "| `x0, y0, x1, y1` | box of the sign in the screenshot, in pixels |",
          "| `crop_w`, `crop_h` | size of the saved crop |",
          f"| `near_dup_of_eval` | 1 if a training crop is a near-duplicate (dHash, <= {DUP_BITS} bits) of a dev or "
          f"final crop; these {dups} crops were removed before training in the paper |",
          "| `original_name`, `sha256` | file name in the original folders, hash of the image |", "",
          "## Recommended use", "",
          "- Train on `train` (drop `near_dup_of_eval = 1`), select models on `dev`, and report `final_test` once.",
          "- Report accuracy on in-scope crops; use out-of-scope crops for unknown-sign rejection.",
          "- Several crops can show the same physical sign (successive screenshots of one drive). For confidence "
          "intervals, resample whole screenshots or groups of nearby screenshots, not single crops.", "",
          "## Limitations", "",
          "- Daytime only; one development drive in rain; no night images.",
          "- Crops are tight boxes drawn by a person, not detector outputs.",
          "- Some classes have very few crops (see `stats/class_counts.csv`).",
          "- Boxes refer to the hand-taken screenshots, not to video frames, so the crops cannot be re-cut "
          "automatically from the videos.", "",
          "## Licence and terms", ""]
    if full:
        t += ["The images are crops of frames from public YouTube videos; the frames remain the property of their "
              "authors (listed in `videos.csv`). They are shared **for non-commercial research and teaching only**, "
              "on request, under these terms: do not redistribute the images, do not use them to identify people or "
              "vehicles, and cite the paper and the videos. The labels and boxes are CC BY 4.0.", ""]
    else:
        t += ["Labels, boxes, class lists and statistics: CC BY 4.0. The images are not included; they are available "
              "from the corresponding author on reasonable request for research use (restricted record of this "
              "dataset). The videos are the property of their authors (`videos.csv`).", ""]
    t += ["## Citation", "",
          "B. Ait Messaad, S. Motahhir, TinyML traffic-sign classification for Moroccan roads under cross-country "
          "domain shift: local adaptation and microcontroller deployment, Digital Engineering (submitted), 2026.", "",
          "## Contact", "", "Badr Ait Messaad, LSATI Laboratory, Sidi Mohamed Ben Abdellah University, Fez, Morocco, "
          "badr.aitmessaad@usmba.ac.ma", ""]
    return "\n".join(t)


TERMS = """MRSC-V v{v} - terms of use of the images

The images are crops of frames from public YouTube videos (see videos.csv). The frames remain the property of
their authors. By receiving this package you agree to:
  1. use the images for non-commercial research or teaching only;
  2. not redistribute the images, in whole or in part (point others to the dataset record instead);
  3. not use the images to identify people, vehicles or number plates;
  4. cite the paper and the source videos in any publication that uses the data.
The labels, boxes, class lists and statistics are released under CC BY 4.0
(https://creativecommons.org/licenses/by/4.0/).

Contact: Badr Ait Messaad, LSATI Laboratory, Sidi Mohamed Ben Abdellah University, Fez - badr.aitmessaad@usmba.ac.ma
"""


def build(project, out):
    rows = collect(project)
    cls_rows, drv_rows, splits = stats(rows)
    for full in (False, True):
        root = out / f"{NAME}_v{VERSION}_{'full' if full else 'public'}"
        if root.exists():
            shutil.rmtree(root)
        root.mkdir(parents=True)
        write_csv(root / "annotations" / "all.csv", rows, COLS)
        for s in splits:
            write_csv(root / "annotations" / f"{s}.csv", [r for r in rows if r["split"] == s], COLS)
        write_csv(root / "stats" / "class_counts.csv", cls_rows, list(cls_rows[0]))
        write_csv(root / "stats" / "drive_counts.csv", drv_rows, list(drv_rows[0]))
        write_csv(root / "classes.csv", [dict(class_id=c["class_id"], class_name=c["class_name"],
                                              class_folder=c["class_folder"]) for c in cls_rows],
                  ["class_id", "class_name", "class_folder"])
        use = {}
        for r in rows:
            use[r["drive"]] = r["split"]
        with open(root / "videos.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["drive", "split", "youtube_id", "url", "paper_ref", "conditions"])
            for d, vid, ref, cond in DRIVES.values():
                w.writerow([d, use.get(d, ""), vid, f"https://www.youtube.com/watch?v={vid}", ref, cond])
        kmap = project / "morocco_crops" / "morocco_map.csv"
        if kmap.exists():
            shutil.copy(kmap, root / "kaggle_class_map.csv")
        (root / "SHA256SUMS.txt").write_text("".join(f"{r['sha256']}  {r['file']}\n" for r in rows), encoding="utf-8")
        (root / "README.md").write_text(readme(rows, cls_rows, drv_rows, splits, full), encoding="utf-8")
        if full:
            (root / "TERMS_OF_USE.txt").write_text(TERMS.format(v=VERSION), encoding="utf-8")
            for r in rows:
                dst = root / r["file"]
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(r["src"], dst)
        else:
            (root / "LICENSE.txt").write_text("Labels, boxes, class lists and statistics: Creative Commons Attribution "
                                              "4.0 International (CC BY 4.0), https://creativecommons.org/licenses/by/4.0/\n",
                                              encoding="utf-8")
        z = shutil.make_archive(str(root), "zip", root.parent, root.name)
        print(f"{root.name}: {len(rows)} crops -> {z}")
    n = Counter(r["split"] for r in rows)
    print("crops per split:", dict(n), "| unknown drive:", sum(r["drive"] == "unknown" for r in rows),
          "| near-duplicates in train:", sum(r["near_dup_of_eval"] for r in rows))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default=".")
    ap.add_argument("--out", default="dataset_release")
    a = ap.parse_args()
    build(Path(a.project), Path(a.project) / a.out if not Path(a.out).is_absolute() else Path(a.out))
