"""
Fill the release folder with the files that are made on the PC (run ONCE before the first push):

  data/annotations/*.csv        label + box of every Moroccan video crop (from the class folders,
                                which hold the final labels, joined with crops_log.csv)
  data/ts1m_subsets/*.csv.gz    which TS-1M images are in each split/class of our subsets
  data/ts1m_classes.csv         the 70 classes
  models/<run>/...              the frozen int8 models used in the paper + SHA256SUMS.txt
  requirements-lock.txt         exact package versions (pip freeze)

No image is copied. Usage (inside `conda activate tinyml`):
    python tools\\export_release.py --project "C:\\path\\to\\Trafic Signe recognition"
"""
import argparse, csv, gzip, hashlib, shutil, subprocess, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

DRIVES = {  # screenshot folder -> (drive name, YouTube id)
    "autorout fes-rabat": ("fes_rabat_motorway", "yK7JmVRO3sM"),
    "fes_video1": ("fes_1", "tVI4_HK3c1A"),
    "fes_video2": ("fes_2", "vD_5qURwOiY"),
    "rabat_video1": ("rabat", "ETjahDd78_o"),
    "casa_video1": ("casablanca", "R215CDuS3A0"),
    "casa marrakech autorout": ("casa_marrakech_motorway", "FUjXMhdKw4Y"),
    "midelt-fes": ("midelt_fes", "mrgWXXPw5uY"),
    "marrakech": ("marrakech", "KkdaPz2aOc0"),
    "fes to ifran": ("fes_ifrane", "pKoqc-MnVkg"),
}
SETS = [("morocco_youtube", "test", "dev_set.csv"),
        ("morocco_final", "test", "final_test.csv"),
        ("morocco_extra", "train", "own_train_crops.csv")]
MODELS = {
    **{f"c48_relu40k_s{s}_maR": ["model_int8.tflite", "model_int8_mcu.tflite", "model.keras", "config.json",
                                  "finetune_results.txt"] for s in (42, 1, 2)},
    **{f"c48_relu40k_s{s}": ["model_int8.tflite", "model.keras", "config.json"] for s in (42, 1, 2)},
    "c32_relu_wider": ["model_int8.tflite", "model.keras", "config.json"],
}
SUBSETS = ["ts1m_ma_48_clean", "ts1m_ma_32_clean"]


def annotations(project):
    out_dir = REPO / "data" / "annotations"
    out_dir.mkdir(parents=True, exist_ok=True)
    for folder, sub, name in SETS:
        log = {}
        logf = project / folder / "crops_log.csv"
        for r in csv.DictReader(open(logf, encoding="utf-8")):
            log[Path(r["crop"].replace("\\", "/")).name] = r
        rows, missing = [], 0
        for d in sorted(p for p in (project / folder / sub).iterdir() if p.is_dir()):
            for f in sorted(d.glob("*.png")):
                r = log.get(f.name)
                if r is None:
                    missing += 1
                    rows.append([f.name, d.name, "", "", "", "", "", "", ""])
                    continue
                shot = Path(r["screenshot"].replace("\\", "/"))
                drive, vid = DRIVES.get(shot.parent.name, ("unknown", ""))
                rows.append([f.name, d.name, drive, vid, shot.name, r["x0"], r["y0"], r["x1"], r["y1"]])
        with open(out_dir / name, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["crop_file", "class", "drive", "youtube_id", "screenshot_file", "x0", "y0", "x1", "y1"])
            w.writerows(rows)
        print(f"{name}: {len(rows)} crops ({missing} without a log line)")


def subsets(project):
    out_dir = REPO / "data" / "ts1m_subsets"
    out_dir.mkdir(parents=True, exist_ok=True)
    for ds in SUBSETS:
        root = project / ds
        if not root.exists():
            print("skip (not found):", ds)
            continue
        n = 0
        with gzip.open(out_dir / f"{ds}.csv.gz", "wt", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["split", "class", "file"])
            for split in sorted(p for p in root.iterdir() if p.is_dir()):
                for c in sorted(p for p in split.iterdir() if p.is_dir()):
                    for f in sorted(c.iterdir()):
                        if f.suffix.lower() in (".png", ".jpg"):
                            w.writerow([split.name, c.name, f.name])
                            n += 1
        print(f"{ds}: {n} images listed")
        for extra in ("classes.csv", "summary.txt"):
            if (root / extra).exists():
                shutil.copy(root / extra, out_dir / f"{ds}_{extra}")
    first = project / SUBSETS[0] / "train"
    if first.exists():
        with open(REPO / "data" / "ts1m_classes.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["index", "class"])
            for i, c in enumerate(sorted(p.name for p in first.iterdir() if p.is_dir())):
                w.writerow([i, c])


def models(project):
    lines = []
    for run, files in MODELS.items():
        for name in files:
            src = project / "runs" / run / name
            if not src.exists():
                print("  missing:", src)
                continue
            dst = REPO / "models" / run / name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(src, dst)
            if dst.suffix in (".tflite", ".keras"):
                lines.append(f"{hashlib.sha256(dst.read_bytes()).hexdigest()}  {run}/{name}")
    (REPO / "models" / "SHA256SUMS.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def freeze():
    out = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True).stdout
    head = f"# python {sys.version.split()[0]}\n"
    (REPO / "requirements-lock.txt").write_text(head + out, encoding="utf-8")


def check():
    bad = [p for p in REPO.rglob("*") if p.is_file() and (
        p.suffix.lower() in (".png", ".jpg", ".jpeg", ".mp4") or p.name in ("test_data.h", "adapt_data.h"))]
    if bad:
        print("\nWARNING - these files must NOT be pushed (images or crop data):")
        for p in bad:
            print("  ", p)
    else:
        print("\ncheck: no images or crop data in the release folder - OK")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default=str(REPO.parents[1]), help="the working project folder")
    a = ap.parse_args()
    project = Path(a.project)
    print("project folder:", project)
    annotations(project)
    subsets(project)
    models(project)
    freeze()
    check()
