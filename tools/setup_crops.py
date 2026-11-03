"""
Put the Moroccan video crops from the restricted MRSC-V record (Zenodo, doi:10.5281/zenodo.23922536)
back into the folder layout the scripts expect, check every image against its SHA-256 hash, and
rebuild each folder's crops_log.csv.

    MRSC-V images/train/<class>/...       -> morocco_extra/train/<class>/<original name>
    MRSC-V images/dev/<class>/...         -> morocco_youtube/test/<class>/<original name>
    MRSC-V images/final_test/<class>/...  -> morocco_final/test/<class>/<original name>

Usage (from the repository root, after unzipping MRSC-V_v1.0_full.zip somewhere):
    python tools/setup_crops.py --mrsc path/to/MRSC-V_v1.0_full
Then, for example:  python src/eval_final.py --test morocco_final --name final   (-> 95.1%)
The screenshots are not distributed, so tests that re-cut crops from screenshots (expert_tests.py jitter)
cannot be re-run.
"""
import argparse, csv, hashlib, shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGET = {"train": ("morocco_extra", "train"), "dev": ("morocco_youtube", "test"), "final_test": ("morocco_final", "test")}
# drive name in MRSC-V -> screenshot folder name used by the scripts (per-drive reports)
FOLDER = {"fes_rabat_motorway": "autorout fes-rabat", "fes_1": "fes_video1", "fes_2": "fes_video2", "rabat": "rabat_video1",
          "casablanca": "casa_video1", "casa_marrakech_motorway": "casa marrakech autorout", "midelt_fes": "midelt-fes",
          "marrakech": "marrakech", "fes_ifrane": "fes to ifran", "unknown": ""}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mrsc", required=True, help="unzipped MRSC-V_v1.0_full folder")
    ap.add_argument("--out", default=str(ROOT), help="where to create morocco_* (default: repository root)")
    a = ap.parse_args()
    src, out = Path(a.mrsc), Path(a.out)
    rows = list(csv.DictReader(open(src / "annotations" / "all.csv", encoding="utf-8")))
    logs, bad = {}, 0
    for r in rows:
        folder, sub = TARGET[r["split"]]
        f = src / r["file"]
        if hashlib.sha256(f.read_bytes()).hexdigest() != r["sha256"]:
            bad += 1
            print("hash mismatch:", f)
            continue
        dst = out / folder / sub / r["class_folder"] / r["original_name"]
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, dst)
        shot_dir = FOLDER.get(r["drive"], "")
        shot = "\\".join(x for x in ("screenshots", shot_dir, r["screenshot_file"]) if x)
        logs.setdefault(folder, []).append({
            "crop": f"{folder}\\{sub}\\{r['class_folder']}\\{r['original_name']}", "screenshot": shot,
            "x0": r["x0"], "y0": r["y0"], "x1": r["x1"], "y1": r["y1"], "class": r["class_folder"],
            "model_top1": "", "model_conf": "", "source": r["youtube_id"]})
    for folder, lines in logs.items():
        with open(out / folder / "crops_log.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(lines[0]))
            w.writeheader()
            w.writerows(lines)
        print(f"{folder}: {len(lines)} crops")
    print("all hashes OK" if bad == 0 else f"{bad} images failed the hash check")


if __name__ == "__main__":
    main()
