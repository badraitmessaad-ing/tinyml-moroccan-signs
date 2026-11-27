"""REVISION: prove that a new test drive shares no image with any data used before.
Compares every new crop with all crops of the training, development and final sets using a
perceptual difference hash (dHash, 64 bits) and reports the closest match for each new crop.
Usage:  python check_new_drive.py morocco_dashcam
Writes: <new>/novelty_check.txt"""
import sys
from pathlib import Path
import numpy as np
from PIL import Image

NEW = Path(sys.argv[1] if len(sys.argv) > 1 else "morocco_dashcam")
OLD = ["morocco_crops", "morocco_extra", "morocco_youtube", "morocco_final", "morocco_synth", "morocco_synth_rare",
       "morocco_dashcam", "morocco_rain_rabat", "morocco_rain_tanger"]
OLD = [r for r in OLD if Path(r).resolve() != NEW.resolve()]

def dhash(p):
    a = np.asarray(Image.open(p).convert("L").resize((9, 8), Image.BILINEAR), np.int16)
    return np.packbits((a[:, 1:] > a[:, :-1]).ravel())

def files(root):
    return [p for p in Path(root).rglob("*") if p.suffix.lower() in (".png", ".jpg", ".jpeg")]

new = files(NEW); old = [p for r in OLD if Path(r).exists() for p in files(r)]
H_old = np.stack([dhash(p) for p in old]); H_new = np.stack([dhash(p) for p in new])
bits = np.unpackbits(H_new[:, None, :] ^ H_old[None, :, :], axis=2).sum(2)   # Hamming distances
best = bits.min(1); arg = bits.argmin(1)
L = [f"new crops: {len(new)} in {NEW}; compared with {len(old)} crops in {', '.join(r for r in OLD if Path(r).exists())}",
     f"closest-match Hamming distance (0 = identical image, <= 4 = near duplicate): min {best.min()}, median {int(np.median(best))}",
     f"near duplicates (distance <= 4): {(best <= 4).sum()}   identical (0): {(best == 0).sum()}", ""]
for i in np.argsort(best)[:15]:
    L.append(f"  {best[i]:2d}  {new[i].name}  ~  {old[arg[i]]}")
t = "\n".join(L); print(t); (NEW / "novelty_check.txt").write_text(t + "\n", encoding="utf-8")
