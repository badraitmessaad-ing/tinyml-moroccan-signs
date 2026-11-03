"""
Batch test on the ESP32: send EVERY test image to the board over USB (sketch ESP32_SignLive) and
compare with the PC, then write a report.

  esp32_batch_report.txt   overall + per class: PC accuracy, ESP32 accuracy, same answer, time
  esp32_batch_results.csv  one line per image (file, true class, PC answer, ESP32 answer, ms)

Usage (close the Arduino Serial Monitor first; the ESP32 must run ESP32_SignLive with the same model):
    python src\\esp32_batch_test.py --run runs\\c48_relu40k_s42_maR --port COM5
Images in 9999_other_sign (signs outside our classes) are sent too and counted in "same answer",
but not in accuracy. Bump 0218 / uneven road 0221 count as equivalent, as in eval_youtube.py.
"""

import argparse
import csv
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from sign_live import Board, PCModel

SAME = [{218, 221}]
EXT = (".png", ".jpg", ".jpeg", ".bmp", ".webp")


def correct(pred, true):
    if pred == true:
        return True
    a, b = int(pred[:4]), int(true[:4])
    return any(a in s and b in s for s in SAME)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=r"runs\c48_relu40k_s42_maR")
    ap.add_argument("--port", required=True, help="e.g. COM5 (see Device Manager or the app's list)")
    ap.add_argument("--test", default=r"morocco_youtube\test")
    ap.add_argument("--data", default=None, help="for the class names; default ts1m_ma_<size>_clean")
    args = ap.parse_args()

    model = PCModel(args.run)
    classes = sorted(d.name for d in (Path(args.data or f"ts1m_ma_{model.size}_clean") / "train").iterdir() if d.is_dir())
    files = sorted(f for f in Path(args.test).rglob("*") if f.suffix.lower() in EXT)
    print(f"{len(files)} images in {args.test}; model {Path(args.run).name} ({model.size}x{model.size})")

    board = Board()
    print("connecting:", board.connect(args.port))
    if Path(args.run).name not in board.info:
        print(f"WARNING: the ESP32 runs another model ({board.info}); upload the sketch made from {args.run}")

    rows = []
    t_start = time.time()
    for k, f in enumerate(files):
        true = f.parent.name
        xq = model.quantize(Image.open(f).convert("RGB"))
        pc, _ = model.predict(xq)
        try:
            esp, ms = board.predict(xq)
        except Exception as e:                                   # one retry after a reconnect
            print(f"  {f.name}: {e}; reconnecting")
            board.connect(args.port)
            esp, ms = board.predict(xq)
        rows.append(dict(file=str(f), true=true, pc=classes[pc[0][0]], pc_prob=pc[0][1],
                         esp=classes[esp[0][0]], esp_prob=esp[0][1], esp_ms=ms,
                         near_tie=min(pc[0][1] - pc[1][1], esp[0][1] - esp[1][1]) < 0.03))
        if (k + 1) % 50 == 0 or k + 1 == len(files):
            same = sum(r["pc"] == r["esp"] for r in rows)
            print(f"  {k + 1}/{len(files)}  same answer {same}/{len(rows)}  ({time.time() - t_start:.0f} s)")
    board.close()

    with open("esp32_batch_results.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    known = [r for r in rows if r["true"] in classes]
    ms = np.array([r["esp_ms"] for r in rows])
    same = [r["pc"] == r["esp"] for r in rows]
    diff = [r for r in rows if r["pc"] != r["esp"]]
    pc_ok = [correct(r["pc"], r["true"]) for r in known]
    esp_ok = [correct(r["esp"], r["true"]) for r in known]
    per = defaultdict(list)
    for r in known:
        per[r["true"]].append(r)
    L = [f"ESP32 batch test: {Path(args.run).name}, {model.size}x{model.size} int8, board: {board.info}",
         f"Test folder: {args.test}: {len(rows)} images ({len(known)} of our classes, {len(rows) - len(known)} outside them).",
         "Every image was resized and quantized on the PC, then the SAME int8 bytes were classified by the PC "
         "(TensorFlow Lite) and by the ESP32 (TensorFlow Lite Micro) over USB.", "",
         f"Same top-1 answer on PC and ESP32: {sum(same)} / {len(rows)} ({100 * np.mean(same):.1f}%)",
         f"   of the {len(diff)} differences, {sum(r['near_tie'] for r in diff)} are near ties (top-2 within 3 points: int8 rounding)",
         f"Accuracy on our classes: PC {100 * np.mean(pc_ok):.1f}%   ESP32 {100 * np.mean(esp_ok):.1f}%   (n = {len(known)})",
         f"ESP32 time per image: mean {ms.mean():.1f} ms, min {ms.min():.1f}, max {ms.max():.1f} "
         f"({1000 / ms.mean():.1f} images per second; model only, without the USB transfer)", "",
         f"{'class':<44}{'n':>5}{'PC acc':>9}{'ESP acc':>9}{'same':>8}"]
    for c in sorted(per, key=lambda c: -len(per[c])):
        rs = per[c]
        L.append(f"{c:<44}{len(rs):>5}{100 * np.mean([correct(r['pc'], c) for r in rs]):>8.1f}%"
                 f"{100 * np.mean([correct(r['esp'], c) for r in rs]):>8.1f}%"
                 f"{sum(r['pc'] == r['esp'] for r in rs):>5}/{len(rs)}")
    if diff:
        L += ["", "Images where PC and ESP32 differ:"]
        L += [f"  {Path(r['file']).parent.name}/{Path(r['file']).name}: PC {r['pc']} ({100 * r['pc_prob']:.0f}%), "
              f"ESP32 {r['esp']} ({100 * r['esp_prob']:.0f}%){'  near tie' if r['near_tie'] else ''}" for r in diff]
    text = "\n".join(L)
    print("\n" + text)
    Path("esp32_batch_report.txt").write_text(text + "\n", encoding="utf-8")
    print("\nwritten esp32_batch_report.txt and esp32_batch_results.csv")


if __name__ == "__main__":
    main()
