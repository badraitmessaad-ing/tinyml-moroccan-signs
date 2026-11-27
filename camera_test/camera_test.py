"""
Camera-in-the-loop test (revision R1.5 / R2): the ESP32-CAM classifies printed signs with the on-board camera.

Before: upload esp32/ESP32_CamSign to the ESP32-CAM and CLOSE the Arduino Serial Monitor.
Run from the project folder:
    python camera_test\\camera_test.py --port COM5 --dist 20
For each sign: hold the printed sign so it fills the middle of the camera view, type its number (1-40)
and press Enter. The board takes 5 photos and classifies each one on the chip.
Type  r  to redo the last sign,  q  to finish. Then run again with --dist 40 for the second distance.

Saves:  camera_test/results_<dist>cm.csv   (one row per photo)
        camera_test/seen/<dist>cm_#NN.png   (what the model saw, 48x48 shown at 240x240)
"""
import argparse, csv, time
from collections import Counter
from pathlib import Path

import numpy as np
import serial
from PIL import Image

HERE = Path(__file__).resolve().parent
SAME = [{218, 221}]


def cid(name):
    return int(name[:4])


def same(a, b):
    return a == b or any(cid(a) in g and cid(b) in g for g in SAME)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", required=True)
    ap.add_argument("--dist", required=True, help="distance label in cm, e.g. 20")
    args = ap.parse_args()

    names = [l.split('"')[1] for l in (HERE.parent / "esp32/ESP32_CamSign/class_names.h").read_text().splitlines()
             if l.strip().startswith('"')]
    key = {int(r["number"].strip("#")): r["class_folder"] for r in csv.DictReader(open(HERE / "signs_key.csv", encoding="utf-8"))}
    (HERE / "seen").mkdir(exist_ok=True)
    out = HERE / f"results_{args.dist}cm.csv"
    rows = list(csv.DictReader(open(out, encoding="utf-8"))) if out.exists() else []
    done = {int(r["sign"]) for r in rows}
    if done:
        print(f"Continuing {out.name}: {len(done)} signs already done.")

    ser = serial.Serial()
    ser.port, ser.baudrate, ser.timeout = args.port, 921600, 5
    ser.dtr = False  # keep the ESP32-CAM-MB auto-reset lines released, so the board keeps running
    ser.rts = False
    ser.open()
    time.sleep(1.0)
    ser.reset_input_buffer()
    ser.write(b"l")  # stop the LIVE printing
    time.sleep(0.5)
    ser.reset_input_buffer()

    last = None
    while True:
        a = input(f"\n[{args.dist} cm] sign number 1-40 (r = redo last, q = quit): ").strip().lower()
        if a == "q":
            break
        if a == "r":
            if last is None:
                continue
            n = last
            rows = [r for r in rows if int(r["sign"]) != n]
        else:
            try:
                n = int(a.strip("#"))
                assert n in key
            except Exception:
                print("  type a number from 1 to 40")
                continue
            if n in done:
                rows = [r for r in rows if int(r["sign"]) != n]
        truth = key[n]
        ser.reset_input_buffer()
        ser.write(b"c")
        res, img = [], None
        t_end = time.time() + 15
        while time.time() < t_end and img is None:
            line = ser.readline().decode(errors="ignore").strip()
            if line.startswith("RES "):
                p = line.split()
                res.append(dict(k=int(p[1]), cap=float(p[2]), pre=float(p[3]), inf=float(p[4]),
                                top=[(int(p[5 + 2 * i]), float(p[6 + 2 * i])) for i in range(3)]))
            elif line.startswith("IMG "):
                img = np.frombuffer(bytes.fromhex(line[4:]), np.uint8)
        if not res:
            print("  no answer from the board: check the port, and that the Serial Monitor is closed")
            continue
        if img is not None and img.size == 48 * 48 * 3:
            Image.fromarray(img.reshape(48, 48, 3)).resize((240, 240), Image.NEAREST).save(HERE / "seen" / f"{args.dist}cm_#{n:02d}.png")
        preds = [names[r["top"][0][0]] for r in res]
        ok = [same(p, truth) for p in preds]
        vote = Counter(preds).most_common(1)[0][0]
        print(f"  truth: {truth[5:]}")
        for r, p, o in zip(res, preds, ok):
            print(f"   photo {r['k'] + 1}: {'OK ' if o else 'XX '} {p[5:]:32s} {r['top'][0][1] * 100:5.1f}%   "
                  f"(capture {r['cap']:.0f} ms, prep {r['pre']:.0f} ms, model {r['inf']:.0f} ms)")
        print(f"  {sum(ok)}/{len(ok)} photos correct; vote of 5 = {vote[5:]} -> {'OK' if same(vote, truth) else 'WRONG'}"
              f"   (image saved in camera_test/seen)")
        for r, p, o in zip(res, preds, ok):
            rows.append(dict(sign=n, dist_cm=args.dist, photo=r["k"] + 1, truth=truth, pred=p, prob=f"{r['top'][0][1]:.4f}",
                             correct=int(o), second=names[r["top"][1][0]], capture_ms=r["cap"], prep_ms=r["pre"], model_ms=r["inf"]))
        done.add(n)
        last = n
        with open(out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)

    if rows:
        signs = sorted({int(r["sign"]) for r in rows})
        acc = np.mean([int(r["correct"]) for r in rows]) * 100
        votes = []
        for s in signs:
            ps = [r["pred"] for r in rows if int(r["sign"]) == s]
            votes.append(same(Counter(ps).most_common(1)[0][0], key[s]))
        t = np.array([[float(r["capture_ms"]), float(r["prep_ms"]), float(r["model_ms"])] for r in rows])
        print(f"\n=== {args.dist} cm: {len(signs)} signs, {len(rows)} photos ===")
        print(f"single photo accuracy: {acc:.1f}%   |   5-photo vote accuracy: {np.mean(votes) * 100:.1f}% ({sum(votes)}/{len(votes)})")
        print(f"mean time per photo: capture {t[:, 0].mean():.0f} ms + preprocessing {t[:, 1].mean():.0f} ms + model {t[:, 2].mean():.0f} ms")
        print(f"results saved in {out}")


if __name__ == "__main__":
    main()
