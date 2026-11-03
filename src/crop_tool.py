"""
Crop tool: cut traffic signs out of screenshots, with class suggestions from our model.

  1. Put your screenshots (PNG/JPG) in one folder, e.g. screenshots\
  2. Run (inside `conda activate tinyml`, from the project folder):
        python src\\crop_tool.py --shots screenshots --out morocco_youtube
  3. For each screenshot:
        - draw a box around a sign with the mouse (whole sign; the tool adds a 10% margin)
        - the model shows its 3 best guesses with an example picture of each class
        - press 1, 2 or 3 to save with that class, O for "other sign" (not in our 70 classes),
          or pick any class in the list (type to filter) and press Enter
        - draw the next sign, or press N / Right arrow for the next screenshot
     Other keys: B / Left = previous screenshot, Esc = cancel the box, Z = undo the last saved crop.
  4. "Source" box (top): type the video link + time once; it is saved with every crop until you change it.

Blind mode (for a FINAL test set): add --blind. The model is not loaded and no guess is shown, so the
labels cannot be pushed toward the model's answer. Choose the class in the list (type to filter, Enter)
or press O for "other sign". Example:
        python src\\crop_tool.py --blind --shots screenshots_final --out morocco_final

Crops are saved at their real size (no enlarging) as PNG:
    <out>/<split>/<class>/<screenshot>_<k>.png
and every crop is logged in <out>/crops_log.csv (screenshot, box, class, model guess, source).
"""

import argparse
import csv
import json
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import tkinter as tk
from tkinter import ttk

import numpy as np
from PIL import Image, ImageOps, ImageTk

OTHER = "9999_other_sign"
IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
LOG_FIELDS = ["crop", "screenshot", "x0", "y0", "x1", "y1", "class", "model_top1", "model_conf", "source"]


def square_box(x0, y0, x1, y1, W, H, margin):
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    side = max(x1 - x0, y1 - y0) * (1 + 2 * margin)
    return (int(round(max(0, cx - side / 2))), int(round(max(0, cy - side / 2))),
            int(round(min(W, cx + side / 2))), int(round(min(H, cy + side / 2))))


class Model:
    """int8 TFLite classifier; returns class probabilities for one PIL crop."""

    def __init__(self, path, size, gray):
        import tensorflow as tf
        self.it = tf.lite.Interpreter(model_path=str(path))
        self.it.allocate_tensors()
        self.i, self.o = self.it.get_input_details()[0], self.it.get_output_details()[0]
        self.size, self.gray = size, gray

    def probs(self, im):
        x = np.asarray(im.convert("RGB").resize((self.size, self.size), Image.BILINEAR), np.float32) / 255.0
        if self.gray:
            x = (x @ np.array([0.2989, 0.5870, 0.1140], np.float32))[..., None]
        si, zi = self.i["quantization"]
        so, zo = self.o["quantization"]
        q = np.clip(np.round(x[None] / si + zi), -128, 127).astype(self.i["dtype"])
        self.it.set_tensor(self.i["index"], q)
        self.it.invoke()
        z = (self.it.get_tensor(self.o["index"])[0].astype(np.float32) - zo) * so
        e = np.exp(z - z.max())
        return e / e.sum()


class App:
    def __init__(self, root, args):
        self.root, self.args = root, args
        self.shots = sorted(p for p in Path(args.shots).rglob("*") if p.suffix.lower() in IMG_EXT)
        if not self.shots:
            raise SystemExit(f"no images found in {args.shots}")
        data = Path(args.data)
        self.classes = sorted(d.name for d in (data / "train").iterdir() if d.is_dir())
        self.all_names = self.classes + [OTHER]
        self.examples, self.examples3 = {}, {}
        for c in self.classes:
            fs = [p for p in sorted((data / "train" / c).iterdir()) if p.suffix.lower() in IMG_EXT]
            if fs:
                self.examples[c] = Image.open(fs[0]).convert("RGB").resize((72, 72), Image.NEAREST)
                # 3 examples spread over the folder, shown under the class list (works in blind mode too)
                pick = [fs[0], fs[len(fs) // 2], fs[-1]] if len(fs) >= 3 else fs
                self.examples3[c] = [Image.open(f).convert("RGB").resize((64, 64), Image.BILINEAR) for f in pick]
        self.model = None
        if not args.blind:
            run = Path(args.model).parent
            cfg = json.loads((run / "config.json").read_text()) if (run / "config.json").exists() else {}
            self.model = Model(args.model, int(cfg.get("size", 32)), bool(cfg.get("gray", False)))
            n_out = self.model.o["shape"][-1]
            if n_out not in (len(self.classes), len(self.classes) + 1):
                raise SystemExit(f"model has {n_out} outputs but {args.data} has {len(self.classes)} classes")
        self.out = Path(args.out)
        self.log_path = self.out / "crops_log.csv"
        self.out.mkdir(parents=True, exist_ok=True)
        if not self.log_path.exists():
            with open(self.log_path, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(LOG_FIELDS)
        with open(self.log_path, encoding="utf-8") as f:
            self.rows = [r for r in csv.DictReader(f) if Path(r["crop"]).exists()]
        done = {r["screenshot"] for r in self.rows}
        self.used_names = {Path(r["crop"]).name for r in self.rows} | {p.name for p in self.out.rglob("*.png")}
        self.total = len(self.rows)
        self.idx = next((k for k, p in enumerate(self.shots) if str(p) not in done), 0)
        self.undo_stack, self.box, self.start, self.rect, self.guess = [], None, None, None, []
        self._build()
        self.load()

    # ------------------------------------------------------------------ UI
    def _build(self):
        r = self.root
        r.title("Sign crop tool")
        top = tk.Frame(r)
        top.pack(fill="x", padx=6, pady=4)
        tk.Label(top, text="Source (video link + time):").pack(side="left")
        self.source = tk.StringVar(value="")
        self.src_entry = tk.Entry(top, textvariable=self.source, width=70)
        self.src_entry.pack(side="left", padx=4)
        self.status = tk.Label(top, text="", fg="#333")
        self.status.pack(side="right")

        mid = tk.Frame(r)
        mid.pack(fill="both", expand=True)
        self.canvas = tk.Canvas(mid, width=self.args.view_w, height=self.args.view_h, bg="#222", cursor="crosshair")
        self.canvas.pack(side="left", padx=4, pady=4)
        side = tk.Frame(mid, width=330)
        side.pack(side="left", fill="y", padx=6)
        tk.Label(side, text="Crop", font=("Segoe UI", 10, "bold")).pack(anchor="w")
        self.crop_lbl = tk.Label(side, bg="#ddd", width=16, height=8)
        self.crop_lbl.pack(anchor="w", pady=2)
        tk.Label(side, text=("BLIND MODE: no model guesses (choose the class below)" if self.args.blind
                             else "Model guesses (press 1 / 2 / 3)"),
                 font=("Segoe UI", 10, "bold"), fg="#a00" if self.args.blind else "#000").pack(anchor="w", pady=(8, 0))
        self.guess_rows = []
        for k in range(0 if self.args.blind else 3):
            fr = tk.Frame(side)
            fr.pack(anchor="w", fill="x", pady=2)
            img = tk.Label(fr)
            img.pack(side="left")
            txt = tk.Button(fr, text="", anchor="w", justify="left", wraplength=230,
                            command=lambda k=k: self.save_guess(k))
            txt.pack(side="left", fill="x", expand=True, padx=4)
            self.guess_rows.append((img, txt))
        tk.Button(side, text="O = other sign (not in our classes)", command=lambda: self.save(OTHER)).pack(
            anchor="w", fill="x", pady=(8, 2))
        tk.Label(side, text="Or any class: type to filter, Up/Down to choose, Enter = save:").pack(anchor="w", pady=(8, 0))
        self.combo_var = tk.StringVar()
        self.combo = tk.Entry(side, textvariable=self.combo_var, width=42)
        self.combo.pack(anchor="w")
        lf = tk.Frame(side)
        lf.pack(anchor="w", fill="x")
        self.listbox = tk.Listbox(lf, height=9, width=42, exportselection=False)
        sb = tk.Scrollbar(lf, orient="vertical", command=self.listbox.yview)
        self.listbox.config(yscrollcommand=sb.set)
        self.listbox.pack(side="left")
        sb.pack(side="left", fill="y")
        self.combo_var.trace_add("write", lambda *a: self._filter())
        self.combo.bind("<Return>", lambda e: self.save_combo())
        self.combo.bind("<Down>", lambda e: self._move_sel(1))
        self.combo.bind("<Up>", lambda e: self._move_sel(-1))
        self.listbox.bind("<Double-Button-1>", lambda e: self.save_combo())
        self.listbox.bind("<Return>", lambda e: self.save_combo())
        tk.Button(side, text="Save with chosen class", command=self.save_combo).pack(anchor="w", pady=2)
        tk.Label(side, text="Examples of the chosen class (training images, not a model guess):",
                 fg="#555").pack(anchor="w")
        self.ex_frame = tk.Frame(side)
        self.ex_frame.pack(anchor="w", pady=2)
        self.ex_lbls = [tk.Label(self.ex_frame) for _ in range(3)]
        for l in self.ex_lbls:
            l.pack(side="left", padx=2)
        self.ex_name = tk.Label(side, text="", fg="#333", wraplength=320, justify="left")
        self.ex_name.pack(anchor="w")
        self.listbox.bind("<<ListboxSelect>>", lambda e: self._show_examples())
        self._filter()
        tk.Label(side, text="\nZOOM: right-drag a rectangle (or mouse wheel) to zoom in,\n"
                            "right-click or F = whole image\n"
                            "Keys: N / Right = next screenshot, B / Left = previous\n"
                            "Esc = cancel box, Z = undo last crop", justify="left", fg="#555").pack(anchor="w")
        self.saved_lbl = tk.Label(side, text="", fg="#060", justify="left", wraplength=320)
        self.saved_lbl.pack(anchor="w", pady=6)

        self.canvas.bind("<ButtonPress-1>", self._press)
        self.canvas.bind("<B1-Motion>", self._drag)
        self.canvas.bind("<ButtonRelease-1>", self._release)
        self.canvas.bind("<ButtonPress-3>", self._zpress)
        self.canvas.bind("<B3-Motion>", self._zdrag)
        self.canvas.bind("<ButtonRelease-3>", self._zrelease)
        self.canvas.bind("<MouseWheel>", self._wheel)                    # Windows / macOS
        self.canvas.bind("<Button-4>", lambda e: self._wheel(e, 120))    # Linux
        self.canvas.bind("<Button-5>", lambda e: self._wheel(e, -120))
        self.zstart, self.zrect = None, None
        r.bind("<Key>", self._key)

    def _typing(self):
        return self.root.focus_get() in (self.src_entry, self.combo, self.listbox)

    def _key(self, e):
        if self._typing():
            if e.keysym == "Escape":
                self.canvas.focus_set()
            return
        k = e.keysym.lower()
        if k in ("1", "2", "3"):
            self.save_guess(int(k) - 1)
        elif k == "o":
            self.save(OTHER)
        elif k in ("n", "right"):
            self.move(1)
        elif k in ("b", "left"):
            self.move(-1)
        elif k == "escape":
            self.clear_box()
        elif k == "z":
            self.undo()
        elif k == "f":
            self.set_view(None)

    def _filter(self):
        t = self.combo_var.get().strip().lower()
        words = t.replace("_", " ").split()
        items = [c for c in self.all_names if all(w in c.lower().replace("_", " ") for w in words)] if words else self.all_names
        self.listbox.delete(0, "end")
        for c in items:
            self.listbox.insert("end", c)
        if items:
            self.listbox.selection_set(0)
            self.listbox.see(0)
        self._show_examples()

    def _show_examples(self):
        if not hasattr(self, "ex_lbls"):
            return
        sel = self.listbox.curselection()
        c = self.listbox.get(sel[0]) if sel else None
        ims = self.examples3.get(c, []) if c else []
        self.tk_ex3 = [ImageTk.PhotoImage(im) for im in ims]
        for k, l in enumerate(self.ex_lbls):
            l.config(image=self.tk_ex3[k] if k < len(self.tk_ex3) else "", width=64 if k < len(self.tk_ex3) else 1)
        self.ex_name.config(text=(c or "") if c != OTHER else "other sign (not one of our 70 classes)")

    def _move_sel(self, d):
        n = self.listbox.size()
        if not n:
            return "break"
        cur = self.listbox.curselection()
        i = min(max((cur[0] if cur else -1) + d, 0), n - 1)
        self.listbox.selection_clear(0, "end")
        self.listbox.selection_set(i)
        self.listbox.see(i)
        self._show_examples()
        return "break"

    # ------------------------------------------------------------------ images
    def load(self):
        p = self.shots[self.idx]
        self.img = ImageOps.exif_transpose(Image.open(p)).convert("RGB")
        self.n_saved_here = sum(1 for row in self.rows if row["screenshot"] == str(p))
        self.box = None
        self.set_view(None)
        self.clear_box()
        self._status()
        self.canvas.focus_set()

    # ------------------------------------------------------------------ view / zoom
    def set_view(self, v):
        """v = (x0, y0, x1, y1) in image pixels, or None for the whole image."""
        W, H = self.img.size
        if v is None:
            v = (0, 0, W, H)
        x0, y0, x1, y1 = v
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(W, x1), min(H, y1)
        if x1 - x0 < 16 or y1 - y0 < 16:
            return
        self.view = (x0, y0, x1, y1)
        self.scale = min(self.args.view_w / (x1 - x0), self.args.view_h / (y1 - y0), 8.0)
        self.render()

    def to_canvas(self, box):
        x0, y0 = self.view[0], self.view[1]
        s = self.scale
        return [(box[0] - x0) * s, (box[1] - y0) * s, (box[2] - x0) * s, (box[3] - y0) * s]

    def to_image(self, x, y):
        return self.view[0] + x / self.scale, self.view[1] + y / self.scale

    def render(self):
        x0, y0, x1, y1 = [int(round(v)) for v in self.view]
        part = self.img.crop((x0, y0, x1, y1))
        w, h = max(1, int((x1 - x0) * self.scale)), max(1, int((y1 - y0) * self.scale))
        part = part.resize((w, h), Image.BILINEAR if self.scale <= 1 else Image.LANCZOS)
        self.tk_img = ImageTk.PhotoImage(part)
        self.canvas.delete("all")
        self.rect, self.zrect = None, None
        self.canvas.create_image(0, 0, anchor="nw", image=self.tk_img)
        p = str(self.shots[self.idx])
        for row in self.rows:
            if row["screenshot"] == p:
                bx = self.to_canvas([int(row[k]) for k in ("x0", "y0", "x1", "y1")])
                self.canvas.create_rectangle(*bx, outline="#0f0", width=2)
                self.canvas.create_text(bx[0] + 2, bx[1] - 2, anchor="sw", text=row["class"][:22], fill="#0f0")
        if self.box:
            self.rect = self.canvas.create_rectangle(*self.to_canvas(self.box), outline="#ff0", width=2)
        zoom = (self.img.size[0] / (self.view[2] - self.view[0]))
        if zoom > 1.01:
            self.canvas.create_text(8, 8, anchor="nw", text=f"zoom x{zoom:.1f}  (right-click or F = whole image)",
                                    fill="#ff0")

    def _zpress(self, e):
        self.zstart = (e.x, e.y)
        self.zrect = self.canvas.create_rectangle(e.x, e.y, e.x, e.y, outline="#3af", width=2, dash=(4, 2))

    def _zdrag(self, e):
        if self.zstart:
            self.canvas.coords(self.zrect, self.zstart[0], self.zstart[1], e.x, e.y)

    def _zrelease(self, e):
        if not self.zstart:
            return
        (ax, ay), self.zstart = self.zstart, None
        if abs(e.x - ax) < 8 or abs(e.y - ay) < 8:       # a simple right-click: back to the whole image
            self.set_view(None)
            return
        X0, Y0 = self.to_image(min(ax, e.x), min(ay, e.y))
        X1, Y1 = self.to_image(max(ax, e.x), max(ay, e.y))
        self.set_view((X0, Y0, X1, Y1))

    def _wheel(self, e, delta=None):
        d = delta if delta is not None else e.delta
        f = 0.7 if d > 0 else 1 / 0.7
        cx, cy = self.to_image(e.x, e.y)
        x0, y0, x1, y1 = self.view
        W, H = self.img.size
        nw, nh = min(W, (x1 - x0) * f), min(H, (y1 - y0) * f)
        nx0 = min(max(0, cx - (cx - x0) * f), W - nw)
        ny0 = min(max(0, cy - (cy - y0) * f), H - nh)
        self.set_view((nx0, ny0, nx0 + nw, ny0 + nh))

    def _status(self):
        p = self.shots[self.idx]
        self.status.config(text=f"screenshot {self.idx + 1}/{len(self.shots)}: {p.parent.name}/{p.name}   "
                                f"| crops here {self.n_saved_here} | total crops {self.total}")

    def move(self, d):
        self.idx = min(max(self.idx + d, 0), len(self.shots) - 1)
        self.load()

    # ------------------------------------------------------------------ box
    def _press(self, e):
        self.start = (e.x, e.y)
        if self.rect:
            self.canvas.delete(self.rect)
        self.rect = self.canvas.create_rectangle(e.x, e.y, e.x, e.y, outline="#ff0", width=2)

    def _drag(self, e):
        if self.start:
            self.canvas.coords(self.rect, self.start[0], self.start[1], e.x, e.y)

    def _release(self, e):
        if not self.start:
            return
        x0, x1 = sorted((self.start[0], e.x))
        y0, y1 = sorted((self.start[1], e.y))
        self.start = None
        if x1 - x0 < 4 or y1 - y0 < 4:
            self.clear_box()
            return
        W, H = self.img.size
        X0, Y0 = self.to_image(x0, y0)
        X1, Y1 = self.to_image(x1, y1)
        self.box = square_box(X0, Y0, X1, Y1, W, H, self.args.margin)
        self.canvas.coords(self.rect, *self.to_canvas(self.box))
        self.crop = self.img.crop(self.box)
        prev = self.crop.resize((128, 128), Image.NEAREST)
        self.tk_crop = ImageTk.PhotoImage(prev)
        self.crop_lbl.config(image=self.tk_crop, width=128, height=128)
        if self.model is None:                       # blind mode: no suggestion at all
            self.guess = []
            p, order = None, []
        else:
            p = self.model.probs(self.crop)
            order = np.argsort(-p)[:3]
            self.guess = [(self.all_names[i] if i < len(self.all_names) else OTHER, float(p[i])) for i in order]
        self.tk_ex = []
        for (img, txt), (c, conf) in zip(self.guess_rows, self.guess):
            ex = self.examples.get(c)
            t = ImageTk.PhotoImage(ex) if ex else None
            self.tk_ex.append(t)
            img.config(image=t if t else "", width=72 if t else 9)
            txt.config(text=f"{len(self.tk_ex)}.  {c}\n    {100*conf:.0f}%")
        self.combo_var.set("")
        small = min(self.box[2] - self.box[0], self.box[3] - self.box[1])
        self.saved_lbl.config(text=f"crop {small} px" + ("  (very small)" if small < 16 else ""), fg="#555")

    def clear_box(self):
        if self.rect:
            self.canvas.delete(self.rect)
        self.rect, self.box, self.start, self.guess = None, None, None, []
        self.crop_lbl.config(image="", width=16, height=8)
        for img, txt in self.guess_rows:
            img.config(image="", width=9)
            txt.config(text="")

    # ------------------------------------------------------------------ save
    def save_guess(self, k):
        if self.box and k < len(self.guess):
            self.save(self.guess[k][0])

    def save_combo(self):
        sel = self.listbox.curselection()
        c = self.listbox.get(sel[0]) if sel else self.combo_var.get().strip()
        if c not in self.all_names:
            self.saved_lbl.config(text=f"no class matches: {self.combo_var.get()}", fg="#a00")
            return
        self.save(c)
        self.combo_var.set("")
        self.canvas.focus_set()

    def save(self, cls):
        if not self.box:
            self.saved_lbl.config(text="draw a box first", fg="#a00")
            return
        shot = self.shots[self.idx]
        d = self.out / self.args.split / cls
        d.mkdir(parents=True, exist_ok=True)
        k = 0
        while f"{shot.stem}_{k}.png" in self.used_names:
            k += 1
        path = d / f"{shot.stem}_{k}.png"
        self.used_names.add(path.name)
        self.crop.save(path)
        top = self.guess[0] if self.guess else ("", 0.0)
        row = [str(path), str(shot), *self.box, cls, top[0], f"{top[1]:.3f}", self.source.get().strip()]
        with open(self.log_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(row)
        self.rows.append(dict(zip(LOG_FIELDS, [str(v) for v in row])))
        self.total += 1
        self.undo_stack.append(path)
        self.n_saved_here += 1
        bx = self.to_canvas(self.box)
        self.canvas.create_rectangle(*bx, outline="#0f0", width=2)
        self.canvas.create_text(bx[0] + 2, bx[1] - 2, anchor="sw", text=cls[:22], fill="#0f0")
        if self.rect:
            self.canvas.delete(self.rect)
        self.rect, self.box = None, None
        self.saved_lbl.config(text=f"saved: {path.name}\n-> {cls}", fg="#060")
        self._status()

    def undo(self):
        if not self.undo_stack:
            return
        path = self.undo_stack.pop()
        if path.exists():
            path.unlink()
        with open(self.log_path, encoding="utf-8") as f:
            rows = list(csv.reader(f))
        rows = [r for r in rows if not r or r[0] != str(path)]
        with open(self.log_path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerows(rows)
        self.rows = [r for r in self.rows if r["crop"] != str(path)]
        self.used_names.discard(path.name)
        self.total -= 1
        self.saved_lbl.config(text=f"removed: {path.name}", fg="#a60")
        self.load()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shots", required=True, help="folder with screenshots")
    ap.add_argument("--out", default="morocco_youtube")
    ap.add_argument("--split", default="test")
    ap.add_argument("--data", default="ts1m_ma_32_clean", help="dataset whose folders are the class names")
    ap.add_argument("--model", default=r"runs\c32_mild40k_s42_maB\model_int8.tflite")
    ap.add_argument("--margin", type=float, default=0.10)
    ap.add_argument("--view-w", type=int, default=1000)
    ap.add_argument("--view-h", type=int, default=720)
    ap.add_argument("--blind", action="store_true",
                    help="do not load the model and show no guesses (use this for the final test set)")
    args = ap.parse_args()
    root = tk.Tk()
    App(root, args)
    root.mainloop()


if __name__ == "__main__":
    main()
