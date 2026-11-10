"""
Make the paper figures (PNG, 300 dpi, plus PDF) in the folder `figures`.

    fig0_pipeline.png       the pipeline, drawn by code
    fig1_steps.png          accuracy and speed-limit accuracy on the YouTube test, step by step
    fig2_speed_confusion.png which speed limit is read as which: before vs after (3 seeds summed)
    fig3_adaptation.png     Kaggle test accuracy vs local photos per class: PC fine-tuning vs on-chip prototypes
    fig4_latency.png        time per image on the boards (read from latency_measurements.csv)
    fig5_robustness.png     accuracy lost under each image corruption: standard vs mild robust training
    fig6_synthetic.png      examples of synthetic speed signs (drawn signs and digit swaps)
    fig7_examples.png       a few YouTube crops fixed by the recipe, and a few still wrong (check image rights)

Usage (inside `conda activate tinyml`, from the project folder):
    python make_figures.py
Each figure is made independently: if one fails (missing files), the others are still written.
"""

import csv
import json
import os
import re
import traceback
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches
from PIL import Image

OUT = Path("figures")
C_BASE, C_MAIN, C_ACC, C_GREY = "#9AA5B1", "#1F6FB2", "#D9822B", "#5F6B7A"
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "TeX Gyre Termes", "DejaVu Serif"], "pdf.fonttype": 42, "axes.grid": False,
                     "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titlesize": 10, "savefig.bbox": "tight", "figure.dpi": 110})
SPEED = {11: "30", 17: "40", 23: "50", 28: "60", 32: "70", 36: "80", 39: "90", 41: "100", 46: "120"}


def save(fig, name):
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / f"{name}.png", dpi=300)
    fig.savefig(OUT / f"{name}.pdf")
    plt.close(fig)
    print("  written", OUT / f"{name}.png")


# ---------------------------------------------------------------- shared: int8 inference on crops
def int8_logits(model_path, x):
    import tensorflow as tf
    it = tf.lite.Interpreter(model_path=str(model_path))
    it.allocate_tensors()
    i, o = it.get_input_details()[0], it.get_output_details()[0]
    si, zi = i["quantization"]
    so, zo = o["quantization"]
    out = []
    for k in range(len(x)):
        it.set_tensor(i["index"], np.clip(np.round(x[k:k + 1] / si + zi), -128, 127).astype(np.int8))
        it.invoke()
        out.append((it.get_tensor(o["index"])[0].astype(np.float32) - zo) * so)
    return np.array(out)


def youtube_items(classes):
    items = []
    for d in sorted(p for p in Path("morocco_youtube/test").iterdir() if p.is_dir()):
        if d.name in classes:
            items += [(f, d.name) for f in sorted(d.glob("*.png"))]
    return items


def load(items, size):
    return np.stack([np.asarray(Image.open(f).convert("RGB").resize((size, size), Image.BILINEAR),
                                np.float32) / 255.0 for f, _ in items])


def run_size(run):
    return int(json.loads((Path("runs") / run / "config.json").read_text()).get("size", 32))



# ---------------------------------------------------------------- figure 0: pipeline
def fig_pipeline():
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 62)
    ax.axis("off")

    def box(x, y, w, h, title, sub, color="white", edge=C_GREY):
        ax.add_patch(matplotlib.patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1.2",
                                                       fc=color, ec=edge, lw=0.9))
        ax.text(x + w / 2, y + h * 0.62, title, ha="center", va="center", fontsize=6.8, weight="bold")
        ax.text(x + w / 2, y + h * 0.27, sub, ha="center", va="center", fontsize=6, color=C_GREY)

    def arrow(x0, y0, x1, y1):
        ax.annotate("", (x1, y1), (x0, y0), arrowprops=dict(arrowstyle="-|>", color=C_GREY, lw=0.8))

    src = "#EEF1F4"
    box(2, 50, 22, 9, "TS-1M (worldwide)", "70 Morocco-relevant classes", src)
    box(28, 50, 22, 9, "Kaggle Moroccan photos", "2,676 crops", src)
    box(54, 50, 22, 9, "3 training drives", "868 own crops", src)
    box(78, 50, 20, 9, "4 test drives", "696 crops, test only", src)
    box(2, 36, 22, 9, "label cleaning", "Confident Learning + check", "white")
    box(54, 36, 17, 9, "synthetic signs", "drawn + digit swaps", "white")
    box(2, 22, 22, 9, "train tiny CNN", "40 K weights, 32/48 px", "white")
    box(28, 22, 48, 9, "fine-tune on Moroccan data", "15% local per batch, cap 40, mirror copies", "white")
    box(28, 8, 22, 9, "int8 conversion", "about 63 KB", "#E3EEF8", C_MAIN)
    box(54, 8, 20, 9, "STM32 / ESP32", "1.8–189 ms per image", "white")
    box(78, 8, 20, 9, "test on drives", "accuracy, speed limits", "white")
    ax.text(74.6, 40, "own\ncrops", fontsize=5.6, color=C_GREY)
    arrow(13, 50, 13, 45); arrow(13, 36, 13, 31); arrow(24, 26.5, 28, 26.5)
    arrow(39, 50, 39, 31); arrow(62.5, 50, 62.5, 45); arrow(62.5, 36, 62.5, 31); arrow(74, 50, 74, 31)
    arrow(52, 22, 40, 17); arrow(50, 12.5, 54, 12.5); arrow(74, 12.5, 78, 12.5)
    ax.annotate("", (88, 17), (88, 50), arrowprops=dict(arrowstyle="-|>", color=C_GREY, lw=0.8, ls="--"))
    ax.text(89, 33, "never used\nfor training", fontsize=6, color=C_GREY)
    save(fig, "fig0_pipeline")


# ---------------------------------------------------------------- figure 1
SAME = [{218, 221}]          # bump / uneven road: either counts as correct (as in eval_youtube.py)


def youtube_scores(run, classes, items, cache):
    """Accuracy on all known-sign crops and on speed limits, for one int8 model."""
    S = run_size(run)
    if S not in cache:
        cache[S] = load(items, S)
    pred = int8_logits(Path("runs") / run / "model_int8.tflite", cache[S]).argmax(1)
    ids = [int(c[:4]) for c in classes]
    ok = np.array([ids[p] == int(c[:4]) or any(ids[p] in g and int(c[:4]) in g for g in SAME)
                   for p, (_, c) in zip(pred, items)])
    sp = np.array([int(c[:4]) in SPEED for _, c in items])
    return 100 * ok.mean(), 100 * ok[sp].mean()


def fig_steps():
    classes = sorted(d.name for d in Path("ts1m_ma_32_clean/train").iterdir() if d.is_dir())
    items = youtube_items(set(classes))
    steps = [("TS-1M only\n32×32", "c32_relu40k_s{}"), ("+ Kaggle\nphotos", "c32_relu40k_s{}_maB"),
             ("+ own crops\n(3 drives)", "c32_relu40k_s{}_maS"), ("+ synthetic\nspeed signs", "c32_relu40k_s{}_maT"),
             ("TS-1M only\n48×48", "c48_relu40k_s{}"), ("full recipe\n48×48", "c48_relu40k_s{}_maT"),
             ("+ synthetic\nrare signs", "c48_relu40k_s{}_maR")]
    cache, rows = {}, []
    for name, pat in steps:
        r = [youtube_scores(pat.format(s), classes, items, cache) for s in (42, 1, 2)
             if (Path("runs") / pat.format(s) / "model_int8.tflite").exists()]
        if r:
            r = np.array(r)
            rows.append((name, r[:, 0].mean(), r[:, 0].std(), r[:, 1].mean(), r[:, 1].std()))
            print(f"  {name.replace(chr(10), ' ')}: all {r[:, 0].mean():.1f}, speed {r[:, 1].mean():.1f} ({len(r)} seeds)")
    if not rows:
        raise RuntimeError("no model found in runs/ for figure 1")
    n_all = len(items)
    n_sp = sum(int(c[:4]) in SPEED for _, c in items)
    x = np.arange(len(rows))
    w = 0.38
    fig, ax = plt.subplots(figsize=(7.4, 2.9))
    a = ax.bar(x - w / 2, [r[1] for r in rows], w, yerr=[r[2] for r in rows], color=C_MAIN, capsize=2,
               label=f"all {n_all} signs")
    b = ax.bar(x + w / 2, [r[3] for r in rows], w, yerr=[r[4] for r in rows], color=C_ACC, capsize=2,
               label=f"speed limits ({n_sp})")
    for bars, errs in ((a, [r[2] for r in rows]), (b, [r[4] for r in rows])):
        for r, e in zip(bars, errs):
            ax.text(r.get_x() + r.get_width() / 2, r.get_height() + e + 1.5, f"{r.get_height():.0f}",
                    ha="center", fontsize=7.5)
    split = next((i for i, r in enumerate(rows) if "48" in r[0]), None)
    if split:
        ax.axvline(split - 0.5, color=C_GREY, lw=0.8, ls="--")
        ax.text((split - 1) / 2, 97, "32×32 input", ha="center", color=C_GREY, fontsize=8)
        ax.text((split + len(rows) - 1) / 2, 97, "48×48 input", ha="center", color=C_GREY, fontsize=8)
    ax.set_xticks(x, [r[0] for r in rows], fontsize=7.5)
    ax.set_ylim(0, 100)
    ax.set_ylabel("accuracy on Moroccan\ndriving videos (%)")
    ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, fontsize=8)
    save(fig, "fig1_steps")


# ---------------------------------------------------------------- figure 2
def fig_speed_confusion():
    classes = sorted(d.name for d in Path("ts1m_ma_32_clean/train").iterdir() if d.is_dir())
    cidx = {c: i for i, c in enumerate(classes)}
    sp = [c for c in classes if int(c[:4]) in SPEED]
    items = [it for it in youtube_items(set(sp))]
    labels = [SPEED[int(c[:4])] for c in sp]
    present = sorted({int(c[:4]) for _, c in items})
    rows = [SPEED[i] for i in present]
    cols = labels + ["other"]
    pairs = [("before: 32×32 + Kaggle", "c32_relu40k_s{}_maB"), ("after: 48×48 full recipe", "c48_relu40k_s{}_maR")]
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.0))
    for ax, (title, pat) in zip(axes, pairs):
        M = np.zeros((len(rows), len(cols)))
        for s in (42, 1, 2):
            run = pat.format(s)
            S = run_size(run)
            z = int8_logits(Path("runs") / run / "model_int8.tflite", load(items, S))
            pred = z.argmax(1)
            for (f, c), p in zip(items, pred):
                r = rows.index(SPEED[int(c[:4])])
                pc = classes[p]
                col = cols.index(SPEED[int(pc[:4])]) if int(pc[:4]) in SPEED else len(cols) - 1
                M[r, col] += 1
        P = M / M.sum(1, keepdims=True) * 100
        ax.imshow(P, cmap="Blues", vmin=0, vmax=100)
        for i in range(len(rows)):
            for j in range(len(cols)):
                if P[i, j] >= 1:
                    ax.text(j, i, f"{P[i, j]:.0f}", ha="center", va="center", fontsize=6.5,
                            color="white" if P[i, j] > 55 else "black")
        ax.set_xticks(range(len(cols)), cols, fontsize=7)
        ax.set_yticks(range(len(rows)), rows, fontsize=7)
        ax.set_xlabel("predicted (km/h)")
        ax.set_ylabel("true (km/h)")
        ax.set_title(title)
        for sp_ in ax.spines.values():
            sp_.set_visible(False)
    save(fig, "fig2_speed_confusion")


# ---------------------------------------------------------------- figure 3
def fig_adaptation():
    # PC fine-tuning with K photos per class (Kaggle test accuracy after fine-tuning), from the logs
    ks = [5, 10, 25, 50]
    ft = defaultdict(list)
    for s in (42, 1, 2):
        for k in ks:
            log = Path("runs") / f"c32_mild40k_s{s}_k{k}.log"
            if log.exists():
                m = re.search(r"^Moroccan accuracy\s+([\d.]+)%\s+([\d.]+)%", log.read_text(errors="ignore"), re.M)
                if m:
                    ft[0].append(float(m.group(1)))
                    ft[k].append(float(m.group(2)))
    # on-chip prototypes (calibrated), from proto_calib.txt
    pr = {}
    txt = Path("proto_calib.txt").read_text()
    for m in re.finditer(r"^\s+(\d+|all)\s+(head only|calibrated)\s+([\d.]+) ± ([\d.]+)", txt, re.M):
        k = 0 if m.group(2) == "head only" else (100 if m.group(1) == "all" else int(m.group(1)))
        pr[k] = (float(m.group(3)), float(m.group(4)))
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    kf = sorted(ft)
    ax.errorbar([max(k, 0.7) for k in kf], [np.mean(ft[k]) for k in kf], [np.std(ft[k]) for k in kf],
                color=C_MAIN, marker="o", ms=4, capsize=2, label="fine-tuning on a PC")
    kp = sorted(k for k in pr if k <= 50)
    ax.errorbar([max(k, 0.7) for k in kp], [pr[k][0] for k in kp], [pr[k][1] for k in kp],
                color=C_ACC, marker="s", ms=4, capsize=2, label="prototypes on the chip (no training)")
    ax.set_xscale("log")
    ax.set_xticks([0.7, 1, 3, 5, 10, 25, 50], ["0", "1", "3", "5", "10", "25", "50"])
    ax.minorticks_off()
    ax.set_xlabel("local photos per Moroccan class")
    ax.set_ylabel("accuracy on Kaggle Moroccan test (%)")
    ax.legend(frameon=False, fontsize=7.5, loc="lower right")
    save(fig, "fig3_adaptation")


# ---------------------------------------------------------------- figure 4
LATENCY_CSV = Path("latency_measurements.csv")
# Board measurements cannot be recomputed on the PC: they are read from this data file.
# If it does not exist, it is created with the measurements recorded so far (ST Edge AI Developer
# Cloud for the STM32 boards, our ESP32 Serial Monitor output); add new rows as you measure.
LATENCY_ROWS = [
    ("STM32F401RE", "Cortex-M4", 84, "c32_relu_wider", 32, 31.9, "ST Edge AI Developer Cloud"),
    ("STM32L4R9", "Cortex-M4", 120, "c32_relu_wider", 32, 22.5, "ST Edge AI Developer Cloud"),
    ("STM32F469", "Cortex-M4", 180, "c32_relu_wider", 32, 17.4, "ST Edge AI Developer Cloud"),
    ("STM32G474", "Cortex-M4", 170, "c32_relu_wider", 32, 16.7, "ST Edge AI Developer Cloud"),
    ("STM32C5A3", "Cortex-M33", 144, "c32_relu_wider", 32, 16.8, "ST Edge AI Developer Cloud"),
    ("STM32U585", "Cortex-M33", 160, "c32_relu_wider", 32, 15.2, "ST Edge AI Developer Cloud"),
    ("STM32H573", "Cortex-M33", 250, "c32_relu_wider", 32, 9.8, "ST Edge AI Developer Cloud"),
    ("STM32F746", "Cortex-M7", 216, "c32_relu_wider", 32, 8.2, "ST Edge AI Developer Cloud"),
    ("STM32H7B3", "Cortex-M7", 280, "c32_relu_wider", 32, 6.4, "ST Edge AI Developer Cloud"),
    ("STM32H747", "Cortex-M7", 400, "c32_relu_wider", 32, 4.5, "ST Edge AI Developer Cloud"),
    ("STM32H743", "Cortex-M7", 480, "c32_relu_wider", 32, 3.4, "ST Edge AI Developer Cloud"),
    ("STM32N6570", "Cortex-M55", 800, "c32_relu_wider", 32, 1.8, "ST Edge AI Developer Cloud"),
    ("ESP32-D0WD-V3", "Xtensa LX6", 240, "c32_mild40k_s42", 32, 82.6, "ESP32_SignTest Serial Monitor"),
    ("ESP32-D0WD-V3", "Xtensa LX6", 240, "c48_relu40k_s42", 48, 189.2, "ESP32_SignTest Serial Monitor"),
]


def fig_latency():
    if not LATENCY_CSV.exists():
        with open(LATENCY_CSV, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["board", "core", "mhz", "model", "input", "ms_per_image", "source"])
            w.writerows(LATENCY_ROWS)
        print(f"  created {LATENCY_CSV} (measured values; add new board measurements there)")
    rows = list(csv.DictReader(open(LATENCY_CSV, encoding="utf-8")))
    # one row per board, two bars: 32x32 (light) and 48x48 (dark); boards sorted by their slowest time
    per = defaultdict(dict)
    label = {}
    for r in rows:
        per[r["board"]][int(r["input"])] = float(r["ms_per_image"])
        label[r["board"]] = f"{r['board']} ({r['core'].replace('Cortex-', '')}, {r['mhz']} MHz)"
    # only boards measured at both sizes, so every pair of bars can be compared
    order = sorted((b for b in per if len(per[b]) == 2), key=lambda b: max(per[b].values()))
    fig, ax = plt.subplots(figsize=(5.6, 0.36 * len(order) + 1.0))
    y = np.arange(len(order))
    h = 0.38
    for size, off, col, col_esp in ((32, -h / 2, C_BASE, "#F0B47A"), (48, h / 2, C_MAIN, C_ACC)):
        for i, b in enumerate(order):
            if size not in per[b]:
                continue
            v = per[b][size]
            ax.barh(i + off, v, height=h, color=col_esp if b.startswith("ESP32") else col)
            ax.text(v * 1.07, i + off, f"{v:g}", va="center", fontsize=6)
    ax.set_xscale("log")
    ax.set_yticks(y, [label[b] for b in order], fontsize=7)
    ax.set_xlabel("time per image (ms, log scale)")
    ax.axvline(100, color=C_GREY, lw=0.8, ls="--")
    ax.text(100, len(order) - 0.45, "  10 images/s", color=C_GREY, fontsize=7)
    ax.set_xlim(1, 600)
    ax.set_ylim(-0.6, len(order) - 0.1)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=C_BASE, label="32×32 input"), Patch(color=C_MAIN, label="48×48 input"),
                       Patch(color=C_ACC, label="ESP32 (no int8 SIMD)")],
              fontsize=7, frameon=False, loc="lower right")
    models = sorted({r["model"] for r in rows})
    save(fig, "fig4_latency")


# ---------------------------------------------------------------- figure 5
def fig_robustness():
    rows = list(csv.DictReader(open("robustness.csv", encoding="utf-8")))
    groups = {"standard": re.compile(r"^c32_relu40k_s\d+$"), "mild robust": re.compile(r"^c32_mild40k_s\d+$")}
    drop = {g: defaultdict(list) for g in groups}
    for g, pat in groups.items():
        clean = {r["run"]: float(r["ts1m_balanced"]) for r in rows if pat.match(r["run"]) and r["corruption"] == "clean"}
        for r in rows:
            if pat.match(r["run"]) and r["corruption"] != "clean" and r["run"] in clean:
                drop[g][r["corruption"]].append(float(r["ts1m_balanced"]) - clean[r["run"]])
    cors = sorted(drop["standard"], key=lambda c: np.mean(drop["standard"][c]))
    if not cors:
        raise RuntimeError("no matching runs in robustness.csv")
    fig, ax = plt.subplots(figsize=(5.4, 0.22 * len(cors) + 0.9))
    y = np.arange(len(cors))
    h = 0.4
    ax.barh(y - h / 2, [np.mean(drop["standard"][c]) for c in cors], h, color=C_BASE, label="standard training")
    ax.barh(y + h / 2, [np.mean(drop["mild robust"].get(c, [0])) for c in cors], h, color=C_MAIN,
            label="mild robust training")
    ax.axvline(0, color="black", lw=0.6)
    ax.set_yticks(y, cors, fontsize=7)
    ax.set_xlabel("change in TS-1M balanced accuracy vs clean images (points, int8, 3 seeds)")
    ax.legend(frameon=False, fontsize=7.5, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2)
    save(fig, "fig5_robustness")


# ---------------------------------------------------------------- figure 6
def fig_synthetic():
    root = Path("morocco_synth/train")
    cls = [d for d in sorted(root.iterdir()) if d.is_dir() and int(d.name[:4]) in SPEED]
    show = [c for c in cls if SPEED[int(c.name[:4])] in ("40", "60", "80", "100", "120")]
    fig, axes = plt.subplots(len(show), 8, figsize=(6.4, 0.85 * len(show)))
    for r, c in enumerate(show):
        fs = sorted(c.glob("drawn_*"))[:4] + sorted(c.glob("swap_*"))[:4]
        for j in range(8):
            ax = axes[r, j]
            ax.axis("off")
            if j < len(fs):
                ax.imshow(Image.open(fs[j]))
        axes[r, 0].text(-0.15, 0.5, f"{SPEED[int(c.name[:4])]}", transform=axes[r, 0].transAxes,
                        ha="right", va="center", fontsize=8)
    fig.subplots_adjust(wspace=0.05, hspace=0.08)
    fig.text(0.31, 0.92, "drawn signs", ha="center", fontsize=8)
    fig.text(0.72, 0.92, "digit swaps (real crop, new number)", ha="center", fontsize=8)
    save(fig, "fig6_synthetic")


# ---------------------------------------------------------------- figure 7
def fig_examples():
    classes = sorted(d.name for d in Path("ts1m_ma_32_clean/train").iterdir() if d.is_dir())
    items = youtube_items(set(classes))
    y = np.array([classes.index(c) for _, c in items])
    before = int8_logits(Path("runs/c32_relu40k_s42/model_int8.tflite"), load(items, 32)).argmax(1)
    after = int8_logits(Path("runs/c48_relu40k_s42_maR/model_int8.tflite"), load(items, 48)).argmax(1)
    rng = np.random.default_rng(0)
    fixed = [k for k in range(len(items)) if before[k] != y[k] and after[k] == y[k]]
    wrong = [k for k in range(len(items)) if after[k] != y[k]]
    pick = list(rng.choice(fixed, min(4, len(fixed)), replace=False)) + list(rng.choice(wrong, min(2, len(wrong)), replace=False))
    short = lambda c: re.sub(r"^\d+_", "", c).replace("maximum_speed_limit_", "speed ").replace("maximum_speed_limit", "speed ").replace("_", " ")[:22]
    fig, axes = plt.subplots(1, len(pick), figsize=(1.25 * len(pick), 1.9))
    for ax, k in zip(np.atleast_1d(axes), pick):
        ax.imshow(Image.open(items[k][0]).convert("RGB").resize((96, 96)))
        ax.axis("off")
        col = "#1a7f37" if after[k] == y[k] else "#b42318"
        ax.set_title(f"true: {short(classes[y[k]])}\nbefore: {short(classes[before[k]])}\nafter: {short(classes[after[k]])}",
                     fontsize=5.5, color=col)
    save(fig, "fig7_examples")


# ---------------------------------------------------------------- figure 8
def fig_backbone():
    """The network, drawn from its definition (widths 16-32-64-96-160, linear head over 70 classes)."""
    from matplotlib.patches import FancyBboxPatch
    blocks = [  # (title, detail, output size at 48x48 input, at 32x32, channels)
        ("input", "RGB", 48, 32, 3),
        ("conv 3\u00d73 /2", "ReLU6", 24, 16, 16),
        ("DW 3\u00d73\nPW 1\u00d71", "ReLU6", 24, 16, 32),
        ("DW 3\u00d73 /2\nPW 1\u00d71", "ReLU6", 12, 8, 64),
        ("DW 3\u00d73\nPW 1\u00d71", "ReLU6", 12, 8, 96),
        ("DW 3\u00d73 /2\nPW 1\u00d71", "ReLU6", 6, 4, 160),
        ("average\npooling", "features", 1, 1, 160),
        ("linear", "class scores", 1, 1, 70),
    ]
    fig, ax = plt.subplots(figsize=(7.4, 2.6))
    ax.set_xlim(0, len(blocks)); ax.set_ylim(0, 1); ax.axis("off")
    for i, (title, detail, s48, s32, ch) in enumerate(blocks):
        x = i + 0.06
        col = C_BASE if i == 0 else (C_ACC if i >= 6 else C_MAIN)
        ax.add_patch(FancyBboxPatch((x, 0.30), 0.86, 0.50, boxstyle="round,pad=0.02,rounding_size=0.05",
                                    fc=col, ec="none", alpha=0.95))
        ax.text(x + 0.43, 0.60, title, ha="center", va="center", fontsize=6.6, color="white", weight="bold")
        ax.text(x + 0.43, 0.38, detail, ha="center", va="center", fontsize=5.8, color="white")
        size = f"{s48}\u00d7{s48}\u00d7{ch}" if s48 > 1 else f"{ch}"
        ax.text(x + 0.43, 0.20, size, ha="center", va="center", fontsize=6.2, color="#222222")
        if s48 > 1:
            ax.text(x + 0.43, 0.09, f"({s32}\u00d7{s32}\u00d7{ch})", ha="center", va="center", fontsize=5.6, color=C_GREY)
        if i < len(blocks) - 1:
            ax.annotate("", xy=(i + 1.06, 0.55), xytext=(i + 0.92, 0.55),
                        arrowprops=dict(arrowstyle="->", color=C_GREY, lw=0.8))
    ax.text(0, 0.95, "DW = depthwise, PW = pointwise, /2 = stride 2; sizes at 48\u00d748 input (32\u00d732 in brackets)", fontsize=6.2, color=C_GREY, va="center")
    save(fig, "fig8_backbone")


def main():
    for name, fn in [("fig0_pipeline", fig_pipeline), ("fig1_steps", fig_steps), ("fig2_speed_confusion", fig_speed_confusion),
                     ("fig3_adaptation", fig_adaptation), ("fig4_latency", fig_latency),
                     ("fig5_robustness", fig_robustness), ("fig6_synthetic", fig_synthetic),
                     ("fig7_examples", fig_examples), ("fig8_backbone", fig_backbone)]:
        print(name, "...")
        try:
            fn()
        except Exception:
            print(f"  FAILED: {name}")
            traceback.print_exc()
    print(f"done: figures in {OUT.resolve()}")


if __name__ == "__main__":
    main()
