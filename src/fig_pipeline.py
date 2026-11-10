"""Fig. 1 (Intelligent Computing version): pipeline overview, drawn in code (vector PDF + 600 dpi PNG)."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle, Polygon, RegularPolygon
import sys
import numpy as np

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['TeX Gyre Termes', 'Times New Roman', 'Liberation Serif', 'DejaVu Serif']
plt.rcParams['pdf.fonttype'] = 42

GRAY, BLUE, ORANGE, SLATE = '#7d8794', '#2b6cb0', '#d98a2b', '#4a5563'
RED, SIGNBLUE, INK, LIGHT = '#d62728', '#1f5fae', '#2b2b2b', '#dbe6f3'
W, H = 2000, 1073
fig = plt.figure(figsize=(10, 10 * H / W))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.axis('off')

def box(x0, y0, x1, y1, color):
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle='round,pad=0,rounding_size=18',
                                fc=color, ec='none', zorder=1))
def panel(x0, y0, s=152):
    ax.add_patch(FancyBboxPatch((x0, y0), s, s, boxstyle='round,pad=0,rounding_size=10', fc='white', ec='none', zorder=2))
def label(x, yc, title, sub, tsize=12.5, ssize=8.8, maxw=None):
    ax.text(x, yc - 8, title, color='white', fontsize=tsize, fontweight='bold', va='bottom', ha='left', zorder=3, linespacing=1.0)
    ax.text(x, yc + 8, sub, color='white', fontsize=ssize, fontweight='bold', va='top', ha='left', zorder=3, linespacing=1.1)
def sign(x, y, r, text='', ring=RED, fs=7):
    ax.add_patch(Circle((x, y), r, fc='white', ec=ring, lw=2.2 * r / 20, zorder=4))
    if text: ax.text(x, y, text, ha='center', va='center', fontsize=fs, fontweight='bold', color=INK, zorder=5)
def disk(x, y, r, arrow):
    ax.add_patch(Circle((x, y), r, fc=SIGNBLUE, ec='none', zorder=4))
    ax.text(x, y, arrow, ha='center', va='center', fontsize=r / 2.3, color='white', zorder=5, family='DejaVu Sans')
def tri(x, y, s):
    ax.add_patch(RegularPolygon((x, y + 2), 3, radius=s, orientation=np.pi, fc='white', ec=RED, lw=2, zorder=4))
def road(x0, y0, w=136, h=116, tag=None, play=False, sign_bbox=False):
    ax.add_patch(Rectangle((x0, y0), w, h * 0.55, fc='#cfe2f3', ec='none', zorder=3))
    ax.add_patch(Rectangle((x0, y0 + h * 0.55), w, h * 0.45, fc='#c3d3b0', ec='none', zorder=3))
    ax.add_patch(Polygon([(x0 + w * 0.40, y0 + h * 0.55), (x0 + w * 0.60, y0 + h * 0.55), (x0 + w * 0.78, y0 + h), (x0 + w * 0.22, y0 + h)],
                         fc='#6b6f75', ec='none', zorder=3))
    for k in range(2):
        yy = y0 + h * (0.62 + 0.18 * k)
        ax.plot([x0 + w / 2, x0 + w / 2], [yy, yy + h * 0.09], color='white', lw=1.5, zorder=4)
    if play:
        ax.add_patch(Circle((x0 + w / 2, y0 + h * 0.32), h * 0.22, fc='white', ec='none', zorder=4))
        ax.add_patch(Polygon([(x0 + w / 2 - 9, y0 + h * 0.32 - 12), (x0 + w / 2 - 9, y0 + h * 0.32 + 12), (x0 + w / 2 + 14, y0 + h * 0.32)],
                             fc=SIGNBLUE if not tag else INK, ec='none', zorder=5))
    if sign_bbox:
        sx, sy = x0 + w * 0.80, y0 + h * 0.28
        ax.plot([sx, sx], [sy, y0 + h * 0.85], color=INK, lw=1.2, zorder=4)
        sign(sx, sy, 14)
        ax.add_patch(Rectangle((sx - 21, sy - 21), 42, 42, fc='none', ec=ORANGE, lw=1.4, ls=(0, (3, 2)), zorder=5))
    if tag:
        ax.add_patch(Rectangle((x0, y0 + h * 0.78), w, h * 0.22, fc=SLATE, ec='none', zorder=5))
        ax.text(x0 + w / 2, y0 + h * 0.89, tag, ha='center', va='center', color='white', fontsize=8, fontweight='bold', zorder=6)
def arrow(x0, y0, x1, y1, dashed=False, z=1):
    ax.annotate('', xy=(x1, y1), xytext=(x0, y0), zorder=z,
                arrowprops=dict(arrowstyle='-|>', color=SLATE, lw=1.6, ls='--' if dashed else '-', mutation_scale=12, shrinkA=0, shrinkB=0))

C = [(52, 488), (532, 968), (1012, 1448), (1492, 1928)]
R = [(52, 248), (307, 503), (562, 758), (817, 1013)]
def cell(ci, ri, color, title, sub, x1=None, x0=None):
    a, b = C[ci]; y0, y1 = R[ri]
    a = x0 or a; b = x1 or b
    box(a, y0, b, y1, color); panel(a + 22, y0 + 22)
    label(a + 193, (y0 + y1) / 2, title, sub)
    return a + 22, y0 + 22

# row 1: data sources
px, py = cell(0, 0, GRAY, 'TS-1M\n(worldwide)', '70 Morocco-relevant\nclasses')
for i, (kind, t) in enumerate([('s', '30'), ('d', '→'), ('t', ''), ('t', ''), ('s', '50'), ('d', '↑'), ('s', ''), ('d', '←'), ('s', '80')]):
    cx, cy = px + 30 + (i % 3) * 46, py + 30 + (i // 3) * 46
    if kind == 's': sign(cx, cy, 19, t)
    elif kind == 'd': disk(cx, cy, 18, t)
    else: tri(cx, cy, 22)
px, py = cell(1, 0, GRAY, 'Kaggle photos', 'Moroccan set,\n2,676 crops'); road(px + 8, py + 18, sign_bbox=True)
px, py = cell(2, 0, GRAY, '3 training\ndrives', '868 own crops'); road(px + 8, py + 18, play=True)
px, py = cell(3, 0, SLATE, '4 development\ndrives', '696 crops,\nmodel selection'); road(px + 8, py + 18, play=True, tag='DEV')
# row 2
px, py = cell(0, 1, BLUE, 'label cleaning', 'Confident Learning\n+ human check')
disk(px + 42, py + 56, 30, '←'); disk(px + 110, py + 56, 30, '←')
ax.text(px + 42, py + 106, '"right"', ha='center', fontsize=7.5, color=INK, zorder=5)
ax.plot([px + 22, px + 62], [py + 106, py + 106], color=RED, lw=1, zorder=6)
ax.text(px + 110, py + 106, '"left"', ha='center', fontsize=7.5, color=INK, zorder=5)
ax.text(px + 42, py + 132, '✗', ha='center', va='center', fontsize=10, color=RED, zorder=5, family='DejaVu Sans')
ax.text(px + 110, py + 132, '✓', ha='center', va='center', fontsize=10, color=SIGNBLUE, zorder=5, family='DejaVu Sans')
px, py = cell(2, 1, BLUE, 'synthetic\nsigns', 'drawn signs +\ndigit swaps', x1=1358)
sign(px + 36, py + 64, 30, '60', fs=11); sign(px + 116, py + 64, 30, '80', fs=11)
arrow(px + 67, py + 64, px + 85, py + 64, z=6)
ax.text(px + 76, py + 124, 'digit swap', ha='center', fontsize=7.5, style='italic', color=INK, zorder=5)
px, py = cell(3, 1, SLATE, '2 final-test\ndrives', '423 crops, labelled\nblind, scored once'); road(px + 8, py + 18, play=True, tag='FINAL')
# row 3
px, py = cell(0, 2, BLUE, 'train tiny CNN', '40 K weights,\n32 or 48 px input')
for k, (s, off) in enumerate([(42, 0), (30, 56), (20, 98)]):
    for j in range(3):
        ax.add_patch(Rectangle((px + 10 + off + j * 5, py + 60 - j * 5), s, s, fc=LIGHT, ec=SIGNBLUE, lw=1, zorder=4 + j))
a, b = 532, 1448; y0, y1 = R[2]
box(a, y0, b, y1, BLUE); panel(a + 22, y0 + 22)
label(a + 193, (y0 + y1) / 2, 'fine-tune on Moroccan data', '15% local crops per batch, at most 40 per class, mirror copies')
px, py = a + 22, y0 + 22
ax.text(px + 76, py + 26, 'one batch', ha='center', fontsize=7.5, style='italic', color=INK, zorder=5)
for j in range(8):
    ax.add_patch(Rectangle((px + 8 + j * 17, py + 46), 13, 46, fc=ORANGE if j == 7 else '#c8ccd2', ec='none', zorder=4))
ax.text(px + 50, py + 116, 'world', ha='center', fontsize=7.5, color=INK, zorder=5)
ax.text(px + 128, py + 116, 'local', ha='center', fontsize=7.5, color=INK, fontweight='bold', zorder=5)
# row 4
px, py = cell(1, 3, ORANGE, 'int8 conversion', '58.7 KB model file')
ax.add_patch(Rectangle((px + 6, py + 28), 140, 40, fc='#eceef1', ec='none', zorder=4))
ax.text(px + 76, py + 48, '0.4172', ha='center', va='center', fontsize=10, color=INK, zorder=5)
arrow(px + 76, py + 72, px + 76, py + 93, z=6)
ax.add_patch(Rectangle((px + 46, py + 96), 60, 38, fc=ORANGE, ec='none', zorder=4))
ax.text(px + 76, py + 115, '53', ha='center', va='center', fontsize=10, color='white', fontweight='bold', zorder=5)
px, py = cell(2, 3, ORANGE, 'STM32 / ESP32', '1.8–189 ms per image')
ax.add_patch(FancyBboxPatch((px + 36, py + 36), 80, 80, boxstyle='round,pad=0,rounding_size=8', fc='#2d3138', ec='none', zorder=5))
for k in range(4):
    for (dx, dy, w_, h_) in [(44 + k * 18, 22, 7, 14), (44 + k * 18, 116, 7, 14), (22, 44 + k * 18, 14, 7), (116, 44 + k * 18, 14, 7)]:
        ax.add_patch(Rectangle((px + dx, py + dy), w_, h_, fc='#9aa1aa', ec='none', zorder=4))
ax.text(px + 76, py + 76, 'MCU', ha='center', va='center', fontsize=8.5, color='white', fontweight='bold', zorder=6)
px, py = cell(3, 3, SLATE, 'test on drives', 'development and\nfinal test')
sign(px + 76, py + 48, 32, '60', fs=11)
ax.text(px + 76, py + 104, 'dev 87.5%', ha='center', va='center', fontsize=9, color=INK, fontweight='bold', zorder=5)
ax.text(px + 76, py + 128, 'final 95.1%', ha='center', va='center', fontsize=9, color=INK, fontweight='bold', zorder=5)
# arrows
arrow(270, 252, 270, 303); arrow(270, 507, 270, 558)
arrow(750, 252, 750, 558)
arrow(1185, 252, 1185, 303); arrow(1185, 507, 1185, 558)
arrow(1403, 252, 1403, 558); ax.text(1412, 400, 'own\ncrops', fontsize=9, color=INK, va='center')
arrow(492, 660, 528, 660)
arrow(750, 762, 750, 813)
arrow(972, 915, 1008, 915); arrow(1452, 915, 1488, 915)
arrow(1710, 252, 1710, 303, dashed=True); arrow(1710, 507, 1710, 813, dashed=True)
ax.text(1728, 650, 'never used\nfor training', fontsize=9.5, color=INK, va='center')
out = sys.argv[1] if len(sys.argv) > 1 else 'Fig1_pipeline'
fig.savefig(out + '.pdf'); fig.savefig(out + '.png', dpi=600)
print('saved', out)
