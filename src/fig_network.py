"""Fig. 2 (Intelligent Computing version): network architecture, drawn in code."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle
import sys

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['TeX Gyre Termes', 'Times New Roman', 'Liberation Serif', 'DejaVu Serif']
plt.rcParams['pdf.fonttype'] = 42
GRAY, BLUE, ORANGE, SLATE, RED, INK, LIGHT = '#9aa3ae', '#2b6cb0', '#d98a2b', '#4a5563', '#d62728', '#2b2b2b', '#dbe6f3'
W, H = 2000, 770
fig = plt.figure(figsize=(10, 10 * H / W))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.axis('off')

ax.text(20, 34, 'DW, depthwise convolution; PW, pointwise convolution; /2, stride 2; sizes for a 48 × 48 input (32 × 32 input in brackets)',
        fontsize=9.5, color=INK, va='center')
cols = [
    ('input', 'RGB', GRAY, '48 × 48 × 3', '(32 × 32 × 3)', 'the sign image', SLATE),
    ('conv 3 × 3 /2', 'ReLU6', BLUE, '24 × 24 × 16', '(16 × 16 × 16)', 'edges and colours', BLUE),
    ('DW 3 × 3\nPW 1 × 1', 'ReLU6', BLUE, '24 × 24 × 32', '(16 × 16 × 32)', 'corners, curves', BLUE),
    ('DW 3 × 3 /2\nPW 1 × 1', 'ReLU6', BLUE, '12 × 12 × 64', '(8 × 8 × 64)', 'red ring, digit parts', BLUE),
    ('DW 3 × 3\nPW 1 × 1', 'ReLU6', BLUE, '12 × 12 × 96', '(8 × 8 × 96)', 'digits and symbols', BLUE),
    ('DW 3 × 3 /2\nPW 1 × 1', 'ReLU6', BLUE, '6 × 6 × 160', '(4 × 4 × 160)', 'the whole sign', BLUE),
    ('average\npooling', 'features', ORANGE, '160', '', 'one value per map', ORANGE),
    ('linear', 'class scores', ORANGE, '70', '', 'best class: 60 km/h', ORANGE),
]
x0, w, gap, y0, y1 = 20, 208, 40.3, 95, 560
for i, (title, sub, col, size, size32, desc, dcol) in enumerate(cols):
    x = x0 + i * (w + gap); cx = x + w / 2
    ax.add_patch(FancyBboxPatch((x, y0), w, y1 - y0, boxstyle='round,pad=0,rounding_size=16', fc=col, ec='none', zorder=1))
    ax.add_patch(FancyBboxPatch((x + 16, y0 + 16), w - 32, 194, boxstyle='round,pad=0,rounding_size=10', fc='white', ec='none', zorder=2))
    ax.text(cx, 378, title, ha='center', va='center', color='white', fontsize=13, fontweight='bold', zorder=3, linespacing=1.05)
    ax.text(cx, 497, sub, ha='center', va='center', color='white', fontsize=11.5, fontweight='bold', zorder=3)
    ax.text(cx, 610, size, ha='center', va='center', color=INK, fontsize=11.5, zorder=3)
    if size32: ax.text(cx, 660, size32, ha='center', va='center', color=INK, fontsize=10, zorder=3)
    ax.text(cx, 709, desc, ha='center', va='center', color=INK, fontsize=10, style='italic', zorder=3)
    if i < len(cols) - 1:
        ax.annotate('', xy=(x + w + gap - 4, 327), xytext=(x + w + 4, 327),
                    arrowprops=dict(arrowstyle='-|>', color=SLATE, lw=1.4, mutation_scale=11, shrinkA=0, shrinkB=0))
    px, py = x + 16, y0 + 16; pc = (px + (w - 32) / 2, py + 97)
    if i == 0:
        ax.add_patch(Circle(pc, 62, fc='white', ec=RED, lw=5, zorder=4))
        ax.text(*pc, '60', ha='center', va='center', fontsize=16, fontweight='bold', color=INK, zorder=5)
    elif i == 1:
        for a in range(6):
            for b in range(6):
                ax.add_patch(Rectangle((px + 18 + a * 23, py + 28 + b * 23), 23, 23, fc='#e6ebf2' if (a + b) % 2 else '#f4f6f9', ec='white', lw=0.8, zorder=3))
        ax.add_patch(Rectangle((px + 41, py + 51), 69, 69, fc='none', ec=RED, lw=2.2, zorder=4))
    elif i in (2, 3, 4, 5):
        n = {2: 3, 3: 4, 4: 5, 5: 6}[i]; s = {2: 96, 3: 78, 4: 78, 5: 58}[i]; step = 8
        bx = pc[0] - s / 2 - (n - 1) * step / 2; by = pc[1] - s / 2 + (n - 1) * step / 2
        for j in reversed(range(n)):
            ax.add_patch(Rectangle((bx + j * step, by - j * step), s, s, fc=LIGHT, ec=BLUE, lw=1.2, zorder=4 + (n - j)))
    elif i == 6:
        for j in reversed(range(3)):
            ax.add_patch(Rectangle((px + 14 + j * 8, py + 70 - j * 8), 50, 50, fc='#fde9d4', ec=ORANGE, lw=1.2, zorder=4 + (3 - j)))
        ax.annotate('', xy=(px + 118, py + 72), xytext=(px + 90, py + 72), zorder=6,
                    arrowprops=dict(arrowstyle='-|>', color=SLATE, lw=1.2, mutation_scale=9, shrinkA=0, shrinkB=0))
        for j in range(5):
            ax.add_patch(Rectangle((px + 128, py + 26 + j * 28), 28, 28, fc=ORANGE if j == 2 else '#fde9d4', ec=ORANGE, lw=1.2, zorder=4))
    elif i == 7:
        for j, (lab, v) in enumerate([('60', 1.0), ('80', 0.48), ('100', 0.27), ('stop', 0.12)]):
            yy = py + 36 + j * 40
            ax.text(px + 44, yy, lab, ha='right', va='center', fontsize=9, color=INK, fontweight='bold' if j == 0 else 'normal', zorder=4)
            ax.add_patch(Rectangle((px + 52, yy - 12), 100 * v, 24, fc=ORANGE if j == 0 else '#c8ccd2', ec='none', zorder=4))
out = sys.argv[1] if len(sys.argv) > 1 else 'Fig2_network'
fig.savefig(out + '.pdf'); fig.savefig(out + '.png', dpi=600)
print('saved', out)
