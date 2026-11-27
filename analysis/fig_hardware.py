"""Hardware figure (R1.5, R1.6, R2.1): the ESP32-CAM bench test and the inline USB power meter, with what the
model saw on the chip. Inputs: figures/photos/setup_side.jpg (phone photo of the camera bench test), usb_meter.jpg (photo of the
USB power meter during the energy measurement), esp32_board.jpg (the ESP32 board used for every ESP32 result),
camera_test/seen/nearcm_#NN.png (48x48 frames saved by camera_test.py) and camera_test/results_nearcm.csv.
Writes figures/fig_hardware.png and .pdf. Run from revision_analysis/."""
import csv, collections
from pathlib import Path
import numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from PIL import Image, ImageOps
ROOT = Path('..'); PH = ROOT / 'figures' / 'photos'; CT = ROOT / 'camera_test'
plt.rcParams.update({'font.family': 'serif', 'font.size': 8})
def photo(name, box, rot=0):
    im = ImageOps.exif_transpose(Image.open(PH / name)).convert('RGB').crop(box)
    return np.asarray(im.rotate(rot, expand=True) if rot else im)
def label(ax, text, xy, xytext):
    ax.annotate(text, xy=xy, xytext=xytext, fontsize=7.5, color='black',
                bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='0.3', lw=0.6, alpha=0.92),
                arrowprops=dict(arrowstyle='->', color='white', lw=1.6, shrinkA=0, shrinkB=2))
R = list(csv.DictReader(open(CT / 'results_nearcm.csv', encoding='utf-8')))
SHOW = [5, 11, 36, 40, 26, 12]                     # 4 correct and 2 wrong signs (wrong ones from framing / similar sign)
NICE = {'0028': '60 km/h', '0135': 'yield', '0342': 'roundabout', '0444': 'parking', '0436': 'ped. crossing',
        '0243': 'pedestrians', '0142': 'stop', '0011': '30 km/h', '0244': 'children', '0140': 'priority'}
def nice(c): return NICE.get(c[:4], ' '.join(c[5:].replace('_', ' ').lower().split()[:2]))
fig = plt.figure(figsize=(7.2, 3.9))
gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 0.9], width_ratios=[2500 / 1153, 660 / 1000, 2212 / 1659], hspace=0.24, wspace=0.04)
# (a) bench test, side view
a = fig.add_subplot(gs[0, 0]); a.imshow(photo('setup_side.jpg', (0, 0, 2500, 1153))); a.axis('off')
label(a, 'ESP32-CAM\n(OV2640, Wi-Fi off)', (1545, 760), (1480, 300))
label(a, 'printed sign\n(A4, "near")', (430, 420), (720, 640))
label(a, 'USB to PC\n(camera_test.py)', (1600, 980), (1000, 1010))
a.set_title('a  Camera-in-the-loop bench test', loc='left', fontsize=9)
# (b) the ESP32 board used for every ESP32 result (timing, PC agreement, energy)
b = fig.add_subplot(gs[0, 1]); b.imshow(photo('esp32_board.jpg', (560, 760, 1220, 1760), rot=180)); b.axis('off')
b.set_title('b  ESP32', loc='left', fontsize=9)
# (c) the USB power meter during the energy measurement of that board
c = fig.add_subplot(gs[0, 2]); c.imshow(photo('usb_meter.jpg', (0, 0, 2212, 1659))); c.axis('off')
label(c, 'USB power meter', (1100, 560), (90, 1480))
label(c, 'to ESP32 board', (2150, 620), (1250, 1080))
c.set_title('c  Power meter', loc='left', fontsize=9)
# (d) what the model saw (48x48 on the chip)
inner = gs[1, :].subgridspec(1, len(SHOW), wspace=0.08)
for k, s in enumerate(SHOW):
    ax = fig.add_subplot(inner[0, k])
    ax.imshow(Image.open(CT / 'seen' / f'nearcm_#{s:02d}.png').convert('RGB'), interpolation='nearest'); ax.set_xticks([]); ax.set_yticks([])
    rs = [r for r in R if int(r['sign']) == s]; truth = rs[0]['truth']
    vote, n = collections.Counter(r['pred'] for r in rs).most_common(1)[0]
    ok = vote[:4] == truth[:4]
    for sp in ax.spines.values(): sp.set_edgecolor('#2a8a3a' if ok else '#c0392b'); sp.set_linewidth(2.0)
    ax.set_xlabel(f"{nice(truth)}\n→ {nice(vote)} ({n}/5)", fontsize=7, color='black' if ok else '#c0392b')
    if k == 0: ax.set_title('d  What the model saw on the chip (48 × 48), true class → 5-frame vote', loc='left', fontsize=9)
for ext in ('png', 'pdf'): fig.savefig(ROOT / 'figures' / f'fig_hardware.{ext}', dpi=300, bbox_inches='tight')
print('written figures/fig_hardware.png and .pdf')
