"""Camera-in-the-loop test on the ESP32-CAM (R1.5 / R2): summary of camera_test\\results_nearcm.csv,
compared with the same 40 crops classified digitally (no camera) by the same frozen model."""
import csv, collections, numpy as np
exec(open('safety_groups.py').read().split('for n in')[0])      # SPEED, groups, kind()
R = list(csv.DictReader(open('../camera_test/results_nearcm.csv', encoding='utf-8')))
key = list(csv.DictReader(open('../camera_test/signs_key.csv', encoding='utf-8')))
signs = sorted({int(r['sign']) for r in R})
ok = np.array([int(r['correct']) for r in R])
print(f"{len(signs)} printed signs x 5 frames = {len(R)} frames")
print(f"per-frame accuracy: {ok.mean()*100:.1f}% ({ok.sum()}/{len(R)})")
votes = {}
for s in signs:
    rs = [r for r in R if int(r['sign']) == s]
    votes[s] = (rs[0]['truth'], collections.Counter(r['pred'] for r in rs).most_common(1)[0][0], sum(int(r['correct']) for r in rs))
vok = [kind(int(t[:4]), int(v[:4])) == 'correct' for t, v, _ in votes.values()]
print(f"5-frame vote accuracy: {np.mean(vok)*100:.1f}% ({sum(vok)}/{len(vok)})")
kinds = collections.Counter(kind(int(r['truth'][:4]), int(r['pred'][:4])) for r in R)
print("frame error types:", dict(kinds))
T = np.array([[float(r['capture_ms']), float(r['prep_ms']), float(r['model_ms'])] for r in R])
print(f"time per frame, median: capture {np.median(T[:,0]):.0f} ms, preprocessing {np.median(T[:,1]):.1f} ms, model {np.median(T[:,2]):.1f} ms")
p = np.array([float(r['prob']) for r in R])
print(f"mean confidence: correct {p[ok==1].mean():.2f}, wrong {p[ok==0].mean():.2f}")
for th in (0.7, 0.9):
    m = p >= th; print(f"  confidence >= {th}: keeps {m.mean()*100:.1f}% of frames at {ok[m].mean()*100:.1f}%")
d = np.load('preds_dashcam.npz'); cl = list(d['classes'])
idx = {(f, o): k for k, (f, o) in enumerate(zip(d['folder'], d['files']))}
print("same 40 crops, digital (no camera), per seed:")
for s in range(3):
    dig = [cl[d['logits'][s][idx[(r['class_folder'], r['source_crop'])]].argmax()] == r['class_folder'] for r in key]
    print(f"  {str(d['runs'][s]).split('/')[-1]}: {sum(dig)}/40")
print("signs wrong by vote:")
for s, (t, v, c) in votes.items():
    if kind(int(t[:4]), int(v[:4])) != 'correct':
        print(f"  #{s:02d} {t[5:]:<36} vote {v[5:]:<34} {c}/5 frames correct  ({kind(int(t[:4]), int(v[:4]))})")
