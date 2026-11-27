"""Per-drive table for the manuscript: every development and held-out drive, frozen models (3 seeds).
Accuracy (mean ± SD over seeds), 95% CI by bootstrap over physical signs (2 per sign, 10 s rule) and over
screenshots, macro-F1, speed-limit accuracy, number of crops, signs and screenshots."""
import numpy as np, collections, warnings
from sklearn.metrics import f1_score
from sign_groups import load, correct, boot
from sign_rule import group_rule
warnings.filterwarnings('ignore')
SPEED_IDS = {11, 17, 23, 28, 32, 36, 39, 41, 46}
NAME = {'fes_1': 'Fes 1', 'fes_2': 'Fes 2', 'fes_rabat_motorway': 'Fes-Rabat motorway (rain, roadwork)', 'rabat': 'Rabat',
        'fes_ifrane': 'Fes-Ifrane', 'marrakech': 'Marrakech', 'tanger': 'Tangier, own dashcam (dry)',
        'Rabat-rain day': 'Rabat, dashcam 2 (rain)', 'tanger-rain day': 'Tangier, dashcam 3 (rain)'}
ROLE = {'dev': 'development', 'final': 'held-out', 'dashcam': 'held-out', 'rain_rabat': 'held-out', 'rain_tanger': 'held-out'}
rng = np.random.default_rng(0)
print(f"{'set':<12}{'drive':<38}{'crops':>6}{'signs':>6}{'shots':>6}  {'accuracy':>11}  {'sign CI':>11}  {'shot CI':>11}  {'macro-F1':>8}  speed limits")
pool = collections.defaultdict(list)
for n in ('dev', 'final', 'dashcam', 'rain_rabat', 'rain_tanger'):
    d, cl, rows = load(n); C = correct(d, cl, rows); ids = np.array([int(c[:4]) for c in cl])
    y = np.array([r['y'] for r in rows]); ins = y >= 0
    sid_all = group_rule(rows); drive = np.array([r['drive'] for r in rows]); shot = np.array([r['shot'] for r in rows])
    for dr in sorted(set(drive[ins])):
        m = ins & (drive == dr); acc = C[:, m]; a = acc.mean(1) * 100
        lo, hi = boot(acc.mean(0), sid_all[m], rng); slo, shi = boot(acc.mean(0), shot[m], rng)
        # predicted label, with the bump/uneven-road pair counted as correct (same as accuracy)
        f1 = np.mean([f1_score(y[m], np.where(C[s, m] > 0, y[m], d['logits'][s][m].argmax(-1)),
                               labels=np.unique(y[m]), average='macro', zero_division=0) for s in range(3)]) * 100
        sp = m & np.array([ids[t] in SPEED_IDS if t >= 0 else False for t in y])
        spt = f"{C[:, sp].mean() * 100:.1f} ({sp.sum()})" if sp.sum() else "-"
        print(f"{ROLE[n]:<12}{NAME.get(dr, dr):<38}{m.sum():>6}{len(set(sid_all[m])):>6}{len(set(shot[m])):>6}  {a.mean():5.1f} ± {a.std():3.1f}  {lo:5.1f}-{hi:5.1f}  {slo:5.1f}-{shi:5.1f}  {f1:8.1f}  {spt}")
        pool[ROLE[n]].append((C[:, m], sid_all[m], n + dr))
for role, parts in pool.items():
    acc = np.concatenate([p[0] for p in parts], 1); sid = np.concatenate([np.array([f"{p[2]}|{u}" for u in p[1]]) for p in parts])
    a = acc.mean(1) * 100; lo, hi = boot(acc.mean(0), sid, rng)
    print(f"pooled {role:<43}{acc.shape[1]:>6}{len(set(sid)):>6}{'':>6}  {a.mean():5.1f} ± {a.std():3.1f}  {lo:5.1f}-{hi:5.1f}   (pooled numbers are not an estimate for Moroccan roads in general)")
