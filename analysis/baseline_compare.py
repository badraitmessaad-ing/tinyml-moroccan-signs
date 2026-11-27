"""Matched baselines evaluated exactly like our model (R1.4, R2.4): the inverted-residual network and the plain
CNN (same weights budget, data, recipe and int8 conversion, trained after our models were frozen) on every drive,
with the same statistics: accuracy over 3 seeds, 95% CI by bootstrap over physical signs, macro-F1, speed limits,
pooled held-out accuracy, paired difference to our model, error severity with and without the tau = 0.3 rule,
calibration (ECE) and risk-coverage (AURC).
Needs preds_<set>.npz (ours) and preds_<set>_ir.npz / preds_<set>_plain.npz (dump_predictions.py --model ir|plain)."""
import numpy as np, collections, warnings
from sklearn.metrics import f1_score
from sign_groups import load, correct, boot
from sign_rule import group_rule
warnings.filterwarnings('ignore')
src = open('safety_tau.py').read()
exec(src.split("mode=sys.argv")[0])          # SPEED, kind(), sm(), decode() from the safety scripts
SPEED_IDS = {11, 17, 23, 28, 32, 36, 39, 41, 46}
SETS = ('dev', 'final', 'dashcam', 'rain_rabat', 'rain_tanger')
ROLE = {'dev': 'development', 'final': 'held-out', 'dashcam': 'held-out', 'rain_rabat': 'held-out', 'rain_tanger': 'held-out'}
NAME = {'fes_1': 'Fes 1', 'fes_2': 'Fes 2', 'fes_rabat_motorway': 'Fes-Rabat motorway (rain, roadwork)', 'rabat': 'Rabat',
        'fes_ifrane': 'Fes-Ifrane', 'marrakech': 'Marrakech', 'tanger': 'Tangier, own dashcam (dry)',
        'Rabat-rain day': 'Rabat, dashcam 2 (rain)', 'tanger-rain day': 'Tangier, dashcam 3 (rain)'}
MODELS = [('', 'ours (depthwise-separable)'), ('_ir', 'inverted residual'), ('_plain', 'plain CNN')]
rng = np.random.default_rng(0)
data = {}
for suf, _ in MODELS:
    for n in SETS:
        d, cl, rows = load(n, suf); data[(suf, n)] = (d, cl, rows, correct(d, cl, rows))
# sign ids and drives come from the crops, identical for every model
meta = {n: (group_rule(data[('', n)][2]), np.array([r['drive'] for r in data[('', n)][2]]), np.array([r['y'] for r in data[('', n)][2]])) for n in SETS}
print("== Per drive: accuracy (mean ± SD over 3 seeds) [95% CI over physical signs] | macro-F1 | speed limits")
for n in SETS:
    sid, drive, y = meta[n]; ins = y >= 0
    for dr in sorted(set(drive[ins])):
        m = ins & (drive == dr)
        print(f"{ROLE[n]:<12}{NAME.get(dr, dr)} ({m.sum()} crops)")
        for suf, lab in MODELS:
            d, cl, rows, C = data[(suf, n)]; ids = np.array([int(c[:4]) for c in cl]); a = C[:, m].mean(1) * 100
            lo, hi = boot(C[:, m].mean(0), sid[m], rng)
            f1 = np.mean([f1_score(y[m], np.where(C[s, m] > 0, y[m], d['logits'][s][m].argmax(-1)), labels=np.unique(y[m]),
                                   average='macro', zero_division=0) for s in range(3)]) * 100
            sp = m & np.array([ids[t] in SPEED_IDS if t >= 0 else False for t in y])
            spt = f"{C[:, sp].mean() * 100:.1f} ({sp.sum()})" if sp.sum() else "-"
            print(f"      {lab:<28} {a.mean():5.1f} ± {a.std():3.1f}  [{lo:5.1f}-{hi:5.1f}]   F1 {f1:5.1f}   speed {spt}")
print("\n== Pooled held-out (873 in-scope crops) and paired difference to our model (bootstrap over physical signs)")
acc = {}; sids = []
for n in SETS[1:]:
    sid, drive, y = meta[n]; ins = y >= 0; sids.append(np.array([f"{n}|{u}" for u in sid[ins]]))
sids = np.concatenate(sids)
for suf, lab in MODELS:
    acc[suf] = np.concatenate([data[(suf, n)][3][:, meta[n][2] >= 0] for n in SETS[1:]], 1)
    a = acc[suf].mean(1) * 100; lo, hi = boot(acc[suf].mean(0), sids, rng)
    print(f"   {lab:<28} {a.mean():5.1f} ± {a.std():3.1f}  [{lo:5.1f}-{hi:5.1f}]")
U = np.unique(sids); g = {u: np.where(sids == u)[0] for u in U}
for suf, lab in MODELS[1:]:
    diff = acc[''].mean(0) - acc[suf].mean(0); bs = []
    for _ in range(2000):
        ii = np.concatenate([g[u] for u in rng.choice(U, len(U))]); bs.append(diff[ii].mean() * 100)
    print(f"   ours - {lab:<21} {diff.mean() * 100:+5.1f} points  [{np.percentile(bs, 2.5):+5.1f} to {np.percentile(bs, 97.5):+5.1f}]")
print("\n== Error severity (errors per seed) and dangerous rate, without / with the tau = 0.3 rule (tuned on our model's dev set)")
for n in SETS:
    y = meta[n][2]; ins = y >= 0
    for suf, lab in MODELS:
        d, cl, rows, C = data[(suf, n)]; ids = np.array([int(c[:4]) for c in cl]); out = []
        for tau in (None, 0.3):
            cnt = collections.Counter(); acc_t = []
            for s in range(3):
                pr = decode(sm(d['logits'][s].astype(float)), ids, tau, 'prob'); ok = 0
                for k in np.where(ins)[0]:
                    kd = kind(ids[y[k]], ids[pr[k]]); cnt[kd] += 1; ok += kd == 'correct'
                acc_t.append(ok / ins.sum() * 100)
            out.append((cnt['dangerous'] / 3 / ins.sum() * 100, np.mean(acc_t), cnt))
        c0 = out[0][2]
        print(f"{n:<12}{lab:<28} dangerous {c0['dangerous']/3:5.1f} safe-side {c0['safe-side']/3:4.1f} same action {c0['same action']/3:4.1f} other {c0['other']/3:5.1f}"
              f" | dangerous rate {out[0][0]:.1f}% -> {out[1][0]:.1f}% with tau (accuracy {out[0][1]:.1f} -> {out[1][1]:.1f})")
print("\n== Calibration (ECE, 10 bins) and risk-coverage (AURC), in-scope crops, 3 seeds pooled; rain drives pooled")
SAMEG = {218: 221, 221: 218}
for grp, parts in (('dev', ['dev']), ('final', ['final']), ('dashcam', ['dashcam']), ('rain', ['rain_rabat', 'rain_tanger'])):
    for suf, lab in MODELS:
        conf, ok = [], []
        for n in parts:
            d, cl, rows, C = data[(suf, n)]; y = meta[n][2]; ins = y >= 0
            for s in range(3):
                P = sm(d['logits'][s].astype(float))[ins]; conf.append(P.max(-1)); ok.append(C[s, ins])
        conf = np.concatenate(conf); ok = np.concatenate(ok).astype(float)
        idx = np.clip(np.digitize(conf, np.linspace(0, 1, 11)) - 1, 0, 9)
        ece = sum(abs(conf[idx == b].mean() - ok[idx == b].mean()) * (idx == b).sum() for b in range(10) if (idx == b).sum()) / len(conf)
        o = np.argsort(-conf); risk = np.cumsum(1 - ok[o]) / np.arange(1, len(o) + 1); aurc = risk.mean()
        print(f"{grp:<10}{lab:<28} ECE {ece*100:4.1f}%  mean confidence {conf.mean():.3f} vs accuracy {ok.mean():.3f}  AURC {aurc*100:.2f}%")
print("\nOn-device cost (STM32G474, ST Edge AI Core 4.1, Section 3.9): ours 42.8 ms / 24.3 KB RAM; inverted residual 86.4 ms / 56.5 KB; plain CNN 56.2 ms / 17.7 KB.")
