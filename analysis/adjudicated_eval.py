"""Sensitivity analysis with corrected labels (R1.2).
Default (--rule annotator2): every crop on which the two annotators disagree takes the SECOND annotator's label
(a fixed rule, independent of the model's answers; the first annotator confirmed these labels on re-inspection);
'cannot tell' answers keep the original label. Option --rule adjudicator: use a blind third adjudicator instead
(adjudication_key.csv + adjudication_choices.csv from make_adjudication_pack.py).
Original description: The main results keep the original labels, fixed before
scoring. Here every crop on which the two annotators disagreed is given the label chosen by a blind third
adjudicator (adjudication_key.csv + adjudication_choices.csv in the project folder); 'Neither' makes the crop an
out-of-scope sign and 'Cannot tell' removes it. Accuracy and dangerous-error rate are recomputed for our model
and the two baselines. Run from revision_analysis/ after make_adjudication_pack.py and the adjudicator's file."""
import csv, collections, numpy as np
from sign_groups import load, boot
from sign_rule import group_rule
exec(open('safety_tau.py').read().split("mode=sys.argv")[0])
import sys
RULE = sys.argv[sys.argv.index('--rule') + 1] if '--rule' in sys.argv else 'annotator2'
new = {}; tally = collections.Counter()
def fkey(path): p = path.split('/'); return (p[0], p[2], p[3])
if RULE == 'annotator2':
    for kf, lf in (('../annotator_key.csv', '../labels_annotator2.csv'), ('../annotator_key_part2.csv', '../labels_annotator2_part2.csv')):
        lab = {r['item']: r['label'] for r in csv.DictReader(open(lf, encoding='utf-8')) if r['label']}
        for r in csv.DictReader(open(kf, encoding='utf-8')):
            l2 = lab.get(r['item'], '')
            if not l2 or l2 == 'UNSURE': continue
            l2 = '9999_other_sign' if l2 == 'OTHER' else l2
            if l2 != r['true_class']:
                new[fkey(r['original_file'])] = l2; tally['changed to annotator 2'] += 1
else:
    K = {r['item']: r for r in csv.DictReader(open('../adjudication_key.csv', encoding='utf-8'))}
    CH = {r['item']: r['choice'] for r in csv.DictReader(open('../adjudication_choices.csv', encoding='utf-8')) if r['choice']}
    for it, r in K.items():
        c = CH.get(it, 'UNSURE'); f = fkey(r['original_file'])
        if c == 'A': lab = r['option_A']; tally['annotator 1' if r['A_is'] == 'annotator1' else 'annotator 2'] += 1
        elif c == 'B': lab = r['option_B']; tally['annotator 2' if r['A_is'] == 'annotator1' else 'annotator 1'] += 1
        elif c == 'NEITHER': lab = '9999_other_sign'; tally['neither'] += 1
        else: lab = None; tally['cannot tell'] += 1
        new[f] = lab
print(f"rule: {RULE}; labels corrected: {dict(tally)}")
FOLD = {'dev': 'morocco_youtube', 'final': 'morocco_final', 'dashcam': 'morocco_dashcam', 'rain_rabat': 'morocco_rain_rabat', 'rain_tanger': 'morocco_rain_tanger'}
BUMP = {218: 221, 221: 218}
rng = np.random.default_rng(0)
for suf, lab_m in (('', 'ours'), ('_ir', 'inverted residual'), ('_plain', 'plain CNN')):
    print(f"\n== {lab_m}")
    pool_o, pool_a, sid_a = [], [], []
    for n, f in FOLD.items():
        d, cl, rows = load(n, suf); ids = np.array([int(c[:4]) for c in cl]); idx = {c: k for k, c in enumerate(cl)}
        y0 = np.array([r['y'] for r in rows]); y1 = y0.copy(); keep = np.ones(len(y0), bool)
        for k in range(len(y0)):
            key = (f, d['folder'][k], d['files'][k])
            if key in new:
                if new[key] is None: keep[k] = False
                else: y1[k] = idx.get(new[key], -1)
        sid = group_rule(rows)
        def stats(y, m):
            C = np.zeros((3, len(y)), bool); dang = 0
            for s in range(3):
                p = d['logits'][s].argmax(-1)
                for k in np.where(m)[0]:
                    C[s, k] = p[k] == y[k] or BUMP.get(ids[p[k]]) == ids[y[k]]
                    dang += kind(ids[y[k]], ids[p[k]]) == 'dangerous'
            return C, dang / 3 / m.sum() * 100
        m0 = y0 >= 0; m1 = (y1 >= 0) & keep
        C0, d0 = stats(y0, m0); C1, d1 = stats(y1, m1)
        lo, hi = boot(C1[:, m1].mean(0), sid[m1], rng)
        print(f"{n:<12} original {C0[:, m0].mean()*100:5.1f} (n={m0.sum()}, dangerous {d0:.1f}%)   corrected labels {C1[:, m1].mean()*100:5.1f} [{lo:.1f}-{hi:.1f}] (n={m1.sum()}, dangerous {d1:.1f}%)")
        if n != 'dev':
            pool_o.append(C0[:, m0]); pool_a.append(C1[:, m1]); sid_a.append(np.array([f"{n}|{u}" for u in sid[m1]]))
    a0 = np.concatenate(pool_o, 1).mean() * 100; A1 = np.concatenate(pool_a, 1); lo, hi = boot(A1.mean(0), np.concatenate(sid_a), rng)
    print(f"{'held-out':<12} original {a0:5.1f}   corrected labels {A1.mean()*100:5.1f} [{lo:.1f}-{hi:.1f}]")
