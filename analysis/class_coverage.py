"""R1.9: the 70 classes grouped by the categories of the Moroccan road-sign regulation (arrete 2805-14 of
1 Aug 2014, amended by arrete 3106-19 of 10 Oct 2019; IGSR), share of encountered signs outside the 70
classes per drive, and accuracy per category (3 frozen seeds)."""
import numpy as np, collections
from sign_groups import load
CAT = {
 'danger': [202,203,204,205,206,208,218,221,223,237,238,239,241,242,243,244,245,252,255,301,302,323],
 'intersection and priority': [132,133,134,135,140,141,142,231,298],
 'prohibition (incl. 9 speed limits)': [11,17,23,28,32,36,39,41,46,113,114,116,123,130,131,148,150,176,178,179,183,184,190,191],
 'obligation': [326,327,328,330,331,336,337,339,340,342],
 'indication': [333,364,406,436,444]}
cat_of = {c: k for k, v in CAT.items() for c in v}
assert sum(len(v) for v in CAT.values()) == 70
SAMEG = {218: 221, 221: 218}
for n, name in (('dev', 'development'), ('final', 'final'), ('dashcam', 'Tangier'), ('rain_rabat', 'Rabat rain'), ('rain_tanger', 'Tangier rain')):
    d, cl, rows = load(n); ids = [int(c[:4]) for c in cl]
    assert set(ids) == set(cat_of)
    tot = collections.Counter(r['drive'] for r in rows)
    oos = collections.Counter(r['drive'] for r in rows if r['y'] < 0)
    print(f"== {name}: {len(rows)} crops, {sum(oos.values())} outside the 70 classes ({100*sum(oos.values())/len(rows):.1f}%)")
    for dr in sorted(tot): print(f"   {dr:<28} {oos[dr]:>3}/{tot[dr]:<4} outside ({100*oos[dr]/tot[dr]:.1f}%)")
    acc = collections.defaultdict(list)
    for s in range(3):
        p = d['logits'][s].argmax(-1)
        for k, r in enumerate(rows):
            if r['y'] < 0: continue
            t, q = ids[r['y']], ids[p[k]]
            acc[cat_of[t]].append(q == t or SAMEG.get(q) == t)
    for c in CAT:
        if acc[c]: print(f"   {c:<36} {len(acc[c])//3:>4} crops   accuracy {100*np.mean(acc[c]):.1f}%")
print("\nclasses per category:", {k: len(v) for k, v in CAT.items()})
