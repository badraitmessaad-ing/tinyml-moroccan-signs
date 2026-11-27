"""REVISION: agreement between the original labels and the second annotator (Cohen's kappa).
Usage:  python score_agreement.py labels_annotator2.csv
Writes: agreement_report.txt (overall, final set, development set; list of disagreements)."""
import csv, sys
from collections import Counter
key = {r["item"]: r for r in csv.DictReader(open("annotator_key.csv", encoding="utf-8"))}
lab = {r["item"]: r["label"] for r in csv.DictReader(open(sys.argv[1], encoding="utf-8")) if r["label"]}
def norm(c): return "OTHER" if c in ("9999_other_sign", "OTHER") else c
def kappa(a, b):
    n = len(a); po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b); pe = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / n / n
    return po, (po - pe) / (1 - pe) if pe < 1 else 1.0
L = []
for part in ("all", "final", "dev", "tangier"):
    its = [i for i in key if i in lab and lab[i] != "UNSURE" and (part == "all" or key[i]["set"] == part)]
    a = [norm(key[i]["true_class"]) for i in its]; b = [norm(lab[i]) for i in its]
    po, k = kappa(a, b) if its else (0, 0)
    uns = sum(1 for i in key if lab.get(i) == "UNSURE" and (part == "all" or key[i]["set"] == part))
    L.append(f"{part:>5}: {len(its)} crops compared, agreement {po*100:.1f}%, Cohen's kappa {k:.3f}, 'cannot tell' {uns}")
L.append(f"not labelled yet: {sum(1 for i in key if i not in lab)}"); L.append("\nDisagreements (item, set, original -> annotator 2):")
for i in sorted(key):
    if i in lab and lab[i] != "UNSURE" and norm(key[i]["true_class"]) != norm(lab[i]):
        L.append(f"  {i} {key[i]['set']:>5}  {key[i]['true_class']}  ->  {lab[i]}")
t = "\n".join(L); print(t); open("agreement_report.txt", "w", encoding="utf-8").write(t + "\n")
