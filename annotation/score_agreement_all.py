"""REVISION: agreement between the original labels and the second annotator over BOTH packs (Cohen's kappa).
Usage (project folder):
    python score_agreement_all.py annotator_key.csv labels_annotator2.csv annotator_key_part2.csv labels_annotator2_part2.csv
Writes agreement_report_all.txt: overall, per set (dev, final, tangier, rain_rabat, rain_tanger), per pack,
'cannot tell' counts and the list of disagreements."""
import csv, sys
from collections import Counter
args = sys.argv[1:]; assert len(args) % 2 == 0 and args, "give pairs: key.csv labels.csv"
key, lab, pack = {}, {}, {}
for p, (kf, lf) in enumerate(zip(args[0::2], args[1::2]), 1):
    for r in csv.DictReader(open(kf, encoding="utf-8")): key[r["item"]] = r; pack[r["item"]] = p
    for r in csv.DictReader(open(lf, encoding="utf-8")):
        if r["label"]: lab[r["item"]] = r["label"]
def norm(c): return "OTHER" if c in ("9999_other_sign", "OTHER") else c
def kappa(a, b):
    n = len(a); po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b); pe = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / n / n
    return po, (po - pe) / (1 - pe) if pe < 1 else 1.0
L = [f"{len(key)} crops in {len(args)//2} pack(s); labelled by annotator 2: {sum(1 for i in key if i in lab)}"]
groups = [("all", lambda i: True)] + [(s, (lambda s: lambda i: key[i]["set"] == s)(s)) for s in ("dev", "final", "tangier", "rain_rabat", "rain_tanger")] \
         + [(f"pack {p}", (lambda p: lambda i: pack[i] == p)(p)) for p in sorted(set(pack.values()))]
for name, f in groups:
    its = [i for i in key if f(i) and i in lab and lab[i] != "UNSURE"]
    if not its: continue
    po, k = kappa([norm(key[i]["true_class"]) for i in its], [norm(lab[i]) for i in its])
    uns = sum(1 for i in key if f(i) and lab.get(i) == "UNSURE")
    L.append(f"{name:>12}: {len(its):5d} crops compared, agreement {po*100:5.1f}%, Cohen's kappa {k:.3f}, 'cannot tell' {uns}")
L.append(f"not labelled yet: {sum(1 for i in key if i not in lab)}")
L.append("\nDisagreements (item, set, original -> annotator 2, original file):")
for i in sorted(key):
    if i in lab and lab[i] != "UNSURE" and norm(key[i]["true_class"]) != norm(lab[i]):
        L.append(f"  {i:>10} {key[i]['set']:>11}  {key[i]['true_class']}  ->  {lab[i]}   ({key[i]['original_file']})")
t = "\n".join(L); print(t); open("agreement_report_all.txt", "w", encoding="utf-8").write(t + "\n")
