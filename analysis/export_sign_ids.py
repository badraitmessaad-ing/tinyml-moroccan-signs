"""Write sign_ids_all.csv: the physical-sign identifier of every test crop (rule of Section 2.2.9: same drive and
class, screenshots at most 10 s apart, at most two screenshots per sign), for the dataset release."""
import csv
from sign_groups import load
from sign_rule import group_rule
SETF = {'dev': 'morocco_youtube', 'final': 'morocco_final', 'dashcam': 'morocco_dashcam',
        'rain_rabat': 'morocco_rain_rabat', 'rain_tanger': 'morocco_rain_tanger'}
with open('sign_ids_all.csv', 'w', newline='', encoding='utf-8') as fh:
    w = csv.writer(fh); w.writerow(['project_folder', 'class_folder', 'original_name', 'sign_id'])
    for n, f in SETF.items():
        d, cl, rows = load(n); sid = group_rule(rows)
        for r, s in zip(rows, sid): w.writerow([f, r['folder'], r['file'], f"{n}_{int(s):04d}"])
print('written sign_ids_all.csv')
