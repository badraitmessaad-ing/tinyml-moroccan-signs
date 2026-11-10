import csv, random, collections, sys
SEEDS=['c48_relu40k_s42_maR','c48_relu40k_s1_maR','c48_relu40k_s2_maR']
def load(f):
    return list(csv.DictReader(open(f, encoding='utf-8')))
def is_speed(c): return 'speed_limit' in c
def stats(rows, B=2000, rng=random.Random(0)):
    shots=collections.defaultdict(list)
    for r in rows: shots[r['screenshot']].append(r)
    keys=list(shots)
    def acc_seed(rs, s): return sum(r[s]==r['true'] for r in rs)/len(rs)
    per_seed=[acc_seed(rows,s) for s in SEEDS]
    mean=sum(per_seed)/3; sd=(sum((a-mean)**2 for a in per_seed)/3)**0.5
    boots=[]
    for _ in range(B):
        samp=[r for k in (rng.choice(keys) for _ in keys) for r in shots[k]]
        boots.append(sum(acc_seed(samp,s) for s in SEEDS)/3)
    boots.sort()
    return len(rows), len(keys), len({r['true'] for r in rows}), mean, sd, boots[int(0.025*B)], boots[int(0.975*B)-1]
for name,f in (('dev','results/recheck_dev_predictions.csv'),('final','results/recheck_final_predictions.csv')):
    rows=[r for r in load(f) if r['true']!='9999_other_sign']
    print('==',name)
    for drive in sorted({r['drive'] for r in rows}):
        rs=[r for r in rows if r['drive']==drive]
        n,ns,nc,m,sd,lo,hi=stats(rs)
        sp=[r for r in rs if is_speed(r['true'])]
        spm=sum(sum(r[s]==r['true'] for r in sp)/len(sp) for s in SEEDS)/3 if sp else float('nan')
        print(f'{drive:24s} crops {n:4d} shots {ns:4d} classes {nc:3d} acc {100*m:5.1f}±{100*sd:.1f} CI {100*lo:5.1f}-{100*hi:5.1f} speed {len(sp):3d} {100*spm:5.1f}')
    n,ns,nc,m,sd,lo,hi=stats(rows)
    print(f'{"ALL":24s} crops {n:4d} shots {ns:4d} classes {nc:3d} acc {100*m:5.1f}±{100*sd:.1f} CI {100*lo:5.1f}-{100*hi:5.1f}')
