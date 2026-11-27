import numpy as np, collections
from sign_groups import load, correct, boot
def group_rule(rows,G=10,cap=2):
    sid=[None]*len(rows); by=collections.defaultdict(list); nxt=0
    for r in rows: by[(r['drive'],r['folder'])].append(r)
    for L in by.values():
        shots=collections.defaultdict(list)
        for r in L: shots[(r['t'],r['shot'])].append(r)
        chain=[]; chains=[]
        for s in sorted(shots):
            if chain and (s[0]-chain[-1][0]>G or len(chain)>=cap): chains.append(chain); chain=[]
            chain.append(s)
        if chain: chains.append(chain)
        for ch in chains:
            m=max(len(shots[s]) for s in ch)
            for s in ch:
                for j,r in enumerate(sorted(shots[s],key=lambda r:r['x'])): sid[r['k']]=nxt+min(j,m-1)
            nxt+=m
    return np.array(sid)
if __name__=='__main__':
    rng=np.random.default_rng(0); out={}
    for n in ('dev','final','dashcam','rain_rabat','rain_tanger'):
        d,cl,rows=load(n); C=correct(d,cl,rows); ins=np.array([r['y']>=0 for r in rows]); acc=C.mean(0)[ins]
        shot=np.array([r['shot'] for r in rows])[ins]; s_lo,s_hi=boot(acc,shot,rng)
        sid=group_rule(rows)[ins]; c=collections.Counter(sid); lo,hi=boot(acc,sid,rng)
        per_seed=[np.mean([C[s][ins][sid==u].mean() for u in np.unique(sid)])*100 for s in range(3)]
        print(f"{n}: {ins.sum()} crops -> {len(c)} signs (crops/sign {ins.sum()/len(c):.2f}, max {max(c.values())}) | acc {acc.mean()*100:.1f} | "
              f"screenshot CI {s_lo:.1f}-{s_hi:.1f} | sign CI {lo:.1f}-{hi:.1f} | one-vote-per-sign {np.mean(per_seed):.1f}±{np.std(per_seed):.1f}")
