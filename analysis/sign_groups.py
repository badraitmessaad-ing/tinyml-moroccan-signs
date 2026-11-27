import numpy as np, csv, re, collections, sys
def tsec(name):
    m=re.search(r'(\d{4})-(\d\d)-(\d\d) (\d\d)(\d\d)(\d\d)',name); 
    return int(m[3])*86400+int(m[4])*3600+int(m[5])*60+int(m[6])
def load(n, suffix=''):
    d=np.load(f'preds_{n}{suffix}.npz'); cl=list(d['classes']); ci={c:i for i,c in enumerate(cl)}
    rows=[]
    ann={}
    if n in ('dev','final'):
        ann={(r['class_folder'],r['original_name']):r for r in csv.DictReader(open({'dev':'dev.csv','final':'final_test.csv'}[n],encoding='utf-8'))}
    for k,(f,o,s) in enumerate(zip(d['folder'],d['files'],d['screenshot'])):
        s=s.replace('\\','/'); r=ann.get((f,o),{})
        drive=r.get('drive') or s.split('/')[-2]
        idx=int(re.search(r'_(\d+)\.png$',o)[1])
        x=float(r['x0'])+float(r['x1']) if r else idx
        rows.append(dict(k=k,folder=f,file=o,shot=s,drive=drive,t=tsec(o),x=x,y=ci.get(f,-1)))
    return d,cl,rows
def group(rows,G):
    sid=[None]*len(rows); by=collections.defaultdict(list)
    for r in rows: by[(r['drive'],r['folder'])].append(r)
    nxt=0
    for key,L in by.items():
        shots=collections.defaultdict(list)
        for r in L: shots[(r['t'],r['shot'])].append(r)
        order=sorted(shots); chain=[]; chains=[]
        for s in order:
            if chain and s[0]-chain[-1][0]>G: chains.append(chain); chain=[]
            chain.append(s)
        if chain: chains.append(chain)
        for ch in chains:
            m=max(len(shots[s]) for s in ch)
            for s in ch:
                for j,r in enumerate(sorted(shots[s],key=lambda r:r['x'])): sid[r['k']]=nxt+min(j,m-1)
            nxt+=m
    return np.array(sid)
SAME=[{218,221}]
def correct(d,cl,rows):
    ids=[int(c[:4]) for c in cl]; C=[]
    for s in range(3):
        p=d['logits'][s].argmax(-1)
        C.append([r['y']>=0 and (p[r['k']]==r['y'] or any(ids[p[r['k']]] in g and ids[r['y']] in g for g in SAME)) for r in rows])
    return np.array(C,float)
def boot(acc,unit,rng,B=2000):
    U=np.unique(unit); g={u:np.where(unit==u)[0] for u in U}; out=[]
    for _ in range(B):
        ii=np.concatenate([g[u] for u in rng.choice(U,len(U))]); out.append(acc[ii].mean()*100)
    return np.percentile(out,[2.5,97.5])
if __name__=='__main__':
    rng=np.random.default_rng(0)
    for n in ('dev','final','dashcam','rain_rabat','rain_tanger'):
        d,cl,rows=load(n); C=correct(d,cl,rows); ins=np.array([r['y']>=0 for r in rows])
        acc=C.mean(0)[ins]; shot=np.array([r['shot'] for r in rows])[ins]
        lo,hi=boot(acc,shot,rng)
        print(f"== {n}: {ins.sum()} crops, {len(set(shot))} screenshots, acc {acc.mean()*100:.1f}, screenshot CI {lo:.1f}-{hi:.1f}")
        for G in (0,10,30,60,120):
            sid=group(rows,G)[ins]; cnt=collections.Counter(sid)
            # sign-level accuracy: each sign weighted equally
            sa=np.array([acc[sid==u].mean() for u in np.unique(sid)])
            lo,hi=boot(acc,sid,rng)
            print(f"   G={G:3d}s: {len(cnt)} signs, crops/sign mean {np.mean(list(cnt.values())):.2f} max {max(cnt.values())}, "
                  f"singletons {sum(v==1 for v in cnt.values())} | sign-bootstrap CI {lo:.1f}-{hi:.1f} | one-vote-per-sign acc {sa.mean()*100:.1f}")
