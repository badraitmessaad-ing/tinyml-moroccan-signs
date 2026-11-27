import numpy as np, collections
from sklearn.metrics import f1_score, balanced_accuracy_score
from sign_groups import load
from sign_rule import group_rule
SAMEG={218:221,221:218}
def sm(z): e=np.exp(z-z.max(-1,keepdims=True)); return e/e.sum(-1,keepdims=True)
rng=np.random.default_rng(0)
out={}
for n in ('dev','final','dashcam','rain_rabat','rain_tanger'):
    d,cl,rows=load(n); ids=np.array([int(c[:4]) for c in cl])
    y=np.array([r['y'] for r in rows]); ins=y>=0
    sid=group_rule(rows)[ins]; y=y[ins]
    P=np.stack([sm(d['logits'][s].astype(float))[ins] for s in range(3)])
    preds=P.argmax(-1)
    # map same-group predictions to truth label
    for s in range(3):
        for k in range(len(y)):
            if preds[s,k]!=y[k] and SAMEG.get(ids[preds[s,k]])==ids[y[k]]: preds[s,k]=y[k]
    labs=np.unique(y)
    def metrics(idx):
        f=[f1_score(y[idx],preds[s,idx],labels=np.unique(y[idx]),average='macro',zero_division=0) for s in range(3)]
        b=[balanced_accuracy_score(y[idx],preds[s,idx]) for s in range(3)]
        a=[(preds[s,idx]==y[idx]).mean() for s in range(3)]
        return np.mean(f)*100,np.mean(b)*100,np.mean(a)*100
    import warnings; warnings.filterwarnings('ignore')
    base=metrics(np.arange(len(y)))
    U=np.unique(sid); g={u:np.where(sid==u)[0] for u in U}; bs=[]
    for _ in range(1000):
        ii=np.concatenate([g[u] for u in rng.choice(U,len(U))]); bs.append(metrics(ii))
    bs=np.array(bs); lo,hi=np.percentile(bs,[2.5,97.5],axis=0)
    sup=collections.Counter(y); 
    print(f"== {n}: {len(y)} crops, {len(labs)} classes; support: >=20: {sum(v>=20 for v in sup.values())}, 6-19: {sum(6<=v<20 for v in sup.values())}, 1-5: {sum(v<=5 for v in sup.values())}")
    print(f"  macro-F1 {base[0]:.1f} [{lo[0]:.1f}-{hi[0]:.1f}]  balanced acc {base[1]:.1f} [{lo[1]:.1f}-{hi[1]:.1f}]  acc {base[2]:.1f} [{lo[2]:.1f}-{hi[2]:.1f}]")
    out[n]=dict(sup={cl[k]:v for k,v in sup.items()})
    # voting on repeated screenshots: signs with >=2 crops
    multi=[u for u in U if len(g[u])>=2]
    single=[];vote=[];pairs=[]
    for s in range(3):
        for u in multi:
            ii=g[u]; t=y[ii[0]]
            c=(preds[s,ii]==y[ii]); single.extend(c)
            pv=P[s,ii].mean(0).argmax(); pv= t if (pv==t or SAMEG.get(ids[pv])==ids[t]) else pv
            vote.append(pv==t)
            if len(ii)==2: pairs.append((1-c[0],1-c[1]))
    pairs=np.array(pairs,float).reshape(-1,2)
    if len(pairs)<5:
        print(f"  repeated signs: {len(multi)} signs with 2+ crops; per-crop acc {np.mean(single)*100 if single else float('nan'):.1f}, vote acc {np.mean(vote)*100 if vote else float('nan'):.1f} (too few pairs for rho)"); continue
    e1,e2=pairs[:,0],pairs[:,1]
    rho=np.corrcoef(e1,e2)[0,1] if e1.std()>0 and e2.std()>0 else float('nan')
    # bootstrap rho over pairs
    rb=[]
    for _ in range(2000):
        ii=rng.integers(0,len(pairs),len(pairs)); a,b=e1[ii],e2[ii]
        if a.std()>0 and b.std()>0: rb.append(np.corrcoef(a,b)[0,1])
    print(f"  repeated signs: {len(multi)} signs with 2+ crops; per-crop acc {np.mean(single)*100:.1f}, prob-average vote acc {np.mean(vote)*100:.1f}")
    print(f"  pairs {len(pairs)//3}/seed: P(err)={e1.mean():.3f},{e2.mean():.3f}; both wrong {np.mean(e1*e2):.3f}; rho={rho:.2f} " + (f"[{np.percentile(rb,2.5):.2f}-{np.percentile(rb,97.5):.2f}]" if rb else "(no pair with an error in both crops: rho undefined)"))
