import numpy as np, csv, os, collections, json
SPEED={11:30,17:40,23:50,28:60,32:70,36:80,39:90,41:100,46:120}
SPEED_IDS={11,17,23,28,36,41,46}   # paper's speed set (7 classes present)
SAME=[{218,221}]
def load(n,c):
    d=np.load(f'preds_{n}.npz'); classes=list(d['classes']); ids=np.array([int(x[:4]) for x in classes])
    ann={(r['class_folder'],r['original_name']):r for r in csv.DictReader(open(c,encoding='utf-8'))}
    rows=[ann[(f,o)] for f,o in zip(d['folder'],d['files'])]
    z=d['logits']; return d,classes,ids,rows,z
def softmax(z): e=np.exp(z-z.max(-1,keepdims=True)); return e/e.sum(-1,keepdims=True)
out={}
for n,c in (('dev','dev.csv'),('final','final_test.csv')):
    d,classes,ids,rows,z=load(n,c); cidx={k:i for i,k in enumerate(classes)}
    y=np.array([cidx.get(r['class_folder'],-1) for r in rows]); ins=y>=0
    drive=np.array([r['drive'] for r in rows]); shot=np.array([r['drive']+'/'+r['screenshot_file'] for r in rows])
    P=softmax(z); pred=P.argmax(-1); conf=P.max(-1)
    energy=np.log(np.exp(z-z.max(-1,keepdims=True)).sum(-1))+z.max(-1)
    def corr(p,t): return p==t or any(ids[p] in g and ids[t] in g for g in SAME)
    C=np.array([[corr(pred[s,k],y[k]) if y[k]>=0 else False for k in range(len(y))] for s in range(3)])
    sp=np.array([ids[t] in SPEED_IDS if t>=0 else False for t in y])
    def rep(mask,lab):
        a=C[:,mask].mean(1)*100; return f"{lab}: {a.mean():.1f} ± {a.std():.1f} (n={mask.sum()})"
    print(f"\n===== {n} =====")
    print(rep(ins,'all in-scope'), '|', rep(ins&sp,'speed'))
    for dr in sorted(set(drive[ins])):
        m=ins&(drive==dr); print('  ',rep(m,dr),'|',rep(m&sp,'speed'))
    # screenshot bootstrap per drive (seed-pooled mean over seeds)
    rng=np.random.default_rng(0)
    for dr in sorted(set(drive[ins]))+['ALL']:
        m=ins&((drive==dr) if dr!='ALL' else True)
        idx=np.where(m)[0]; S=sorted(set(shot[idx])); g={s:idx[shot[idx]==s] for s in S}
        acc=C.mean(0)
        bs=[]
        for _ in range(2000):
            pick=rng.choice(len(S),len(S)); ii=np.concatenate([g[S[j]] for j in pick]); bs.append(acc[ii].mean()*100)
        print(f"   CI screenshot-bootstrap {dr}: {np.percentile(bs,2.5):.1f}-{np.percentile(bs,97.5):.1f} ({len(S)} screenshots)")
    # calibration (in-scope), pooled over seeds
    cf=conf[:,ins].ravel(); ok=C[:,ins].ravel()
    bins=np.linspace(0,1,11); ece=0; rel=[]
    for a,b in zip(bins[:-1],bins[1:]):
        m=(cf>a)&(cf<=b)
        if m.sum(): ece+=m.mean()*abs(ok[m].mean()-cf[m].mean()); rel.append((round(a,1),int(m.sum()),round(cf[m].mean(),3),round(ok[m].mean(),3)))
    print(f"   ECE(10 bins)={ece*100:.2f}%  mean conf={cf.mean():.3f} acc={ok.mean():.3f}")
    for r in rel: print('     bin',r)
    # risk-coverage on in-scope by max-softmax
    order=np.argsort(-cf); okr=ok[order]; cov=np.arange(1,len(okr)+1)/len(okr); risk=1-np.cumsum(okr)/np.arange(1,len(okr)+1)
    aurc=risk.mean()
    print(f"   AURC={aurc*100:.2f}%  risk@100%={risk[-1]*100:.1f}  ", ' '.join(f"risk@{c}%={risk[int(c/100*len(okr))-1]*100:.1f}" for c in (95,90,80)))
    # OOD (other signs) acceptance vs threshold on max softmax
    unk=~ins
    if unk.sum():
        for th in (0.5,0.7,0.9):
            acc_known=(conf[:,ins]>=th).mean()*100; acc_unk=(conf[:,unk]>=th).mean()*100
            sel=(conf[:,ins]>=th); a_sel=(C[:,ins][sel]).mean()*100
            print(f"   thr {th}: in-scope kept {acc_known:.1f}%, acc of kept {a_sel:.1f}%, OTHER accepted {acc_unk:.1f}% (n other={unk.sum()})")
    # safety: speed-limit errors direction
    cnt=collections.Counter()
    for s in range(3):
        for k in np.where(ins&sp)[0]:
            t=ids[y[k]]; p=ids[pred[s,k]]
            if C[s,k]: continue
            if p in SPEED: cnt['over (reads higher limit)' if SPEED[p]>SPEED[t] else 'under (reads lower limit)']+=1
            else: cnt['speed read as non-speed sign']+=1
        for k in np.where(ins&~sp)[0]:
            p=ids[pred[s,k]]
            if not C[s,k] and p in SPEED: cnt['non-speed read as speed limit']+=1
    print('   speed-related errors (3 seeds pooled):',dict(cnt))
    conf_pairs=collections.Counter()
    for s in range(3):
        for k in np.where(ins&sp&~C[s])[0]: conf_pairs[(SPEED.get(ids[y[k]]),SPEED.get(ids[pred[s,k]],classes[pred[s,k]][:20]))]+=1
    print('   speed confusions true->pred:',conf_pairs.most_common(12))
    out[n]=dict(drive=drive,ins=ins,sp=sp,C=C,conf=conf,y=y,pred=pred,classes=classes,shot=shot)
np.save('cache.npy',out,allow_pickle=True)
