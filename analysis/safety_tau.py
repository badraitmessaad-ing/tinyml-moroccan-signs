import numpy as np, collections, sys
exec(open('safety_groups.py').read().split('for n in')[0])   # reuse SPEED, groups, kind()
def sm(z): e=np.exp(z-z.max(-1,keepdims=True)); return e/e.sum(-1,keepdims=True)
def decode(P,ids,tau,mode):
    pred=P.argmax(-1).copy()
    if tau is None: return pred
    sp_idx=[i for i,c in enumerate(ids) if c in SPEED]
    for k in range(len(pred)):
        t=ids[pred[k]]
        if t not in SPEED: continue
        cand=[i for i in sp_idx if SPEED[ids[i]]<SPEED[t] and P[k,i]>=tau*P[k,pred[k]]]
        if cand:
            pred[k]= max(cand,key=lambda i:P[k,i]) if mode=='prob' else min(cand,key=lambda i:SPEED[ids[i]])
    return pred
mode=sys.argv[1] if len(sys.argv)>1 else 'prob'
for n in ('dev','final','dashcam','rain_rabat','rain_tanger'):
    d=np.load(f'preds_{n}.npz'); cl=list(d['classes']); ids=np.array([int(c[:4]) for c in cl]); ci={c:i for i,c in enumerate(cl)}
    y=np.array([ci.get(f,-1) for f in d['folder']]); ins=y>=0
    for tau in (None,0.3):
        cnt=collections.Counter(); ex=collections.Counter(); acc=[]
        for s in range(3):
            pr=decode(sm(d['logits'][s].astype(float)),ids,tau,mode); ok=0
            for k in np.where(ins)[0]:
                kd=kind(ids[y[k]],ids[pr[k]]); cnt[kd]+=1; ok+=kd=='correct'
                if kd=='dangerous': ex[(cl[y[k]][5:28],cl[pr[k]][5:28])]+=1
            acc.append(ok/ins.sum()*100)
        tot=ins.sum(); err=sum(v for k,v in cnt.items() if k!='correct')/3
        print(f"{n:8s} tau={tau}: acc {np.mean(acc):.1f}±{np.std(acc):.1f}  errors/seed {err:.1f} of {tot} |",
              ' '.join(f"{k} {cnt[k]/3:.1f}" for k in ('dangerous','safe-side','same action','other')),
              f"| dangerous rate {cnt['dangerous']/3/tot*100:.1f}%")
        if tau: print('    dangerous:',[(a,b,round(v/3,1)) for (a,b),v in ex.most_common(8)])
