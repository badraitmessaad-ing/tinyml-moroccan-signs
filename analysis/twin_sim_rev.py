"""REVISION: adds the empirical rho = 0.29 (repeated screenshots, dev set). Road digital-twin update simulation driven by the measured final-test confusion matrix.
Frozen models; nothing is tuned. Seed fixed."""
import csv,numpy as np,collections,json
D=""  # run from the project folder
r=list(csv.reader(open(D+'eval_final_confusion.csv')))
cols=r[0][1:]; P={x[0]:np.array(list(map(float,x[1:]))) for x in r[1:]}
P={k:v/v.sum() for k,v in P.items()}
log=list(csv.DictReader(open(D+'morocco_final/crops_log.csv')))
inv=[l['class'] for l in log if l['class'] in P]
cls=sorted(P); speed=[c for c in cls if 'speed' in c]
rng=np.random.default_rng(2026)
def read(c): return cols[rng.choice(len(cols),p=P[c])]
def passes(c,k,rho):
    out=[read(c)]
    for _ in range(k-1): out.append(out[0] if rng.random()<rho else read(c))
    return out
def vote(v): return collections.Counter(v).most_common(1)[0][0]
def changed(c):
    pool=[s for s in speed if s!=c] if c in speed else [s for s in cls if s!=c and s not in speed]
    return pool[rng.integers(len(pool))]
R=300; res={}; resid=collections.Counter()
for rho in (0.0,0.29,0.5,1.0):
  for k in (1,3,5,7,9):
    acc=[];tp=[];fa=[]
    for _ in range(R):
        ch=rng.random(len(inv))<0.05
        a=t=f=n=0
        for c,isch in zip(inv,ch):
            real=changed(c) if isch else c
            v=vote(passes(real,k,rho))
            a+= v==real
            if isch: t+= v!=c; n+=1
            else:
                f+= v!=c
                if v!=c and rho==0 and k==7: resid[(c,v)]+=1
        acc.append(a/len(inv)); tp.append(t/max(n,1)); fa.append(100*f/(len(inv)-n))
    res[(rho,k)]=(100*np.mean(acc),100*np.std(acc),100*np.mean(tp),np.mean(fa),np.std(fa))
    print(rho,k,['%.2f'%x for x in res[(rho,k)]])
tot=sum(resid.values())
print('top residual false alarms at k=7 rho=0 (share):')
for (a,b),n in resid.most_common(8): print(f'  {a} -> {b}: {100*n/tot:.0f}%')
json.dump({f'{a}_{b}':v for (a,b),v in res.items()},open('twin_results_rev.json','w'),indent=1)
# figure
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
ks=[1,3,5,7,9]
fig,ax=plt.subplots(1,2,figsize=(7.2,3.1),dpi=300)
sty={0.0:('-','o','independent passes'),0.29:('-','D','measured \u03c1 = 0.29'),0.5:('--','s','half of passes repeat the first error'),1.0:(':','^','all passes repeat the first error')}
for rho,(ls,mk,lab) in sty.items():
    ax[0].plot(ks,[res[(rho,k)][0] for k in ks],ls=ls,marker=mk,color='#1f3b73',label=lab,ms=4)
    ax[1].plot(ks,[res[(rho,k)][3] for k in ks],ls=ls,marker=mk,color='#b5462a',label=lab,ms=4)
ax[0].set_ylabel('Sign layer correct (%)'); ax[1].set_ylabel('False change alerts per 100 signs')
for a in ax: a.set_xlabel('Passes per sign'); a.set_xticks(ks); a.grid(alpha=.3,lw=.5); a.spines[['top','right']].set_visible(False)
h,l=ax[0].get_legend_handles_labels(); fig.legend(h,l,loc='lower center',ncol=2,fontsize=7,frameon=False)
fig.tight_layout(rect=(0,0.08,1,1)); fig.savefig('figures/fig9_twin_rev.png',dpi=300); fig.savefig('figures/fig9_twin_rev.pdf'); print('saved')
