import numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from sign_groups import load
plt.rcParams.update({'font.family':'serif','font.size':9,'axes.spines.top':False,'axes.spines.right':False})
SAMEG={218:221,221:218}
def sm(z): e=np.exp(z-z.max(-1,keepdims=True)); return e/e.sum(-1,keepdims=True)
names={'dev':'Development (696)','final':'Final (423)','dashcam':'Tangier, dry (210)','rain':'Rain, 2 drives (240)'}
cols={'dev':'#7a7a7a','final':'#1f5fa8','dashcam':'#c0562b','rain':'#2a8a6a'}
GROUPS={'dev':['dev'],'final':['final'],'dashcam':['dashcam'],'rain':['rain_rabat','rain_tanger']}
fig,ax=plt.subplots(1,2,figsize=(7.0,3.0))
ax[0].plot([0,1],[0,1],ls='--',c='0.6',lw=0.8)
summary=[]
for n in GROUPS:
  conf=[];ok=[]
  for part in GROUPS[n]:
    d,cl,rows=load(part); ids=np.array([int(c[:4]) for c in cl]); y=np.array([r['y'] for r in rows]); ins=y>=0
    for s in range(3):
        P=sm(d['logits'][s].astype(float))[ins]; p=P.argmax(-1); yy=y[ins]
        c=(p==yy)|np.array([SAMEG.get(ids[a])==ids[b] for a,b in zip(p,yy)])
        conf.append(P.max(-1)); ok.append(c)
  if True:
    conf=np.concatenate(conf); ok=np.concatenate(ok).astype(float)
    bins=np.linspace(0,1,11); idx=np.clip(np.digitize(conf,bins)-1,0,9)
    xs,ys,ws=[],[],[]
    for b in range(10):
        m=idx==b
        if m.sum()>=5: xs.append(conf[m].mean()); ys.append(ok[m].mean()); ws.append(m.sum())
    ece=sum(abs(conf[idx==b].mean()-ok[idx==b].mean())*(idx==b).sum() for b in range(10) if (idx==b).sum())/len(conf)
    ax[0].plot(xs,ys,'o-',c=cols[n],ms=3,lw=1.2,label=f"{names[n]}, ECE {ece*100:.1f}%")
    o=np.argsort(-conf); cov=np.arange(1,len(o)+1)/len(o); risk=np.cumsum(1-ok[o])/np.arange(1,len(o)+1)
    aurc=np.trapezoid(risk,cov)
    ax[1].plot(cov*100,risk*100,c=cols[n],lw=1.2,label=f"{names[n]}, AURC {aurc*100:.2f}%")
    summary.append((n,ece,aurc))
ax[0].set_xlabel('Confidence (max softmax)'); ax[0].set_ylabel('Accuracy'); ax[0].set_xlim(0.2,1.0); ax[0].set_ylim(0.2,1.02)
ax[0].set_title('a  Reliability (in-scope crops, 3 seeds)',loc='left',fontsize=9); ax[0].legend(frameon=False,fontsize=7,loc='lower right')
ax[1].set_xlabel('Coverage (% of crops kept)'); ax[1].set_ylabel('Error on kept crops (%)'); ax[1].set_xlim(0,100); ax[1].set_ylim(0,14)
ax[1].set_title('b  Risk–coverage',loc='left',fontsize=9); ax[1].legend(frameon=False,fontsize=7,loc='upper left')
plt.tight_layout()
for ext in ('png','pdf'): plt.savefig(f'fig_calibration_risk_coverage.{ext}',dpi=300)
print(summary)
