import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'font.family':'serif','font.size':9,'axes.spines.top':False,'axes.spines.right':False})
ours=[('31k gray',44.2,97.05),('31k RGB',44.5,97.53),('40k gray',53.4,97.37),('40k RGB',53.7,97.42),('52k gray',66.9,96.81),('52k RGB',67.2,96.73)]
quoted=[('AykoNet-Lite (M0+)',36.8,94.6),('AykoNet-Pro (M0+)',80.2,95.9),('ST FD-MobileNet\n(128×128, H747)',192,99.51),('MicronNet (fp16,\nCortex-A53)',1024,98.9)]
fig,ax=plt.subplots(figsize=(4.6,3.2))
allp=[(x,y) for _,x,y in ours+quoted]; front=[]; best=-1
for x,y in sorted(allp):
    if y>best: front.append((x,y)); best=y
ax.step([p[0] for p in front],[p[1] for p in front],where='post',c='0.75',lw=1,zorder=0,label='Pareto front')
ax.scatter([x for _,x,_ in ours],[y for *_,y in ours],c='#1f5fa8',s=22,zorder=3,label='Ours (int8, measured, 3 seeds)')
ax.scatter([x for _,x,_ in quoted],[y for *_,y in quoted],facecolors='white',edgecolors='#c0562b',s=26,lw=1.2,zorder=3,label='Published (quoted, not reproduced)')
for n,x,y in ours:
    if n in ('31k gray','31k RGB','52k gray'): ax.annotate(n,(x,y),xytext=((-14,-13) if n=='31k gray' else (4,3)),textcoords='offset points',fontsize=7,color='#1f5fa8')
off={'AykoNet-Lite (M0+)':(4,-3),'AykoNet-Pro (M0+)':(4,-3),'ST FD-MobileNet\n(128×128, H747)':(-10,-22),'MicronNet (fp16,\nCortex-A53)':(-68,-20)}
for n,x,y in quoted: ax.annotate(n,(x,y),xytext=off[n],textcoords='offset points',fontsize=7,color='#8a3a1b')
ax.set_xscale('log'); ax.set_xticks([30,50,100,200,500,1000]); ax.set_xticklabels(['30','50','100','200','500','1000'])
ax.set_xlabel('Model size on the device (KB, log scale)'); ax.set_ylabel('GTSRB test accuracy (%)'); ax.set_ylim(94,100.2)
ax.legend(frameon=False,fontsize=7,loc='lower right')
plt.tight_layout()
for e in ('png','pdf'): plt.savefig(f'fig_gtsrb_pareto.{e}',dpi=300)
