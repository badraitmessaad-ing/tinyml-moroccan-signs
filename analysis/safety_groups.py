import numpy as np, csv, collections
SPEED={11:30,17:40,23:50,28:60,32:70,36:80,39:90,41:100,46:120}
SAME=[{218,221}]
# "same driver action" groups, defined from the highway code meaning (not from the errors)
ACTION=[{252,255},            # animals on the road: slow down, watch
        {190,191},            # no stopping / no parking: do not park here
        {203,204,205,206},    # dangerous curve(s): slow down for bends
        {218,221},            # bump / uneven road: slow down
        {243,244,436},        # pedestrians / children / crossing: slow down, watch for people
        {237,238,239},        # road narrows (both / left / right): slow down, merge
        {301,302},            # railway crossing (with / without barriers): slow down, look
        {231,342}]            # roundabout ahead / roundabout
# obligations whose loss is dangerous: reading them as any other sign removes a stop, yield, ban or entry prohibition
CRITICAL={142,135,141,183,148,113,114,116,123}
def kind(t,p):
    if t==p or any(t in g and p in g for g in SAME): return 'correct'
    if t in SPEED and p in SPEED: return 'dangerous' if SPEED[p]>SPEED[t] else 'safe-side'
    if t in CRITICAL and not (p in CRITICAL and t in {142,135} and p in {142,135}): return 'dangerous'
    if any(t in g and p in g for g in ACTION): return 'same action'
    return 'other'
for n in ('dev','final','dashcam','rain_rabat','rain_tanger'):
    d=np.load(f'preds_{n}.npz'); cl=list(d['classes']); ids=np.array([int(c[:4]) for c in cl]); ci={c:i for i,c in enumerate(cl)}
    y=[ci.get(f,-1) for f in d['folder']]
    cnt=collections.Counter(); ex=collections.Counter()
    for s in range(3):
        pr=d['logits'][s].argmax(-1)
        for k,t in enumerate(y):
            if t<0: continue
            kd=kind(ids[t],ids[pr[k]]); cnt[kd]+=1
            if kd=='dangerous': ex[(cl[t][5:25],cl[pr[k]][5:25])]+=1
    err=sum(v for k,v in cnt.items() if k!='correct'); tot=sum(cnt.values())
    print(f"{n}: errors per seed {err/3:.1f} of {tot/3:.0f};", {k:round(v/3,1) for k,v in cnt.items() if k!='correct'})
    print('   dangerous examples:',ex.most_common(6))
