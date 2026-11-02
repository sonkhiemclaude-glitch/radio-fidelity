"""Apply corrected admission test to cached localization losses.

Includes training in the pipeline cost; resampling never retrains the localizer.
Targets clipped mean positioning error, not median error or link likelihood.
"""
import json,time
from pathlib import Path
import numpy as np
from exp11_cluster_admission import selected,interval
import dataset as D
ROOT=Path(__file__).resolve().parents[1]
def main():
 t=time.time();models=json.loads((ROOT/'results/exp13_models.json').read_text());rows=[];B=2000
 for arm in ['free','equal']:
  for seed in range(3):
   for model in ['knn','mlp','transformer']:
    paths=[ROOT/f'data/pred_{arm}_guess_F{lv}_s{seed}_{model}.npz' for lv in range(6)]
    if not all(p.exists() for p in paths):raise FileNotFoundError('All six saved prediction levels are required')
    z=[np.load(p) for p in paths];q=z[0]['query'];assert all(np.array_equal(v['query'],q) for v in z)
    err=np.stack([v['error'] for v in z],1);d=D.load();ub,inv=np.unique(D.block_id(d['x'][q],d['y'][q]),return_inverse=True);counts=np.bincount(inv)
    for cap in [1000.,5000.,10000.]:
     loss=np.clip(err,0,cap);truth=loss.mean(0);harm=np.r_[False,np.maximum.accumulate(np.diff(truth)>=0)]
     blk=np.stack([np.bincount(inv,weights=loss[:,j])/counts for j in range(6)],1)
     for design,units,p in [('message_wr',loss,None),('spatial_pps_wr',blk,counts/counts.sum())]:
      for n in ([64,256,1000,5000] if cap==5000 else [1000]):
       rng=np.random.default_rng(3108+seed+n);sel=[];arg=[]
       for b in range(0,B,50):
        ix=rng.choice(len(units),(min(50,B-b),n),replace=True,p=p);a,c=selected(units[ix],cap);sel.extend(a);arg.extend(c)
       sel=np.asarray(sel);arg=np.asarray(arg);k=int(harm[sel].sum());rows.append(dict(arm=arm,seed=seed,model=model,clip_m=cap,design=design,n_units=n,B=B,population_mean_loss_m=truth.tolist(),global_argmin=int(truth.argmin()),harmful_count=k,harmful_rate=k/B,harmful_ci95=interval(k,B),selected_counts=np.bincount(sel,minlength=6).tolist(),exact_global=float(np.mean(sel==truth.argmin())),mean_level=float(sel.mean()),argmin_harmful=float(harm[arg].mean())))
    print(arm,seed,model,'done',time.time()-t,flush=True)
 out=dict(rows=rows,B=B,target='Clipped mean localization error over the fixed query population; independent WR evaluation units; not median error.',runtime_s=time.time()-t)
 (ROOT/'results/exp14_downstream_admission.json').write_text(json.dumps(out,indent=2))
if __name__=='__main__':main()
