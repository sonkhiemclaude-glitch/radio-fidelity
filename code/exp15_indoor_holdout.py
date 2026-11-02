"""Device-held-out indoor sensitivity with training-only AP and grid estimates.

Complements, rather than retroactively relabels, the original transductive arm.
All AP filtering/centroids and simulated reference-grid support use train devices.
"""
import json,time
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
import learners as L
from exp7_indoor import predict,PRI,SENS,FLOOR_DBM,GRID_M
ROOT=Path(__file__).resolve().parents[1]
def main():
 t=time.time();z=np.load(ROOT/'data/uji.npz');rows=[];splits=[]
 for split in range(3):
  rng=np.random.default_rng(3108+split);phones=np.unique(z['phone']);hold=rng.choice(phones,max(1,len(phones)//3),replace=False);te=np.isin(z['phone'],hold);tr=~te
  keep=z['heard'][tr].sum(0)>=20;R=z['R'][:,keep].astype(float);heard=z['heard'][:,keep];ng=R.shape[1];x,y,fl,bld=z['x'],z['y'],z['floor'],z['bld'];fh=float(z['floor_h']);zz=fl*fh
  lin=np.where(heard[tr],10**(R[tr]/10),0);total=lin.sum(0);ax=(lin*x[tr,None]).sum(0)/total;ay=(lin*y[tr,None]).sum(0)/total;af=(lin*fl[tr,None]).sum(0)/total
  ab=np.array([np.bincount(bld[tr],weights=lin[:,j]).argmax() for j in range(ng)]);az=af*fh
  gx=[];gy=[];gf=[];gb=[]
  for b in np.unique(bld[tr]):
   for f in np.unique(fl[tr&(bld==b)]):
    m=tr&(bld==b)&(fl==f)
    if m.sum()<30:continue
    X,Y=np.meshgrid(np.arange(x[m].min(),x[m].max()+GRID_M,GRID_M),np.arange(y[m].min(),y[m].max()+GRID_M,GRID_M));p=np.c_[X.ravel(),Y.ravel()];dd,_=cKDTree(np.c_[x[m],y[m]]).query(p);p=p[dd<=6]
    gx.extend(p[:,0]);gy.extend(p[:,1]);gf.extend([f]*len(p));gb.extend([b]*len(p))
  gx,gy,gf,gb=map(np.asarray,[gx,gy,gf,gb]);nref=len(gx);gi=np.tile(np.arange(ng),nref)
  gd=np.sqrt((np.repeat(gx,ng)-ax[gi])**2+(np.repeat(gy,ng)-ay[gi])**2+(np.repeat(gf*fh,ng)-az[gi])**2);gnf=np.abs(np.repeat(gf*fh,ng)-az[gi])/fh;gdb=(np.repeat(gb,ng)!=ab[gi]).astype(float)
  q=np.flatnonzero(te);qfp=np.where(heard[q],R[q],FLOOR_DBM).astype(np.float32);qxy=np.c_[x[q],y[q]]
  ii,jj=np.nonzero(heard[q]);i=q[ii];dist=np.sqrt((x[i]-ax[jj])**2+(y[i]-ay[jj])**2+(zz[i]-az[jj])**2);nf=np.abs(zz[i]-az[jj])/fh;db=(bld[i]!=ab[jj]).astype(float)
  for mode in ['guess','randomised']:
   for seed in range(5 if mode=='guess' else 1):
    gen=np.random.default_rng(3108+977*seed);pick=lambda key: PRI[key]['mu'] if mode=='randomised' else gen.normal(PRI[key]['mu'],PRI[key]['tau'])
    phi=np.zeros(ng+4);phi[0]=pick('offset');phi[1]=pick('exponent');phi[2:2+ng]=0 if mode=='randomised' else gen.normal(0,PRI['ap']['tau'],ng);phi[2+ng]=pick('floor');phi[3+ng]=pick('building')
    for lv in range(6):
     gp=predict(lv,phi,gd,gi,gnf,gdb,ng).reshape(nref,ng).astype(np.float32);gp[gp<SENS]=FLOOR_DBM;live=(gp>FLOOR_DBM).any(1)
     if live.sum()<5:continue
     xy=L.knn_localize(gp[live],np.c_[gx,gy][live],qfp);e=L.positioning_errors(xy,qxy)
     yhat=predict(lv,phi,dist,jj,nf,db,ng);rmse=float(np.sqrt(np.mean((yhat-R[i,jj])**2)))
     rows.append(dict(split=split,mode=mode,seed=seed,level=lv,median_m=float(np.median(e)),p90_m=float(np.percentile(e,90)),received_rssi_rmse_db=rmse))
  trainfp=np.where(heard[tr],R[tr],FLOOR_DBM).astype(np.float32);pred=L.knn_localize(trainfp,np.c_[x[tr],y[tr]],qfp);e=L.positioning_errors(pred,qxy)
  splits.append(dict(split=split,held_devices=hold.tolist(),n_train=int(tr.sum()),n_test=int(te.sum()),n_ap=ng,n_ref=nref,real_only_median_m=float(np.median(e))))
  (ROOT/'results/exp15_indoor_holdout.json').write_text(json.dumps(dict(rows=rows,splits=splits,runtime_s=time.time()-t,scope='Training-only AP locations and grid; held-out phones; observed-RSSI risk, not censored candidate risk.'),indent=2));print('split',split,'done',time.time()-t,flush=True)
if __name__=='__main__':main()
