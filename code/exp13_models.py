"""Paired simulator-fidelity evaluation with stronger downstream learners.

Residual MLP and receiver-geometry Transformer, free/equal synthetic corpus arms,
three paired seeds; raw and road-projected errors. All models use only simulator
labels in transfer arms. Real-only baselines use exactly the original query set
and blocked five-fold splits. Fixed optimization, no test-label tuning.
"""
import argparse,json,time,os
from pathlib import Path
import numpy as np
import torch
from torch import nn
from scipy.spatial import cKDTree
import dataset as D,simulator as S,learners as L,priors as PR
from exp2_ladder import prior_draw,SEED,build_query,fingerprints_for
ROOT=Path(__file__).resolve().parents[1]

class ResidualMLP(nn.Module):
 def __init__(self,ng,coords):
  super().__init__();self.input=nn.Linear(2*ng,256);self.blocks=nn.ModuleList([nn.Sequential(nn.LayerNorm(256),nn.Linear(256,256),nn.GELU(),nn.Dropout(.1),nn.Linear(256,256)) for _ in range(3)]);self.head=nn.Linear(256,2)
 def forward(self,x):
  h=self.input(torch.cat([x,(x>-1.399).float()],-1))
  for block in self.blocks:h=h+block(h)
  return self.head(h)

class GeometryTransformer(nn.Module):
 def __init__(self,ng,coords):
  super().__init__();self.register_buffer('coords',coords);self.token=nn.Linear(4,64);self.receiver=nn.Parameter(torch.randn(1,ng,64)*.02);self.cls=nn.Parameter(torch.zeros(1,1,64))
  layer=nn.TransformerEncoderLayer(64,4,128,.1,activation='gelu',batch_first=True,norm_first=True)
  self.enc=nn.TransformerEncoder(layer,2,enable_nested_tensor=False);self.head=nn.Sequential(nn.LayerNorm(64),nn.Linear(64,128),nn.GELU(),nn.Linear(128,2))
 def forward(self,x):
  coords=self.coords[None].expand(x.shape[0],-1,-1)
  token=self.token(torch.cat([x[...,None],(x>-1.399).float()[...,None],coords],-1))+self.receiver
  h=self.enc(torch.cat([self.cls.expand(x.shape[0],-1,-1),token],1));return self.head(h[:,0])

def fit_predict(kind,X,Y,Q,coords,seed,epochs,device):
 torch.manual_seed(seed);np.random.seed(seed);model=(ResidualMLP if kind=='mlp' else GeometryTransformer)(X.shape[1],torch.tensor(coords,dtype=torch.float32)).to(device)
 xt=torch.tensor(X/100.,device=device);yt=torch.tensor(Y,device=device,dtype=torch.float32);qt=torch.tensor(Q/100.,device=device)
 opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4)
 for epoch in range(epochs):
  model.train();ix=torch.randperm(len(X),device=device)
  for start in range(0,len(X),1024):
   b=ix[start:start+1024];opt.zero_grad(set_to_none=True)
   loss=(model(xt[b])-yt[b]).square().mean();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5);opt.step()
 model.eval()
 with torch.no_grad():out=torch.cat([model(qt[s:s+1024]) for s in range(0,len(Q),1024)]).cpu().numpy()
 return out

def summary(pred,target,roadtree,roadxy):
 e=np.linalg.norm(pred-target,axis=1);_,idx=roadtree.query(pred);p=roadxy[idx];ep=np.linalg.norm(p-target,axis=1)
 return dict(median_m=float(np.median(e)),mean_m=float(e.mean()),p90_m=float(np.percentile(e,90)),projected_median_m=float(np.median(ep)),projected_p90_m=float(np.percentile(ep,90)))

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--epochs',type=int,default=60);ap.add_argument('--seeds',type=int,default=3);ap.add_argument('--arms',nargs='+',default=['free','equal']);ap.add_argument('--skip-real',action='store_true');ap.add_argument('--refresh-equal',action='store_true');args=ap.parse_args()
 t=time.time();torch.set_num_threads(4);device='cuda' if torch.cuda.is_available() else 'cpu';print('DEVICE',device,flush=True)
 d=D.load();ng=d['n_gw'];g=S.Geometry(str(ROOT/'data/geometry.npz'));rx,ry=g.road_x,g.road_y;xy=np.c_[rx,ry];center=xy.mean(0);scale=3000.;coords=(np.c_[d['gw_x'],d['gw_y']]-center)/scale;tree=cKDTree(xy)
 q=build_query(d);qfp=fingerprints_for(q,d,ng);qxy=np.c_[d['x'][q],d['y'][q]];fold=D.block_folds(qxy[:,0],qxy[:,1],n_folds=5,seed=SEED)
 gi=np.tile(np.arange(ng),len(xy));px=np.repeat(rx,ng);py=np.repeat(ry,ng)
 gd,gn,dep=D.link_geometry(px,py,d['gw_x'],d['gw_y'],gi,g,cache=str(ROOT/'data/links_grid.npz'));zf=np.load(ROOT/'data/tobit_fits.npz')
 # Four paired shadow realizations per coefficient seed, shared across levels.
 shadows=[]
 for j in range(4):
  f=S.ShadowField(rx.min()-500,ry.min()-500,rx.max()+500,ry.max()+500,25.,1.,PR.SHADOW_ELL,ng,seed=SEED+100*j);shadows.append(f(px,py,gi))
 # Determine one common cap before model fitting, using simulator output only.
 min_live=len(xy)
 for ss in range(args.seeds):
  for mm in ['guess','randomised','oracle']:
   for ll in range(6):
    pp,sg=prior_draw('uninformed',mm,ll,np.random.default_rng(SEED+977*ss),ng,zf['phi'][ll],float(zf['sigma'][ll]))
    for shadow in shadows:
     fp=S.rssi(ll,pp,gd,gi,gn,dep,ng,sg*shadow).reshape(-1,ng)
     min_live=min(min_live,int((fp>=S.SENS[7]).any(1).sum()))
 common_cap=min(1000,min_live)
 if common_cap<32:raise ValueError('Too few common audible reference points')
 print('COMMON_REFERENCE_CAP',common_cap,flush=True)
 path=ROOT/'results/exp13_models.json';out=json.loads(path.read_text()) if path.exists() else dict(rows=[],real_rows=[],protocol=dict(n_query=len(q),spreading_factor=7,n_folds=5,block_m=500,epochs=args.epochs,n_seeds=args.seeds,shadow_realizations=4,equal_reference_cap=1000,device=device,architectures='Residual MLP 3x256; geometry Transformer 2x64, 4 heads; AdamW lr=0.001; batch=1024; fixed epochs; no query-label tuning'))
 if out['protocol']['epochs']!=args.epochs or out['protocol']['n_seeds']!=args.seeds:
  raise ValueError('Checkpoint optimization/seed protocol differs; move the JSON and predictions aside before a new run')
 if not args.refresh_equal and out['rows'] and out['protocol']['equal_reference_cap']!=common_cap:
  raise ValueError('Checkpoint equal-arm cap differs; use --refresh-equal to recompute that arm')
 out.pop('runtime_s',None)
 if args.refresh_equal:
  archive=ROOT/'results/intermediate';archive.mkdir(exist_ok=True)
  (archive/'exp13_capped1000.json').write_text(json.dumps(out,indent=2))
  out['rows']=[r for r in out['rows'] if r['arm']!='equal']
 out['protocol']['equal_reference_cap']=common_cap
 out['protocol']['minimum_audible_reference_count']=min_live
 done={(r['arm'],r['mode'],r['level'],r['seed'],r['model']) for r in out['rows']}
 def save():
  out['last_invocation_runtime_s']=time.time()-t;out['total_simulator_model_runtime_s']=sum(r['runtime_s'] for r in out['rows']);tmp=path.with_suffix('.json.tmp');tmp.write_text(json.dumps(out,indent=2));tmp.replace(path)
 for seed in range(args.seeds):
  for mode in ['guess','randomised','oracle']:
   for lv in range(6):
    phi,sig=prior_draw('uninformed',mode,lv,np.random.default_rng(SEED+977*seed),ng,zf['phi'][lv],float(zf['sigma'][lv]));db=[];ys=[];ids=[]
    for shadow in shadows:
     fp=S.rssi(lv,phi,gd,gi,gn,dep,ng,sig*shadow).reshape(-1,ng).astype(np.float32);fp[fp<S.SENS[7]]=L.FLOOR;live=(fp>L.FLOOR).any(1);db.append(fp[live]);ys.append(xy[live]);ids.append(np.flatnonzero(live))
    for arm in args.arms:
     xx=[];yy=[]
     # Equal reference cap separately per realization, deterministic paired RNG.
     for j,(X,Y) in enumerate(zip(db,ys)):
      ix=np.arange(len(X))
      if arm=='equal':ix=np.random.default_rng(SEED+seed*100+j).choice(len(X),common_cap,replace=False)
      xx.append(X[ix]);yy.append(Y[ix])
     X=np.concatenate(xx);Y=np.concatenate(yy)
     for kind in ['knn','mlp','transformer']:
      key=(arm,mode,lv,seed,kind)
      if key in done:continue
      tic=time.time()
      if kind=='knn':pred=L.knn_localize(X,Y,qfp)
      else:pred=fit_predict(kind,X,(Y-center)/scale,qfp,coords,SEED+seed,args.epochs,device)*scale+center
      stats=summary(pred,qxy,tree,xy)
      row=dict(arm=arm,mode=mode,level=lv,seed=seed,model=kind,n_train=len(X),runtime_s=time.time()-tic,**stats);out['rows'].append(row)
      # Per-query errors support paired spatial-bootstrap comparisons and a
      # downstream admission experiment without retraining on campaign labels.
      np.savez_compressed(ROOT/f'data/pred_{arm}_{mode}_F{lv}_s{seed}_{kind}.npz',prediction=pred,error=np.linalg.norm(pred-qxy,axis=1),query=q,fold=fold)
      done.add(key);save();print(row,flush=True)
 if not args.skip_real:
  realdone={(r['seed'],r['model']) for r in out['real_rows']}
  for seed in range(args.seeds):
   for kind in ['knn','mlp','transformer']:
    if (seed,kind) in realdone:continue
    pred=np.zeros_like(qxy)
    for f in range(5):
     tr=fold!=f;te=fold==f
     if kind=='knn':p=L.knn_localize(qfp[tr],qxy[tr],qfp[te])
     else:p=fit_predict(kind,qfp[tr],(qxy[tr]-center)/scale,qfp[te],coords,SEED+seed,args.epochs,device)*scale+center
     pred[te]=p
    row=dict(seed=seed,model=kind,**summary(pred,qxy,tree,xy));out['real_rows'].append(row);save();print('REAL',row,flush=True)
    np.savez_compressed(ROOT/f'data/pred_real_s{seed}_{kind}.npz',prediction=pred,error=np.linalg.norm(pred-qxy,axis=1),query=q,fold=fold)
 save()
if __name__=='__main__':main()
