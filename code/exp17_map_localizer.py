"""Direct map-input localization with a paired no-map ablation.

RSSI/reception query MLP scores the fixed road candidates. Candidate embeddings
encode gateway distances, coordinates and a CNN over building/road patches.
All map inputs depend only on published geometry, never query positions.
Equal synthetic corpora exactly match exp13 (799 points x four shadow fields).
Soft spatial labels at 100 m reflect the 50 m road spacing; fixed 40 epochs.
"""
import argparse,json,time,hashlib
from pathlib import Path
import numpy as np
import torch
from torch import nn
from scipy.ndimage import uniform_filter
from scipy.spatial import cKDTree
import dataset as D,simulator as S,learners as L,priors as PR
from exp2_ladder import prior_draw,SEED,build_query,fingerprints_for
from exp13_models import summary
ROOT=Path(__file__).resolve().parents[1]

def map_patches(g):
 path=ROOT/'data/map_candidate_patches.npz'
 if path.exists():
  z=np.load(path);assert np.array_equal(z['road_x'],g.road_x) and np.array_equal(z['road_y'],g.road_y)
  return z['patches']
 roads=np.zeros(g.occ.shape,np.float32)
 ix=np.floor((g.road_x-g.x_lo)/g.res).astype(int);iy=np.floor((g.road_y-g.y_lo)/g.res).astype(int)
 roads[np.clip(iy,0,g.ny-1),np.clip(ix,0,g.nx-1)]=1
 channels=[]
 for step in [25.,100.]:
  size=int(step/g.res)
  # Fractional building occupancy and presence of a road reference per patch cell.
  for raster,factor in [(g.occ.astype(np.float32),1.),(roads,float(size*size))]:
   pooled=np.clip(uniform_filter(raster,size=size,mode='constant')*factor,0,1)
   delta=np.arange(-4,5)*step/g.res
   xx=np.rint(ix[:,None,None]+delta[None,None,:]).astype(int)
   yy=np.rint(iy[:,None,None]+delta[None,:,None]).astype(int)
   valid=(xx>=0)&(xx<g.nx)&(yy>=0)&(yy<g.ny)
   channels.append(np.where(valid,pooled[np.clip(yy,0,g.ny-1),np.clip(xx,0,g.nx-1)],0))
 patches=np.stack(channels,1).astype(np.float32)
 np.savez_compressed(path,patches=patches,road_x=g.road_x,road_y=g.road_y)
 return patches

class MapCandidateLocalizer(nn.Module):
 def __init__(self,ng,candidate_features,patches,xy,use_map):
  super().__init__();self.use_map=use_map
  self.register_buffer('features',torch.as_tensor(candidate_features,dtype=torch.float32))
  self.register_buffer('patches',torch.as_tensor(patches if use_map else np.zeros_like(patches),dtype=torch.float32))
  self.register_buffer('xy',torch.as_tensor(xy,dtype=torch.float32))
  self.query=nn.Sequential(nn.Linear(ng*2,256),nn.GELU(),nn.LayerNorm(256),nn.Linear(256,128),nn.GELU(),nn.Linear(128,64))
  self.cnn=nn.Sequential(nn.Conv2d(4,8,3,padding=1),nn.GELU(),nn.Conv2d(8,8,3,stride=3,padding=1),nn.GELU(),nn.Flatten(),nn.Linear(8*3*3,32),nn.GELU())
  self.candidate=nn.Sequential(nn.Linear(candidate_features.shape[1]+32,128),nn.GELU(),nn.Linear(128,64))
 def keys(self):return self.candidate(torch.cat([self.features,self.cnn(self.patches)],1))
 def forward(self,x,keys=None):
  q=self.query(torch.cat([x,(x>-1.399).float()],1));k=self.keys() if keys is None else keys
  return q@k.T/8

def fit_predict(X,Y,Q,feat,patches,xy,seed,use_map,epochs,device,smoke=False):
 torch.manual_seed(seed);model=MapCandidateLocalizer(X.shape[1],feat,patches,xy,use_map).to(device)
 xt=torch.as_tensor(X/100.,device=device);qt=torch.as_tensor(Q/100.,device=device)
 yy=torch.as_tensor(Y,dtype=torch.float32,device=device)
 # Fixed Gaussian label width 100 m in normalized coordinates (3 km scale).
 target=torch.softmax(-torch.cdist(yy,model.xy).square()/(2*(100/3000)**2),dim=1)
 opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4);history=[];start=time.time();map_gradient=None
 for epoch in range(epochs):
  model.train();ix=torch.randperm(len(X),device=device);losses=[]
  for a in range(0,len(X),1024):
   ids=ix[a:a+1024];opt.zero_grad(set_to_none=True);logits=model(xt[ids]);loss=-(target[ids]*logits.log_softmax(1)).sum(1).mean()
   assert torch.isfinite(loss)
   loss.backward()
   if map_gradient is None:map_gradient=float(model.cnn[0].weight.grad.abs().sum().detach())
   nn.utils.clip_grad_norm_(model.parameters(),5);opt.step();losses.append(float(loss.detach()))
  history.append(float(np.mean(losses)))
 model.eval()
 with torch.no_grad():
  keys=model.keys();pred=torch.cat([torch.softmax(model(qt[a:a+512],keys),1)@model.xy for a in range(0,len(Q),512)]).cpu().numpy()
 if smoke:
  assert (map_gradient>0) if use_map else (map_gradient==0)
 return pred,dict(runtime_s=time.time()-start,train_cross_entropy=history,initial_map_gradient_l1=map_gradient)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--epochs',type=int,default=40);ap.add_argument('--smoke',action='store_true');args=ap.parse_args()
 torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=True;device='cuda' if torch.cuda.is_available() else 'cpu'
 t=time.time();d=D.load();g=S.Geometry(str(ROOT/'data/geometry.npz'));xy=np.c_[g.road_x,g.road_y];center=xy.mean(0);scale=3000.;xyn=(xy-center)/scale
 patches=map_patches(g);gwxy=np.c_[d['gw_x'],d['gw_y']]
 dist=np.linalg.norm(xy[:,None,:]-gwxy[None,:,:],axis=2)/scale
 freq=np.array([1.,2.,4.,8.]);phase=xyn[:,:,None]*freq[None,None,:]*np.pi
 feat=np.c_[xyn,dist,np.sin(phase).reshape(len(xy),-1),np.cos(phase).reshape(len(xy),-1)].astype(np.float32)
 q=build_query(d);Q=fingerprints_for(q,d,d['n_gw']);qxy=np.c_[d['x'][q],d['y'][q]];fold=D.block_folds(qxy[:,0],qxy[:,1],n_folds=5,seed=SEED);tree=cKDTree(xy)
 ng=d['n_gw'];gi=np.tile(np.arange(ng),len(xy));px=np.repeat(xy[:,0],ng);py=np.repeat(xy[:,1],ng)
 gd,gn,dep=D.link_geometry(px,py,d['gw_x'],d['gw_y'],gi,g,cache=str(ROOT/'data/links_grid.npz'));zf=np.load(ROOT/'data/tobit_fits.npz')
 cap=json.loads((ROOT/'results/exp13_models.json').read_text())['protocol']['equal_reference_cap']
 shadows=[]
 for j in range(4):
  field=S.ShadowField(xy[:,0].min()-500,xy[:,1].min()-500,xy[:,0].max()+500,xy[:,1].max()+500,25.,1.,PR.SHADOW_ELL,ng,seed=SEED+100*j);shadows.append(field(px,py,gi))
 path=ROOT/'results/exp17_map_localizer.json'
 protocol=dict(epochs=args.epochs,seeds=3,arm='equal',equal_reference_cap=cap,n_train=4*cap,n_query=len(q),road_candidates=len(xy),map_patch_shape=list(patches.shape[1:]),map_cell_sizes_m=[25,100],map_channels='building fraction and road-point presence at each scale',label_sigma_m=100,optimizer='AdamW lr=.001 wd=.0001 batch=1024 gradient_clip=5',device=device,tf32=True,query_sha256=hashlib.sha256(q.tobytes()).hexdigest(),scope='Map-aware candidate scoring, direct map CNN and paired zero-map ablation; no query-label model selection')
 out=json.loads(path.read_text()) if path.exists() and not args.smoke else dict(protocol=protocol,rows=[],real_rows=[])
 if out['protocol']!=protocol:raise ValueError('Checkpoint protocol differs')
 def save():
  out['runtime_s']=time.time()-t;tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(out,indent=2));tmp.replace(path)
 done={(r['mode'],r['level'],r['seed'],r['map_input']) for r in out['rows']}
 for seed in range(3):
  for mode in ['guess','oracle']:
   for lv in range(6):
    phi,sig=prior_draw('uninformed',mode,lv,np.random.default_rng(SEED+977*seed),ng,zf['phi'][lv],float(zf['sigma'][lv]));xx=[];yy=[]
    for j,shadow in enumerate(shadows):
     fp=S.rssi(lv,phi,gd,gi,gn,dep,ng,sig*shadow).reshape(-1,ng).astype(np.float32);fp[fp<S.SENS[7]]=L.FLOOR;live=(fp>L.FLOOR).any(1)
     ix=np.random.default_rng(SEED+seed*100+j).choice(live.sum(),cap,replace=False);xx.append(fp[live][ix]);yy.append(xyn[live][ix])
    X=np.concatenate(xx);Y=np.concatenate(yy)
    for use_map in [False,True]:
     tag='map' if use_map else 'no_map';key=(mode,lv,seed,tag)
     if key in done:continue
     if args.smoke:
      pred,train=fit_predict(X[:128],Y[:128],Q[:32],feat,patches,xyn,SEED+seed,use_map,2,device,True);print('SMOKE',tag,train,flush=True)
     else:
      pred,train=fit_predict(X,Y,Q,feat,patches,xyn,SEED+seed,use_map,args.epochs,device);pred=pred*scale+center
      row=dict(mode=mode,level=lv,seed=seed,map_input=tag,n_train=len(X),**summary(pred,qxy,tree,xy),**train)
      np.savez_compressed(ROOT/f'data/pred_mapcnn_{tag}_{mode}_F{lv}_s{seed}.npz',prediction=pred,error=np.linalg.norm(pred-qxy,axis=1),query=q,fold=fold)
      out['rows'].append(row);save();print({k:v for k,v in row.items() if k!='train_cross_entropy'},flush=True)
    if args.smoke:return
 realdone={(r['seed'],r['map_input']) for r in out['real_rows']}
 for seed in range(3):
  for use_map in [False,True]:
   tag='map' if use_map else 'no_map'
   if (seed,tag) in realdone:continue
   pred=np.zeros_like(qxy);hist=[];tic=time.time()
   for f in range(5):
    tr=fold!=f;te=fold==f
    p,train=fit_predict(Q[tr],(qxy[tr]-center)/scale,Q[te],feat,patches,xyn,SEED+seed,use_map,args.epochs,device);pred[te]=p*scale+center;hist.append(train)
   row=dict(seed=seed,map_input=tag,runtime_s=time.time()-tic,fold_training=hist,**summary(pred,qxy,tree,xy))
   np.savez_compressed(ROOT/f'data/pred_mapcnn_{tag}_real_s{seed}.npz',prediction=pred,error=np.linalg.norm(pred-qxy,axis=1),query=q,fold=fold)
   out['real_rows'].append(row);save();print('REAL',tag,seed,row['median_m'],flush=True)
 print('COMPLETE',len(out['rows']),len(out['real_rows']),time.time()-t,flush=True)
if __name__=='__main__':main()
