"""Correctly ranged admission bounds with message and spatial cluster units.

Design-based certification targets the fixed empirical candidate population.
Independent draws WITH replacement of messages, or spatial blocks with PPS,
make the cluster-average differences independent conditional on the corpus.
PPS = probability proportional to number of messages, preserving link weighting.
Without-replacement campaigns are also reported as empirical stress checks only.
"""
import json,time,os
from pathlib import Path
import numpy as np
from scipy.stats import beta
import dataset as D, simulator as S, tobit as TB, theory as T
from exp2_ladder import prior_draw, SEED
ROOT=Path(__file__).resolve().parents[1]
B=int(os.environ.get('ADMISSION_CAMPAIGNS','2000'))

def interval(k,n):
 return [0.0 if k==0 else float(beta.ppf(.025,k,n-k+1)),1.0 if k==n else float(beta.ppf(.975,k+1,n-k))]

def selected(v,cap):
 # v: campaign x sampled independent units x level
 d=np.diff(v,axis=2); n=v.shape[1];log=np.log(40.)
 u=d.mean(1)+np.sqrt(2*d.var(1,ddof=1)*log/n)+7*(2*cap)*log/(3*(n-1))
 return np.cumprod(u<0,axis=1).sum(1),v.mean(1).argmin(1)

def main():
 start=time.time(); d=D.load();g=S.Geometry(str(ROOT/'data/geometry.npz'))
 c=D.candidates(d,g,cache=str(ROOT/'data/links_cand.npz'))
 psi=S.features(c['dist'],c['gw'],c['n_cross'],c['depth'],d['n_gw']);base=S.free_space(c['dist'])
 z=np.load(ROOT/'data/tobit_fits.npz');rows=[];populations=[]
 message_blocks=D.block_id(d['x'][c['msgs']],d['y'][c['msgs']])
 ub,inv=np.unique(message_blocks,return_inverse=True);counts=np.bincount(inv)
 for seed in range(5):
  rng0=np.random.default_rng(SEED+977*seed)
  phi,sig=prior_draw('uninformed','guess',0,rng0,d['n_gw'],z['phi'][0],float(z['sigma'][0]))
  raw=np.stack([TB.link_loss(phi[S.level_columns(l,d['n_gw'])],sig,psi[:,S.level_columns(l,d['n_gw'])],base,c['y'],c['got'],c['thr']) for l in range(6)],1)
  for cap in [8.,15.,30.,60.]:
   loss=np.clip(raw,0,cap);truth=loss.mean(0);bad=np.r_[False,np.maximum.accumulate(np.diff(truth)>=0)]
   msg=loss.reshape(-1,d['n_gw'],6).mean(1)
   block=np.stack([np.bincount(inv,weights=msg[:,l])/counts for l in range(6)],1)
   populations.append(dict(seed=seed,cap=cap,risk=truth.tolist(),global_argmin=int(truth.argmin()),first_nonbeneficial=int(np.flatnonzero(bad)[0]-1) if bad.any() else 5))
   for design,units,prob,ns in [('message_wr',msg,None,[32,64,128,256,512,1000,2000]),('spatial_pps_wr',block,counts/counts.sum(),[32,64,128,256,512,1000,2000])]:
    for n in ns:
     rng=np.random.default_rng(SEED+seed*10000+n);sel=[];arg=[]
     # Bound memory while preserving an explicit independent sampling design.
     for offset in range(0,B,100):
      idx=rng.choice(len(units),size=(min(100,B-offset),n),replace=True,p=prob)
      a,b=selected(units[idx],cap);sel.extend(a.tolist());arg.extend(b.tolist())
     sel=np.array(sel);arg=np.array(arg);k=int(bad[sel].sum())
     rows.append(dict(seed=seed,cap=cap,design=design,n_units=n,links_per_message=d['n_gw'],B=B,harmful_count=k,harmful_rate=k/B,harmful_ci95=interval(k,B),exact_global=float(np.mean(sel==truth.argmin())),mean_level=float(sel.mean()),selected_counts=np.bincount(sel,minlength=6).tolist(),argmin_harmful=float(bad[arg].mean()),argmin_exact=float(np.mean(arg==truth.argmin()))))
   # Intact-message and intact-spatial-block WOR stress checks, main cap only.
   if cap==15:
    for design,units in [('message_wor',msg),('spatial_uniform_wor',block)]:
     for n in [32,128,512]:
      if n>len(units):continue
      rng=np.random.default_rng(SEED+seed+n);sels=[]
      for _ in range(B):
       idx=rng.choice(len(units),n,replace=False);a,_=selected(units[idx][None],cap);sels.append(int(a[0]))
      # Uniform spatial target differs from link target; score against its own mean.
      target=units.mean(0);bh=np.r_[False,np.maximum.accumulate(np.diff(target)>=0)];k=int(bh[sels].sum())
      rows.append(dict(seed=seed,cap=cap,design=design,n_units=n,B=B,harmful_count=k,harmful_rate=k/B,harmful_ci95=interval(k,B),exact_global=float(np.mean(np.array(sels)==target.argmin())),mean_level=float(np.mean(sels)),target='message-weighted' if design=='message_wor' else 'uniform-spatial-block',certified=False))
   print('seed',seed,'clip',cap,'done',round(time.time()-start,1),flush=True)
 out=dict(B=B,alpha=.05,loss_difference_range='2 * clip',n_messages=len(msg),n_spatial_blocks=len(block),n_gateways=d['n_gw'],sampling='Independent with-replacement message draws; independent PPS block draws. Guarantees conditional on fixed corpus, not unseen deployment.',populations=populations,rows=rows,runtime_s=time.time()-start)
 (ROOT/'results/exp11_cluster_admission.json').write_text(json.dumps(out,indent=2))
if __name__=='__main__':main()
