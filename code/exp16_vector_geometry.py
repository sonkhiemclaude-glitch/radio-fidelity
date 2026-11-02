"""Exact polygon-ray geometry for the entire candidate and road-link population.

Vectorized intersections with a spatial index; one-dimensional interval unions
count connected occupied runs and eliminate raster discretization entirely.
"""
import concurrent.futures,json,time
from pathlib import Path
import numpy as np
import shapely,shapefile
from shapely.geometry import shape
from shapely.ops import transform
from shapely.strtree import STRtree
import dataset as D,simulator as S,tobit as TB,learners as L,priors as PR
from exp2_ladder import prior_draw,SEED,build_query,fingerprints_for
ROOT=Path(__file__).resolve().parents[1]

def exact_batch(x,y,gx,gy,gw,polys,tree):
 origin=np.c_[x,y];end=np.c_[gx[gw],gy[gw]];delta=end-origin;dist=np.linalg.norm(delta,axis=1);direction=delta/np.maximum(dist[:,None],1e-12)
 assert np.all(dist<100000)
 rays=shapely.linestrings(np.stack([origin,end],axis=1));ij=tree.query(rays,predicate='intersects');nc=np.zeros(len(x),int);dep=np.zeros(len(x))
 if ij.shape[1]==0:return nc,dep
 intersections=shapely.intersection(polys[ij[1]],rays[ij[0]])
 parts,idx=shapely.get_parts(intersections,return_index=True);parts2,idx2=shapely.get_parts(parts,return_index=True);rayid=ij[0][idx[idx2]]
 use=(shapely.get_type_id(parts2)==1)&(shapely.length(parts2)>1e-8);parts2=parts2[use];rayid=rayid[use]
 if not len(parts2):return nc,dep
 a=shapely.get_point(parts2,0);b=shapely.get_point(parts2,-1)
 p=np.c_[shapely.get_x(a),shapely.get_y(a)];q=np.c_[shapely.get_x(b),shapely.get_y(b)]
 av=((p-origin[rayid])*direction[rayid]).sum(1);bv=((q-origin[rayid])*direction[rayid]).sum(1)
 start=np.minimum(av,bv);stop=np.maximum(av,bv);order=np.lexsort((start,rayid));rayid=rayid[order];start=start[order];stop=stop[order]
 # Offsets allow one cumulative maximum to perform all per-ray interval unions.
 shifted=stop+rayid*100000.;previous=np.r_[-np.inf,np.maximum.accumulate(shifted)[:-1]]-rayid*100000.
 new=start>previous+1e-7;nc=np.bincount(rayid,weights=new,minlength=len(x)).astype(int)
 dep=np.bincount(rayid,weights=np.maximum(0,stop-np.maximum(start,previous)),minlength=len(x))
 return nc,dep

def main():
 t=time.time();g=S.Geometry(str(ROOT/'data/geometry.npz'));d=D.load();raw=np.load(ROOT/'data/lorawan.npz');ng=d['n_gw'];lat0=float(raw['lat0']);lon0=float(raw['lon0']);sx=np.pi/180*6371000*np.cos(np.deg2rad(lat0));sy=np.pi/180*6371000
 stem=ROOT/'raw/osm_be/gis_osm_buildings_a_free_1'
 import shpfast
 off=shpfast.read_shx(str(stem)+'.shx');bb=shpfast.bboxes(str(stem)+'.shp',off);lo=lon0+g.x_lo/sx;hi=lon0+(g.x_lo+g.nx*g.res)/sx;la=lat0+g.y_lo/sy;lb=lat0+(g.y_lo+g.ny*g.res)/sy
 select=np.flatnonzero((bb[:,2]>=lo)&(bb[:,0]<=hi)&(bb[:,3]>=la)&(bb[:,1]<=lb));reader=shapefile.Reader(str(stem),encoding='utf-8');polys=[]
 def enu(x,y,z=None):return (np.asarray(x)-lon0)*sx,(np.asarray(y)-lat0)*sy
 for i in select:
  p=transform(enu,shape(reader.shape(int(i)).__geo_interface__))
  if not p.is_valid:p=shapely.make_valid(p)
  polys.append(p)
 polys=np.array(polys,dtype=object);tree=STRtree(polys);c=D.candidates(d,g,cache=str(ROOT/'data/links_cand.npz'))
 def compute(name,x,y,gw):
  path=ROOT/f'data/links_{name}_vector.npz'
  dist=np.hypot(d['gw_x'][gw]-x,d['gw_y'][gw]-y)
  if path.exists():
   z=np.load(path);return dist,z['n_cross'],z['depth']
  tic=time.time();nc=np.zeros(len(x),int);dep=np.zeros(len(x));chunk=500
  def job(a):
   b=min(a+chunk,len(x));n,p=exact_batch(x[a:b],y[a:b],d['gw_x'],d['gw_y'],gw[a:b],polys,tree);return a,b,n,p
  with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
   for a,b,n,p in pool.map(job,range(0,len(x),chunk)):
    nc[a:b]=n;dep[a:b]=p
    if a%25000==0:print(name,a,'/',len(x),'seconds',time.time()-tic,flush=True)
  np.savez_compressed(path,d=dist,n_cross=nc,depth=dep);return dist,nc,dep
 dist,nc,dep=compute('cand',c['x'],c['y_pos'],c['gw'])
 ref=np.load(ROOT/'data/geometry_vector_reference.npz');assert np.array_equal(nc[ref['indices']],ref['n_cross']);assert np.allclose(dep[ref['indices']],ref['depth'],rtol=0,atol=1e-6)
 rx,ry=g.road_x,g.road_y;gi=np.tile(np.arange(ng),len(rx));px=np.repeat(rx,ng);py=np.repeat(ry,ng);gd,gn,gdep=compute('grid',px,py,gi)
 psi=S.features(dist,c['gw'],nc,dep,ng);base=S.free_space(dist);q=build_query(d);qfp=fingerprints_for(q,d,ng);qxy=np.c_[d['x'][q],d['y'][q]];zf=np.load(ROOT/'data/tobit_fits.npz')
 field=S.ShadowField(rx.min()-500,ry.min()-500,rx.max()+500,ry.max()+500,25.,1.,PR.SHADOW_ELL,ng,seed=SEED);shadow=field(px,py,gi)
 risk=[];loc=[];fits=[]
 for lv in range(6):
  cols=S.level_columns(lv,ng);rv=[];ev=[]
  for rep in range(20):
   phi,sig=prior_draw('uninformed','guess',lv,np.random.default_rng(SEED+977*rep),ng,zf['phi'][lv],float(zf['sigma'][lv]));rv.append(TB.risk(phi[cols],sig,psi[:,cols],base,c['y'],c['got'],c['thr']))
   fp=S.rssi(lv,phi,gd,gi,gn,gdep,ng,sig*shadow).reshape(-1,ng).astype(np.float32);fp[fp<S.SENS[7]]=L.FLOOR;live=(fp>L.FLOOR).any(1);pred=L.knn_localize(fp[live],np.c_[rx,ry][live],qfp);ev.append(float(np.median(L.positioning_errors(pred,qxy))))
  risk.append(rv);loc.append(ev)
  if lv>=4:
   ph,sg,f=TB.fit_best(np.ascontiguousarray(psi[:,cols]),base,c['y'],c['got'],c['thr'],phi0=zf['phi'][lv][cols],sigma0=float(zf['sigma'][lv]));fits.append(dict(level=lv,nll=float(f),sigma=float(sg),coefficients=ph.tolist()))
 out=dict(snapshot='2026-08-01',n_buildings=len(polys),n_candidate_links=len(dist),n_grid_links=len(gd),n_query=len(q),n_rep=20,risk_mean=np.mean(risk,1).tolist(),risk_reps=risk,loc_median_mean=np.mean(loc,1).tolist(),loc_median_reps=loc,link_argmin=int(np.argmin(np.mean(risk,1))),loc_argmin=int(np.argmin(np.mean(loc,1))),joint_fits=fits,runtime_s=time.time()-t,reference_checks='3,000 scalar interval unions match bulk exact computation')
 (ROOT/'results/exp16_vector_geometry.json').write_text(json.dumps(out,indent=2));print(out,flush=True)
if __name__=='__main__':main()
