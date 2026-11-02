"""Pinned-snapshot 5/2/1m geometry convergence, with exact vector spot checks.

Caches are resolution-specific. Original submission caches are never overwritten.
Road reference points and evaluated candidate messages are held fixed.
"""
import json,time,zipfile,os
from pathlib import Path
import numpy as np
import dataset as D, simulator as S, tobit as TB
from exp2_ladder import prior_draw,SEED,build_query,fingerprints_for
import learners as L
ROOT=Path(__file__).resolve().parents[1]

def main():
 import shapefile, shapely
 from shapely.geometry import shape,LineString
 from shapely.ops import transform,unary_union
 from shapely.strtree import STRtree
 from rasterio.features import rasterize
 from affine import Affine
 t=time.time();raw=ROOT/'raw';ext=raw/'osm_be';ext.mkdir(exist_ok=True)
 with zipfile.ZipFile(raw/'belgium-260801-free.shp.zip') as z:
  for name in z.namelist():
   if Path(name).name.startswith('gis_osm_buildings_a_free_1.'):
    out=ext/Path(name).name
    if not out.exists():out.write_bytes(z.read(name))
 old=np.load(ROOT/'data/geometry.npz'); d=D.load();geom0=S.Geometry(str(ROOT/'data/geometry.npz'))
 xlo=float(old['x_lo']);ylo=float(old['y_lo']);ny,nx=old['occ_shape'];xhi=xlo+nx*float(old['res']);yhi=ylo+ny*float(old['res'])
 rawd=np.load(ROOT/'data/lorawan.npz');lat0=float(rawd['lat0']);lon0=float(rawd['lon0']);R=6371000.;sx=np.pi/180*R*np.cos(np.deg2rad(lat0));sy=np.pi/180*R
 def enu(x,y,z=None):return (np.asarray(x)-lon0)*sx,(np.asarray(y)-lat0)*sy
 reader=shapefile.Reader(str(ext/'gis_osm_buildings_a_free_1'),encoding='utf-8')
 import shpfast
 off=shpfast.read_shx(str(ext/'gis_osm_buildings_a_free_1.shx'));bb=shpfast.bboxes(str(ext/'gis_osm_buildings_a_free_1.shp'),off)
 lo=lon0+xlo/sx;hi=lon0+xhi/sx;la=lat0+ylo/sy;lb=lat0+yhi/sy
 ix=np.flatnonzero((bb[:,2]>=lo)&(bb[:,0]<=hi)&(bb[:,3]>=la)&(bb[:,1]<=lb))
 polys=[]
 for i in ix:
  p=transform(enu,shape(reader.shape(int(i)).__geo_interface__))
  if not p.is_valid:p=shapely.make_valid(p)
  if not p.is_empty:polys.append(p)
 print('polygons',len(polys),'elapsed',time.time()-t,flush=True)
 c=D.candidates(d,geom0,cache=str(ROOT/'data/links_cand.npz'));ng=d['n_gw'];base=S.free_space(c['dist'])
 rx,ry=geom0.road_x,geom0.road_y;gi=np.tile(np.arange(ng),len(rx));px=np.repeat(rx,ng);py=np.repeat(ry,ng)
 q=build_query(d);qfp=fingerprints_for(q,d,ng);qxy=np.c_[d['x'][q],d['y'][q]]
 # Use original shadow correlation length rather than a new protocol choice.
 import priors as PR
 field=S.ShadowField(rx.min()-500,ry.min()-500,rx.max()+500,ry.max()+500,25.,1.,PR.SHADOW_ELL,ng,seed=SEED)
 shadow=field(px,py,gi);zf=np.load(ROOT/'data/tobit_fits.npz')
 # Vector reference uses a union of per-polygon intersections, matching occupied
 # runs rather than counting adjacent polygons as separate buildings.
 rng=np.random.default_rng(SEED);sample=np.sort(rng.choice(len(c['dist']),3000,replace=False));tree=STRtree(polys);exact_nc=[];exact_dep=[]
 for idx in sample:
  ray=LineString([(c['x'][idx],c['y_pos'][idx]),(d['gw_x'][c['gw'][idx]],d['gw_y'][c['gw'][idx]])])
  hits=tree.query(ray,predicate='intersects');parts=[polys[j].intersection(ray) for j in hits]
  # Count connected occupied runs on the one-dimensional ray. Unary union can
  # leave connected segments split at polygon boundaries, so counting its
  # geometries directly overcounts adjacent/overlapping footprints.
  origin=np.asarray(ray.coords[0]);direction=np.asarray(ray.coords[-1])-origin;length=np.linalg.norm(direction);direction/=length
  intervals=[]
  def collect(x):
   if x.is_empty:return
   if x.geom_type=='LineString' and x.length>1e-8:
    pts=np.asarray(x.coords);v=(pts[[0,-1]]-origin)@direction;intervals.append((float(v.min()),float(v.max())))
   else:
    for v in getattr(x,'geoms',[]):collect(v)
  for part in parts:collect(part)
  merged=[]
  for start,stop in sorted(intervals):
   if merged and start<=merged[-1][1]+1e-7:merged[-1][1]=max(merged[-1][1],stop)
   else:merged.append([start,stop])
  exact_nc.append(len(merged));exact_dep.append(sum(stop-start for start,stop in merged))
 exact_nc=np.array(exact_nc);exact_dep=np.array(exact_dep)
 np.savez_compressed(ROOT/'data/geometry_vector_reference.npz',indices=sample,n_cross=exact_nc,depth=exact_dep)
 result_path=ROOT/'results/exp12_geometry.json'
 rows=json.loads(result_path.read_text())['rows'] if result_path.exists() else []
 for row in rows:
  cached=np.load(ROOT/f"data/links_cand_r{row['resolution_m']:g}.npz")
  row.update(crossing_mae_vector=float(np.mean(np.abs(cached['n_cross'][sample]-exact_nc))),crossing_bias_vector=float(np.mean(cached['n_cross'][sample]-exact_nc)),depth_mae_vector_m=float(np.mean(np.abs(cached['depth'][sample]-exact_dep))),depth_bias_vector_m=float(np.mean(cached['depth'][sample]-exact_dep)))
 if rows:result_path.write_text(json.dumps(dict(rows=rows,n_exact_links=len(sample),n_rep=20,n_query=len(q),vector_reference='Merged one-dimensional polygon-intersection intervals',runtime_s=time.time()-t),indent=2))
 for res in [5.,2.,1.]:
  if any(row['resolution_m']==res for row in rows):continue
  path=ROOT/f'data/geometry_r{res:g}.npz'
  if not path.exists():
   occ=rasterize(((p,1) for p in polys),out_shape=(int(np.ceil((yhi-ylo)/res)),int(np.ceil((xhi-xlo)/res))),transform=Affine(res,0,xlo,0,res,ylo),fill=0,dtype='uint8').astype(bool)
   np.savez_compressed(path,occ=np.packbits(occ),occ_shape=occ.shape,x_lo=xlo,y_lo=ylo,res=res,road_x=rx,road_y=ry,n_buildings=len(polys),keep_gw=old['keep_gw'])
   del occ
  geom=S.Geometry(str(path));nc,dep=geom.los(c['x'],c['y_pos'],d['gw_x'][c['gw']],d['gw_y'][c['gw']]);dist=c['dist']
  np.savez_compressed(ROOT/f'data/links_cand_r{res:g}.npz',d=dist,n_cross=nc,depth=dep)
  gd,gn,gdep=D.link_geometry(px,py,d['gw_x'],d['gw_y'],gi,geom,cache=str(ROOT/f'data/links_grid_r{res:g}.npz'))
  psi=S.features(dist,c['gw'],nc,dep,ng);fits=[];loc=[];risk=[]
  for lv in range(6):
   cols=S.level_columns(lv,ng)
   if lv>=4:
    ph,sg,f=TB.fit_best(np.ascontiguousarray(psi[:,cols]),base,c['y'],c['got'],c['thr'],phi0=zf['phi'][lv][cols],sigma0=float(zf['sigma'][lv]));fits.append(dict(level=lv,nll=float(f),sigma=float(sg),coefficients=ph.tolist()))
   vals=[];rvals=[]
   for rep in range(20):
    phi,sig=prior_draw('uninformed','guess',lv,np.random.default_rng(SEED+977*rep),ng,zf['phi'][lv],float(zf['sigma'][lv]))
    rvals.append(TB.risk(phi[cols],sig,psi[:,cols],base,c['y'],c['got'],c['thr']))
    gp=S.rssi(lv,phi,gd,gi,gn,gdep,ng,sig*shadow);fp=gp.reshape(-1,ng).astype(np.float32);fp[fp<S.SENS[7]]=L.FLOOR;live=(fp>L.FLOOR).any(1)
    pred=L.knn_localize(fp[live],np.c_[rx,ry][live],qfp);vals.append(float(np.median(L.positioning_errors(pred,qxy))))
   loc.append(vals);risk.append(rvals)
  row=dict(resolution_m=res,n_buildings=len(polys),snapshot='2026-08-01',crossing_mae_vector=float(np.mean(np.abs(nc[sample]-exact_nc))),crossing_bias_vector=float(np.mean(nc[sample]-exact_nc)),depth_mae_vector_m=float(np.mean(np.abs(dep[sample]-exact_dep))),depth_bias_vector_m=float(np.mean(dep[sample]-exact_dep)),loc_median_mean=np.mean(loc,1).tolist(),loc_median_reps=loc,risk_mean=np.mean(risk,1).tolist(),risk_reps=risk,link_argmin=int(np.argmin(np.mean(risk,1))),loc_argmin=int(np.argmin(np.mean(loc,1))),joint_fits=fits)
  rows.append(row);(ROOT/'results/exp12_geometry.json').write_text(json.dumps(dict(rows=rows,n_exact_links=len(sample),n_rep=20,n_query=len(q),runtime_s=time.time()-t,notes='All resolutions share a pinned snapshot; submission snapshot unknown. Exact reference counts connected occupied runs.'),indent=2))
  print('resolution',res,'done',time.time()-t,flush=True)
if __name__=='__main__':main()
