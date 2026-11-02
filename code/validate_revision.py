"""Fail closed on incomplete/mixed-protocol release results and prediction files."""
import itertools,json,hashlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
def load(name):return json.loads((ROOT/'results'/name).read_text())
checks={}
a=load('exp11_cluster_admission.json')
assert len(a['rows'])==305 and sum(r['B'] for r in a['rows'])==610000
assert a['n_messages']==10000 and a['n_gateways']==31
for r in a['rows']:
 assert r['B']==2000 and 0<=r['harmful_count']<=r['B']
 assert np.isclose(r['harmful_rate'],r['harmful_count']/r['B'])
checks['link_campaigns']=610000
m=load('exp13_models.json');keys=[(r['arm'],r['mode'],r['level'],r['seed'],r['model']) for r in m['rows']]
expected=set(itertools.product(['free','equal'],['guess','randomised','oracle'],range(6),range(3),['knn','mlp','transformer']))
assert len(keys)==len(set(keys))==324 and set(keys)==expected
assert {(r['seed'],r['model']) for r in m['real_rows']}==set(itertools.product(range(3),['knn','mlp','transformer']))
assert len(m['real_rows'])==9
cap=m['protocol']['equal_reference_cap'];assert {r['n_train'] for r in m['rows'] if r['arm']=='equal'}=={4*cap}
q=None
for r in m['rows']:
 z=np.load(ROOT/f"data/pred_{r['arm']}_{r['mode']}_F{r['level']}_s{r['seed']}_{r['model']}.npz")
 if q is None:q=z['query']
 assert np.array_equal(q,z['query']) and len(q)==12000
 assert np.all(np.isfinite(z['prediction'])) and np.all(z['error']>=0)
 assert np.isclose(np.median(z['error']),r['median_m'],rtol=0,atol=1e-8)
 assert np.isclose(z['error'].mean(),r['mean_m'],rtol=0,atol=1e-8)
for r in m['real_rows']:
 z=np.load(ROOT/f"data/pred_real_s{r['seed']}_{r['model']}.npz")
 assert np.array_equal(q,z['query']) and np.isclose(np.median(z['error']),r['median_m'])
checks['synthetic_model_rows']=324;checks['real_model_rows']=9;checks['equal_arm_n_train']=4*cap
checks['query_sha256']=hashlib.sha256(q.tobytes()).hexdigest()
from exp18_spatial_bootstrap import comparisons
mc=load('exp17_map_localizer.json')
assert len(mc['rows'])==72 and len(mc['real_rows'])==6
assert {(r['mode'],r['level'],r['seed'],r['map_input']) for r in mc['rows']}==set(itertools.product(['guess','oracle'],range(6),range(3),['map','no_map']))
assert {(r['seed'],r['map_input']) for r in mc['real_rows']}==set(itertools.product(range(3),['map','no_map']))
assert mc['protocol']['epochs']==40 and mc['protocol']['query_sha256']==checks['query_sha256']
fold=np.load(ROOT/'data/pred_real_s0_knn.npz')['fold']
for group in ['rows','real_rows']:
 for r in mc[group]:
  name=(f"pred_mapcnn_{r['map_input']}_{r['mode']}_F{r['level']}_s{r['seed']}.npz" if group=='rows' else f"pred_mapcnn_{r['map_input']}_real_s{r['seed']}.npz")
  z=np.load(ROOT/'data'/name)
  assert np.array_equal(q,z['query']) and np.array_equal(fold,z['fold'])
  assert np.all(np.isfinite(z['prediction'])) and np.all(np.isfinite(z['error'])) and np.all(z['error']>=0)
  assert np.isclose(np.median(z['error']),r['median_m'],rtol=0,atol=1e-8)
  assert np.isclose(np.mean(z['error']),r['mean_m'],rtol=0,atol=1e-8)
  hist=[r] if group=='rows' else r['fold_training']
  if group=='rows':assert r['n_train']==4*cap
  else:assert len(hist)==5
  for h in hist:
   assert len(h['train_cross_entropy'])==40 and np.all(np.isfinite(h['train_cross_entropy']))
   assert (h['initial_map_gradient_l1']>0) if r['map_input']=='map' else (h['initial_map_gradient_l1']==0)
checks['map_synthetic_rows']=72;checks['map_real_rows']=6
boot=load('exp18_spatial_bootstrap.json');draws=np.load(ROOT/'data/exp18_spatial_bootstrap_replicates.npz')
assert boot['protocol']['B']==2000 and boot['protocol']['query_sha256']==checks['query_sha256']
assert len(boot['rows'])==74
assert {(r['comparison'],r['block_m']) for r in boot['rows']}==set(itertools.product([r['comparison'] for r in comparisons(True)],[500,1000]))
for r in boot['rows']:
 delta=np.mean([np.median(np.load(ROOT/'data'/p)['error']) for p in r['a']])-np.mean([np.median(np.load(ROOT/'data'/p)['error']) for p in r['b']])
 assert np.isclose(delta,r['difference_m'],rtol=0,atol=1e-8)
 vals=draws[f"{r['block_m']}_{r['comparison']}"]
 assert len(vals)==2000 and np.all(np.isfinite(vals))
 assert np.allclose(np.percentile(vals,[2.5,97.5]),r['ci95_m'],rtol=0,atol=1e-8)
checks['paired_spatial_interval_rows']=74
d=load('exp14_downstream_admission.json');assert len(d['rows'])==216 and sum(r['B'] for r in d['rows'])==432000
for r in d['rows']:
 assert r['B']==2000 and sum(r['selected_counts'])==2000
 assert np.isclose(r['harmful_rate'],r['harmful_count']/r['B'])
 assert r['global_argmin']==int(np.argmin(r['population_mean_loss_m']))
checks['downstream_campaigns']=432000
g=load('exp12_geometry.json');assert {r['resolution_m'] for r in g['rows']}=={1.,2.,5.}
v=load('exp16_vector_geometry.json');assert v['n_candidate_links']==310000 and v['n_grid_links']==349742 and v['n_rep']==20
ref=np.load(ROOT/'data/geometry_vector_reference.npz');vc=np.load(ROOT/'data/links_cand_vector.npz')
assert np.array_equal(vc['n_cross'][ref['indices']],ref['n_cross'])
assert np.allclose(vc['depth'][ref['indices']],ref['depth'],rtol=0,atol=1e-6)
checks['vector_scalar_agreement']=len(ref['indices'])
i=load('exp15_indoor_holdout.json');assert len(i['rows'])==108
assert all(r['all_within_tolerance'] for r in load('raw_validation.json'))
checks['indoor_rows']=108;checks['raw_radio_validation']=True
checks['status']='passed';(ROOT/'results/release_validation.json').write_text(json.dumps(checks,indent=2));print(json.dumps(checks,indent=2))
