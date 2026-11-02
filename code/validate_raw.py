"""Check independently reprocessed public corpora against submission caches.

Report exact equality separately from a declared absolute floating-point
coordinate tolerance of 1e-8 metre (relative tolerance zero).
"""
import json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
rows=[]
for name in ['lorawan.npz','uji.npz']:
 old=np.load(ROOT/'data'/name,allow_pickle=True);new=np.load(ROOT/'data/from_raw'/name,allow_pickle=True)
 fields={}
 for k in old.files:
  a,b=old[k],new[k];numeric=a.dtype.kind in 'fci'
  exact=bool(np.array_equal(a,b,equal_nan=True)) if numeric else bool(np.array_equal(a,b))
  close=bool(np.allclose(a,b,rtol=0,atol=1e-8,equal_nan=True)) if numeric else exact
  fields[k]=dict(exact=exact,within_tolerance=close)
  if numeric:fields[k]['max_abs_difference']=float(np.nanmax(np.abs(a-b))) if a.size else 0.0
 rows.append(dict(file=name,all_exact=all(v['exact'] for v in fields.values()),all_within_tolerance=all(v['within_tolerance'] for v in fields.values()),absolute_tolerance=1e-8,relative_tolerance=0,fields=fields))
(ROOT/'results/raw_validation.json').write_text(json.dumps(rows,indent=2));print(json.dumps(rows,indent=2))
if not all(row['all_within_tolerance'] for row in rows):raise SystemExit('Raw validation failed')
