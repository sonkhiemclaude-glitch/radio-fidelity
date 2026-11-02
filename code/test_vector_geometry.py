"""Independent vector-reference checks: touching/overlapping footprints and holes."""
import numpy as np
from shapely.geometry import box,Polygon
from shapely.strtree import STRtree
from exp16_vector_geometry import exact_batch

def evaluate(polys):
 p=np.array(polys,dtype=object)
 return exact_batch(np.array([-1.,10.,-1.]),np.array([0.,0.,2.]),np.array([10.,-1.,10.]),np.array([0.,0.,2.]),np.array([0,1,2]),p,STRtree(p))
nc,dep=evaluate([box(0,-1,2,1),box(2,-1,4,1),box(6,-1,8,1)])
assert np.array_equal(nc,[2,2,0]),(nc,dep)
assert np.allclose(dep,[6,6,0]),(nc,dep)
nc,dep=evaluate([box(0,-1,3,1),box(1,-1,4,1)])
assert np.array_equal(nc,[1,1,0]) and np.allclose(dep,[4,4,0]),(nc,dep)
poly=Polygon([(0,-1),(8,-1),(8,1),(0,1)],holes=[[(2,-.5),(6,-.5),(6,.5),(2,.5)]])
nc,dep=evaluate([poly])
assert np.array_equal(nc,[2,2,0]) and np.allclose(dep,[4,4,0]),(nc,dep)
print('Vector checks passed: touching polygons, overlaps, holes, reverse rays, empty rays')
