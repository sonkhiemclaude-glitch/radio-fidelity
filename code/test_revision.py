"""Regression checks for corrected ranges and cluster-level certification."""
import numpy as np
import theory as T
from exp11_cluster_admission import selected
r=np.random.default_rng(31)
# Reproduce the analytic range correction exactly, including its constant.
x=np.array([-15.,15.,-15.,15.]);alpha=.05
want=x.mean()+np.sqrt(2*x.var(ddof=1)*np.log(2/alpha)/len(x))+7*30*np.log(2/alpha)/(3*(len(x)-1))
assert np.isclose(T.eb_bound(x,alpha,30),want)
try:T.eb_bound(x,alpha,15)
except ValueError:pass
else:raise AssertionError('Underestimated observed range accepted')
# Cluster-averaged tests must not gain 31-fold pseudo-replication from gateways.
units=r.uniform(0,15,(1000,128,6));a,_=selected(units,15)
for i in range(20):
 assert a[i]==T.fixed_sequence_test([units[i,:,j+1]-units[i,:,j] for j in range(5)],rng_range=30)
# A spatially constant harmful rung remains harmful under PPS cluster draws.
for i in range(100):
 v=r.uniform(0,1,(1,256,3));v[:,:,1]=v[:,:,0]-.5;v[:,:,2]=v[:,:,1]+.1
 a,_=selected(v,15);assert a[0]<=1
print('Revision checks passed: range scaling, cluster sample count, fixed sequence')
