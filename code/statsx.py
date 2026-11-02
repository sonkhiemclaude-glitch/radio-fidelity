"""Self-contained paired inference; replacement for unavailable shared helper.

Original results remain archived. Newly computed intervals use percentile
bootstrap of paired observations, two-sided Wilcoxon signed-rank tests, and Holm.
"""
import numpy as np
from scipy import stats

def bootstrap_ci(x, B=10000, seed=3108):
 x=np.asarray(x,float);x=x[np.isfinite(x)]
 rng=np.random.default_rng(seed)
 v=x[rng.integers(0,len(x),(B,len(x)))].mean(1)
 return float(x.mean()),*map(float,np.percentile(v,[2.5,97.5]))

def paired_test(a,b):
 d=np.asarray(a)-np.asarray(b);d=d[np.isfinite(d)]
 if np.all(d==0):return {'p':1.0}
 return {'p':float(stats.wilcoxon(d).pvalue)}

def holm_bonferroni(p,alpha=.05):
 p=np.asarray(p,float);order=np.argsort(p);adj=np.empty(len(p))
 adj[order]=np.minimum(1,np.maximum.accumulate(p[order]*(len(p)-np.arange(len(p)))))
 return adj,adj<=alpha
