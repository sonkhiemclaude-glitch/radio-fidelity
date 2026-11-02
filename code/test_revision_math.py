"""Independent numerical checks for corrected proof distinctions and edge cases."""
import numpy as np
rng=np.random.default_rng(20261102)
# Profile lower coefficients numerically, independently of the Gram formula.
for trial in range(50):
 X=rng.normal(size=(500,3));A=rng.normal(size=(500,2))+X@rng.normal(size=(3,2));y=rng.normal(size=500)+X@rng.normal(size=3)
 b0=np.linalg.lstsq(X,y,rcond=None)[0];r=X@b0-y
 AP=A-X@np.linalg.lstsq(X,A,rcond=None)[0];G=AP.T@AP/len(y)
 oracle=np.linalg.lstsq(AP,-r,rcond=None)[0];b=rng.normal(size=2)
 lower=np.linalg.lstsq(X,y-A@b,rcond=None)[0]
 actual=np.mean((X@lower+A@b-y)**2)-np.mean(r**2)
 formula=(b-oracle)@G@(b-oracle)-oracle@G@oracle
 assert np.isclose(actual,formula,atol=1e-10)
print('PASS: profiled increments agree with independent least-squares fits')
# Two nonzero PSD matrices need not give strict improvement/penalty.
G=np.diag([1.,0.]);Sigma=np.diag([0.,2.]);assert np.trace(Sigma@G)==0
print('PASS: nonzero covariance in null features has zero penalty')
# Exact ratio against a genuinely nonquadratic convex risk and discrete priors.
for trial in range(100):
 star=rng.uniform(.02,.4);b=rng.uniform(-.3,.5,20);h=rng.uniform(.5,3)
 risk=lambda x:h*(x-star)**2+.4*(x-star)**4
 NH=h*np.mean((b-star)**2);DH=h*star**2;r0=.4*star**4;er=.4*np.mean((b-star)**4)
 rho=np.mean(risk(b))/risk(0);rhoH=NH/DH
 bound=(er+rhoH*abs(r0))/(DH-abs(r0))
 assert abs(rho-rhoH)<=bound+1e-10
print('PASS: Hessian-ratio remainder bound on convex quartic risks')
