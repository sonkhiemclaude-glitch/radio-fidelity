"""Check integer-weight medians against explicit whole-block replication."""
import numpy as np
from exp18_spatial_bootstrap import block_medians
rng = np.random.default_rng(97)
for n in [5, 16, 101]:
    inverse = np.arange(n) % 4
    error = rng.integers(0, 9, n).astype(float)
    weights = rng.multinomial(4, np.full(4, .25), size=100)
    actual = block_medians(error, inverse, weights, batch=7)
    expected = np.array([np.median(np.repeat(error, w[inverse])) for w in weights])
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(actual-block_medians(error,inverse,weights),np.zeros(100))
    shifted = block_medians(error+123,inverse,weights)
    np.testing.assert_array_equal(shifted-actual,np.full(100,123.))
print('Spatial bootstrap: explicit replication, ties, even/odd sizes and paired cancellation passed')
