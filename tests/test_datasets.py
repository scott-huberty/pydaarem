import numpy as np
from pydaarem.datasets import fetch_mixdata


def test_fetch_mixdata():
    L, x = fetch_mixdata()

    assert L.shape == (100_000, 10)
    assert x.shape == (10,)
    assert np.all(L >= 0)
    assert np.all(x >= 0)
    np.testing.assert_allclose(x.sum(), 1.0, atol=1e-8)
