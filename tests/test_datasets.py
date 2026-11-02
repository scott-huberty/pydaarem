import importlib
import sys

import numpy as np
import pytest
from pydaarem.datasets import fetch_mixdata


def test_fetch_mixdata():
    L, x = fetch_mixdata()

    assert L.shape == (100_000, 10)
    assert x.shape == (10,)
    assert np.all(L >= 0)
    assert np.all(x >= 0)
    np.testing.assert_allclose(x.sum(), 1.0, atol=1e-8)


@pytest.mark.parametrize("missing", ["pooch", "rdata"])
def test_import_without_extra_gives_install_hint(missing, monkeypatch):
    monkeypatch.setitem(sys.modules, missing, None)
    monkeypatch.delitem(sys.modules, "pydaarem.datasets")

    with pytest.raises(ImportError, match=r"pip install 'pydaarem\[datasets\]'") as e:
        importlib.import_module("pydaarem.datasets")
    assert repr(missing) in str(e.value)
