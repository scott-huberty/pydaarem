"""Download and load example data sets."""

import numpy as np
import pooch
import rdata

from ._data import BASE_URL, DATA_DIR_ENV, REGISTRY

_FETCHER = pooch.create(
    path=pooch.os_cache("pydaarem"),
    base_url=BASE_URL,
    registry=REGISTRY,
    env=DATA_DIR_ENV,
)


def fetch_mixdata():
    """Download and load the mixture proportions example data.

    The data comes from the ``mixem`` analysis in the stephenslab/daarem
    repository. The first call downloads the file and verifies its SHA-256
    hash. Later calls use the local copy.

    Returns
    -------
    L : ndarray, shape (n_samples, n_components)
        Conditional likelihood matrix. Row ``i`` contains the likelihood of
        sample ``i`` under each mixture component.
    x : ndarray, shape (n_components,)
        Mixture proportions of the reference solution. The upstream analysis
        computed this solution with the mixsqp package.

    Examples
    --------
    >>> from pydaarem.datasets import fetch_mixdata
    >>> L, x = fetch_mixdata()
    >>> L.shape
    (100000, 10)
    """
    data = rdata.read_rda(_FETCHER.fetch("mixdata.RData"))
    return np.asarray(data["L"], dtype=float), np.asarray(data["x"], dtype=float)
