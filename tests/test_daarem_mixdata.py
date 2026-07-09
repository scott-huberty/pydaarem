import numpy as np
from numpy.testing import assert_allclose
from rpy2 import robjects
from rpy2.robjects import numpy2ri

from pydaarem import daarem


DATA_PATH = "/Users/scotterik/amica_test_data/daarem/mixdata.RData"
R_ROOT = "/Users/scotterik/devel/projects/amica-python"


def load_mixdata():
    with (robjects.default_converter + numpy2ri.converter).context():
        env = robjects.Environment()
        robjects.r["load"](DATA_PATH, envir=env)
        return np.asarray(env["L"], dtype=float)


def project_iterate(x):
    x = np.maximum(np.asarray(x, dtype=float), 0.0)
    return x / np.sum(x)


def mixem_update(L, x, e):
    p = L * x + e[:, None]
    p = p / np.sum(p, axis=1, keepdims=True)
    return np.mean(p, axis=0)


def mixobjective(L, x, e):
    y = L @ x + e
    if np.all(y > 0):
        return float(np.sum(np.log(y)))
    return -np.inf


def mixdaarem_update(x, L, e):
    return mixem_update(L, project_iterate(x), e)


def mixdaarem_objective(x, L, e):
    return mixobjective(L, project_iterate(x), e)


def r_mixdaarem_reference(numiter=200, order=10):
    ref = robjects.r(
        f"""
        source("{R_ROOT}/optimizers/daarem/R/DampingFind.R")
        source("{R_ROOT}/optimizers/daarem/R/daarem_base_objfn.R")
        source("{R_ROOT}/optimizers/daarem/R/daarem_base_noobjfn.R")
        source("{R_ROOT}/optimizers/daarem/R/daarem.R")
        source("{R_ROOT}/optimizers/examples/daarem/code/misc.R")
        source("{R_ROOT}/optimizers/examples/daarem/code/mixem.R")
        load("{DATA_PATH}")
        e <- 1e-15 * apply(L, 1, max)
        x0 <- rep(1 / ncol(L), ncol(L))
        out <- suppressWarnings(
          daarem(x0, mixdaarem.update, mixdaarem.objective, L, e,
                 control = list(maxiter = {numiter}, order = {order}, tol = 0,
                                mon.tol = 0.05, kappa = 20, alpha = 1.2)))
        list(par = project.iterate(out$par),
             value = out$objfn.track[-1],
             fpevals = out$fpevals,
             objfevals = out$objfevals,
             convergence = out$convergence)
        """
    )
    with (robjects.default_converter + numpy2ri.converter).context():
        return {
            "par": np.asarray(ref.rx2("par"), dtype=float),
            "value": np.asarray(ref.rx2("value"), dtype=float),
            "fpevals": int(ref.rx2("fpevals")[0]),
            "objfevals": int(ref.rx2("objfevals")[0]),
            "convergence": bool(ref.rx2("convergence")[0]),
        }


def test_daarem_matches_r_mixdata_example():
    L = load_mixdata()
    e = 1e-15 * np.max(L, axis=1)
    x0 = np.repeat(1.0 / L.shape[1], L.shape[1])

    got = daarem(
        x0,
        mixdaarem_update,
        mixdaarem_objective,
        L,
        e,
        control={
            "maxiter": 200,
            "order": 10,
            "tol": 0,
            "mon.tol": 0.05,
            "kappa": 20,
            "alpha": 1.2,
        },
    )
    ref = r_mixdaarem_reference(numiter=200, order=10)

    assert got["convergence"] == ref["convergence"]
    assert got["fpevals"] == ref["fpevals"]
    assert got["objfevals"] == ref["objfevals"]
    assert_allclose(project_iterate(got["par"]), ref["par"], rtol=0, atol=1e-10)
    assert_allclose(got["objfn.track"][1:], ref["value"], rtol=0, atol=1e-8)
