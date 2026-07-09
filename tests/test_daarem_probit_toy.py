import numpy as np
from numpy.testing import assert_allclose
from rpy2 import robjects
from rpy2.robjects import numpy2ri
from scipy import special

from pydaarem import daarem


R_ROOT = "/Users/scotterik/devel/projects/amica-python"


def inv_mills_ratio(x, lt):
    x = np.asarray(x, dtype=float)
    log_pdf = -0.5 * x * x - 0.5 * np.log(2.0 * np.pi)
    log_prob = special.log_ndtr(-x) if lt else special.log_ndtr(x)
    return np.exp(log_pdf - log_prob)


def probit_update(beta_hat, X, y):
    linear_pred = X @ beta_hat
    z = (
        linear_pred
        + y * inv_mills_ratio(linear_pred, lt=False)
        - (1.0 - y) * inv_mills_ratio(linear_pred, lt=True)
    )
    return np.linalg.solve(X.T @ X, X.T @ z)


def probit_loglik(beta_hat, X, y):
    linear_pred = X @ beta_hat
    fvals1 = special.log_ndtr(linear_pred)
    fvals2 = special.log_ndtr(-linear_pred)
    return float(np.sum(y * fvals1) + np.sum((1.0 - y) * fvals2))


def r_probit_toy_reference(n=200, npars=5, maxiter=80):
    out = robjects.r(
        f"""
        source("{R_ROOT}/optimizers/daarem/R/DampingFind.R")
        source("{R_ROOT}/optimizers/daarem/R/InvMillsRatio.R")
        source("{R_ROOT}/optimizers/daarem/R/ProbitSimulate.R")
        source("{R_ROOT}/optimizers/daarem/R/ProbitUpdate.R")
        source("{R_ROOT}/optimizers/daarem/R/ProbitLogLik.R")
        source("{R_ROOT}/optimizers/daarem/R/daarem_base_objfn.R")
        source("{R_ROOT}/optimizers/daarem/R/daarem_base_noobjfn.R")
        source("{R_ROOT}/optimizers/daarem/R/daarem.R")
        set.seed(1)
        n <- {n}
        npars <- {npars}
        true.beta <- .5 * rt(npars, df=2) + 2
        XX <- matrix(rnorm(n * npars), nrow=n, ncol=npars)
        yy <- ProbitSimulate(true.beta, XX)
        beta.init <- rep(0.0, npars)
        fit <- suppressWarnings(
          daarem(par=beta.init, fixptfn=ProbitUpdate, objfn=ProbitLogLik,
                 X=XX, y=yy, control=list(maxiter={maxiter})))
        list(
          X=XX,
          y=yy,
          par=fit$par,
          value.objfn=fit$value.objfn,
          fpevals=fit$fpevals,
          objfevals=fit$objfevals,
          convergence=fit$convergence,
          objfn.track=fit$objfn.track
        )
        """
    )
    with (robjects.default_converter + numpy2ri.converter).context():
        return {
            "X": np.asarray(out.rx2("X"), dtype=float),
            "y": np.asarray(out.rx2("y"), dtype=float),
            "par": np.asarray(out.rx2("par"), dtype=float),
            "value.objfn": float(out.rx2("value.objfn")[0]),
            "fpevals": int(out.rx2("fpevals")[0]),
            "objfevals": int(out.rx2("objfevals")[0]),
            "convergence": bool(out.rx2("convergence")[0]),
            "objfn.track": np.asarray(out.rx2("objfn.track"), dtype=float),
        }


def test_daarem_matches_r_probit_toy_example():
    ref = r_probit_toy_reference()
    beta_init = np.zeros(ref["X"].shape[1])

    got = daarem(
        beta_init,
        probit_update,
        probit_loglik,
        ref["X"],
        ref["y"],
        control={"maxiter": 80},
    )

    assert got["convergence"] == ref["convergence"]
    assert got["fpevals"] == ref["fpevals"]
    assert got["objfevals"] == ref["objfevals"]
    assert_allclose(got["par"], ref["par"], rtol=0, atol=1e-10)
    assert_allclose(got["value.objfn"], ref["value.objfn"], rtol=0, atol=1e-10)
    assert_allclose(got["objfn.track"], ref["objfn.track"], rtol=0, atol=1e-10)
