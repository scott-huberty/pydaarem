import functools
import inspect
import warnings

import numpy as np
import pytest
from numpy.testing import assert_allclose
from pydaarem import daarem
from pydaarem.daarem import _CONTROL_DEFAULTS
from rpy2 import robjects
from rpy2.robjects import numpy2ri
from scipy import special


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


@functools.cache
def load_r():
    """Attach the R daarem package and define the R test helpers once."""
    robjects.r(
        """
        library(daarem)
        run_daarem <- function(par, X, y, control, use_objfn) {
          suppressWarnings(
            if (use_objfn) {
              daarem(par, ProbitUpdate, ProbitLogLik, X=X, y=y, control=control)
            } else {
              daarem(par, ProbitUpdate, X=X, y=y, control=control)
            })
        }
        find_control_defaults <- function() {
          for (e in as.list(body(daarem))[-1]) {
            if (is.call(e) && identical(e[[1]], as.name("<-")) &&
                identical(e[[2]], as.name("control.default"))) {
              return(eval(e[[3]], baseenv()))
            }
          }
          stop("control.default not found in body(daarem)")
        }
        """
    )


@functools.cache
def probit_data(n=200, npars=5, seed=1):
    """Simulate probit data in R, returning both the R and NumPy versions."""
    load_r()
    out = robjects.r(
        f"""
        set.seed({seed})
        true.beta <- .5 * rt({npars}, df=2) + 2
        XX <- matrix(rnorm({n} * {npars}), nrow={n}, ncol={npars})
        yy <- ProbitSimulate(true.beta, XX)
        list(X=XX, y=yy)
        """
    )
    with (robjects.default_converter + numpy2ri.converter).context():
        X = np.asarray(out.rx2("X"), dtype=float)
        y = np.asarray(out.rx2("y"), dtype=float)
    return {"X_r": out.rx2("X"), "y_r": out.rx2("y"), "X": X, "y": y}


def _r_value(name, value):
    if value is robjects.NULL:
        return None
    if name in ("fpevals", "objfevals"):
        return int(value[0])
    if name == "convergence":
        return bool(value[0])
    if name == "value.objfn":
        return float(value[0])
    with (robjects.default_converter + numpy2ri.converter).context():
        return np.asarray(value, dtype=float)


def run_r(data, control, use_objfn):
    """Run R's daarem on probit data, returning a dict shaped like pydaarem's."""
    load_r()
    r_control = robjects.r["list"](**control)
    beta_init = robjects.FloatVector(np.zeros(data["X"].shape[1]))
    res = robjects.globalenv["run_daarem"](
        beta_init, data["X_r"], data["y_r"], r_control, use_objfn
    )
    return {name: _r_value(name, res.rx2(name)) for name in res.names}


def run_py(data, control, use_objfn):
    """Run pydaarem on probit data, also reporting whether it warned."""
    beta_init = np.zeros(data["X"].shape[1])
    objfn = probit_loglik if use_objfn else None
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        got = daarem(
            beta_init, probit_update, objfn, data["X"], data["y"], control=control
        )
    warned = any(
        issubclass(w.category, RuntimeWarning) and "did not converge" in str(w.message)
        for w in caught
    )
    return got, warned


def assert_daarem_equal(got, ref, rtol=1e-9, atol=1e-10):
    # R and NumPy use different BLAS/LAPACK routines, so floats agree only to
    # rounding error. Counts and flags must agree exactly, and are checked
    # first so that a diverged trajectory is reported as such.
    assert list(got) == list(ref)
    for name, expected in ref.items():
        if expected is None:
            assert got[name] is None, name
        elif isinstance(expected, (bool, int)):
            assert got[name] == expected, name
    for name, expected in ref.items():
        if isinstance(expected, (float, np.ndarray)):
            assert_allclose(got[name], expected, rtol=rtol, atol=atol, err_msg=name)


BOTH = (True, False)
TIGHT = {"rtol": 1e-9, "atol": 1e-10}
# For runs that amplify rounding error. Counts and flags still match exactly.
# Across OpenBLAS kernels (OPENBLAS_CORETYPE) the largest relative difference in
# these cases is 1.5e-6, so rtol has a margin of about 6. The final residual
# norms are rounding noise near tol, so atol does not depend on the BLAS build.
LOOSE = {"rtol": 1e-5, "atol": 1e-7}
BIG = {"n": 2000, "npars": 20, "seed": 2}
# (id, probit_data kwargs, control, objfn paths to run, float tolerances)
CASES = [
    ("maxiter80", {}, {"maxiter": 80}, BOTH, TIGHT),
    ("defaults", {}, {}, BOTH, TIGHT),
    ("convtype-objfn", {}, {"convtype": "objfn", "tol": 1e-10}, (True,), TIGHT),
    ("intermed", {}, {"intermed": True}, BOTH, TIGHT),
    ("intermed-noconv", {}, {"intermed": True, "maxiter": 5}, BOTH, TIGHT),
    # A loose mon.tol accepts objective decreases, so the cycle-level check
    # decreases shrink.count (first changing the path at iteration 17); a huge
    # cycl.mon.tol suppresses that. Non-monotone runs amplify rounding error,
    # so stop at 22 iterations, before R and Python drift apart.
    (
        "cycl-decrease",
        BIG,
        {"mon.tol": 100.0, "maxiter": 22, "intermed": True},
        (True,),
        LOOSE,
    ),
    (
        "cycl-mon-tol",
        BIG,
        {"mon.tol": 100.0, "cycl.mon.tol": 1e6, "maxiter": 22, "intermed": True},
        (True,),
        LOOSE,
    ),
    ("noconv", {}, {"maxiter": 5}, BOTH, TIGHT),
    # With 10 nearly collinear lag columns the ridge solve amplifies rounding
    # error, so the iterates agree to ~1e-7 relative while all counts match.
    # The no-objfn path checks tol only at the end of each cycle of 10. With
    # tol=1e-8 the last residual is 9e-9, so the BLAS kernel decides whether
    # one more cycle runs. With tol=1e-7 the last residual is 11 times smaller
    # than tol and the residual one cycle earlier is 3 times larger.
    ("order10", BIG, {"order": 10, "tol": 1e-7}, BOTH, LOOSE),
    ("damping", {}, {"kappa": 10, "alpha": 1.5}, BOTH, TIGHT),
    ("resid-tol", {}, {"resid.tol": 0.3}, (False,), TIGHT),
]
PARAMS = [
    pytest.param(
        data_kwargs,
        control,
        use_objfn,
        tols,
        id=f"{case_id}-{'objfn' if use_objfn else 'noobjfn'}",
    )
    for case_id, data_kwargs, control, paths, tols in CASES
    for use_objfn in paths
]


@pytest.mark.parametrize("data_kwargs, control, use_objfn, tols", PARAMS)
def test_daarem_matches_r(data_kwargs, control, use_objfn, tols):
    data = probit_data(**data_kwargs)
    ref = run_r(data, control, use_objfn)
    got, warned = run_py(data, control, use_objfn)

    assert_daarem_equal(got, ref, **tols)
    assert warned == (not ref["convergence"])


def test_signature_matches_r():
    load_r()
    r_formals = list(robjects.r("names(formals(daarem))"))

    params = list(inspect.signature(daarem).parameters.values())
    py_formals = []
    for p in params:
        if p.kind is p.VAR_POSITIONAL:
            py_formals.append("...")
        elif p.kind is not p.VAR_KEYWORD:
            py_formals.append(p.name)
    assert py_formals == r_formals

    # R's ``...`` maps to ``*args, **kwargs``: arguments before it can be
    # positional, arguments after it can only be passed by name.
    P = inspect.Parameter
    dots = r_formals.index("...")
    assert all(p.kind is P.POSITIONAL_OR_KEYWORD for p in params[:dots])
    assert params[dots].kind is P.VAR_POSITIONAL
    assert all(p.kind is P.KEYWORD_ONLY for p in params[dots + 1 : -1])
    assert params[-1].kind is P.VAR_KEYWORD

    # ``par`` and ``fixptfn`` are required in both. ``objfn`` uses ``None``
    # where R uses ``missing()``, and ``control`` uses ``None`` for ``list()``.
    sig = inspect.signature(daarem).parameters
    assert sig["par"].default is P.empty
    assert sig["fixptfn"].default is P.empty
    assert sig["objfn"].default is None
    assert sig["control"].default is None


def test_control_defaults_match_r():
    load_r()
    r_defaults = robjects.globalenv["find_control_defaults"]()
    r_dict = {name: r_defaults.rx2(name)[0] for name in r_defaults.names}

    assert list(_CONTROL_DEFAULTS) == list(r_dict)
    for name, value in r_dict.items():
        assert _CONTROL_DEFAULTS[name] == value, name


@pytest.mark.parametrize("use_objfn", BOTH, ids=["objfn", "noobjfn"])
def test_return_keys_match_r(use_objfn):
    data = probit_data()
    ref = run_r(data, {"maxiter": 5}, use_objfn)
    got, _ = run_py(data, {"maxiter": 5}, use_objfn)
    assert list(got) == list(ref)


def test_unknown_control_raises():
    with pytest.raises(ValueError, match="unknown names in control"):
        daarem(np.zeros(2), lambda x: x, control={"max_iter": 10})
