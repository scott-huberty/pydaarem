import functools
import inspect
import warnings

import numpy as np
import pytest
from numpy.testing import assert_allclose
from pydaarem import daarem
from pydaarem.daarem import _CONTROL_DEFAULTS, _damping_find
from rpy2 import robjects
from rpy2.rinterface_lib.embedded import RRuntimeError
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
        fail_on_call <- function(f, fail_at, mode) {
          force(f)
          n <- 0
          function(...) {
            n <<- n + 1
            if (n == fail_at) {
              if (mode == "error") stop("injected failure")
              return(NaN)
            }
            f(...)
          }
        }
        run_daarem <- function(par, X, y, control, use_objfn, fail_fn, fail_at,
                               fail_mode) {
          fixptfn <- ProbitUpdate
          objfn <- ProbitLogLik
          if (fail_fn == "fixptfn") fixptfn <- fail_on_call(fixptfn, fail_at, fail_mode)
          if (fail_fn == "objfn") objfn <- fail_on_call(objfn, fail_at, fail_mode)
          suppressWarnings(
            if (use_objfn) {
              daarem(par, fixptfn, objfn, X=X, y=y, control=control)
            } else {
              daarem(par, fixptfn, X=X, y=y, control=control)
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


def run_r(data, control, use_objfn, fail=("", 0, "")):
    """Run R's daarem on probit data, returning a dict shaped like pydaarem's.

    ``fail`` is ``(function name, call number, "error" or "nan")``. On that
    call, the named function stops with an error or returns NaN.
    """
    load_r()
    r_control = robjects.r["list"](**control)
    beta_init = robjects.FloatVector(np.zeros(data["X"].shape[1]))
    res = robjects.globalenv["run_daarem"](
        beta_init, data["X_r"], data["y_r"], r_control, use_objfn, *fail
    )
    return {name: _r_value(name, res.rx2(name)) for name in res.names}


def fail_on_call(f, fail_at, mode):
    """Make ``f`` raise an error or return NaN on call number ``fail_at``."""
    calls = 0

    def wrapped(*args):
        nonlocal calls
        calls += 1
        if calls == fail_at:
            if mode == "error":
                raise RuntimeError("injected failure")
            return np.nan
        return f(*args)

    return wrapped


def run_py(data, control, use_objfn, fail=("", 0, "")):
    """Run pydaarem on probit data, also reporting whether it warned."""
    fail_fn, fail_at, fail_mode = fail
    fixptfn, objfn = probit_update, probit_loglik
    if fail_fn == "fixptfn":
        fixptfn = fail_on_call(fixptfn, fail_at, fail_mode)
    if fail_fn == "objfn":
        objfn = fail_on_call(objfn, fail_at, fail_mode)
    beta_init = np.zeros(data["X"].shape[1])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        got = daarem(
            beta_init,
            fixptfn,
            objfn if use_objfn else None,
            data["X"],
            data["y"],
            control=control,
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
# these cases is 2e-7, so rtol has a margin of 50.
LOOSE = {"rtol": 1e-5, "atol": 1e-10}
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
    # Covers nlag above 3. As the iterates converge, the lag columns align and
    # the Anderson least-squares problem becomes ill-conditioned (condition
    # number 1e8 for order 10 on BIG), so rounding decides discrete steps.
    # In 30 fixed-point evaluations the condition number stays below 5e4 and
    # the history is full (8 columns) for 3 cycles.
    (
        "order8",
        {"n": 200, "npars": 16, "seed": 2},
        {"order": 8, "maxiter": 30, "intermed": True},
        BOTH,
        LOOSE,
    ),
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


# A function in a try block on that call: the objfn path calls fixptfn once per
# iteration in a try block. Calls 1 and 2 of objfn and of the no-objfn fixptfn
# are the start values, and call 3 is the first proposal, in a try block.
# (id, objfn path, failure)
FAILURES = [
    ("fixptfn-error", True, ("fixptfn", 10, "error")),
    ("objfn-error", True, ("objfn", 3, "error")),
    ("objfn-nan", True, ("objfn", 3, "nan")),
    ("fixptfn-error-noobjfn", False, ("fixptfn", 3, "error")),
]


@pytest.mark.parametrize(
    "use_objfn, fail", [f[1:] for f in FAILURES], ids=[f[0] for f in FAILURES]
)
def test_daarem_failure_matches_r(use_objfn, fail):
    data = probit_data()
    control = {"maxiter": 80, "intermed": True}
    ref = run_r(data, control, use_objfn, fail)
    got, warned = run_py(data, control, use_objfn, fail)

    assert_daarem_equal(got, ref, **TIGHT)
    assert warned == (not ref["convergence"])


def test_invalid_convtype_raises_like_r():
    data = probit_data()
    control = {"convtype": "other"}
    with pytest.raises(RRuntimeError):
        run_r(data, control, True)
    with pytest.raises(ValueError, match="convtype"):
        run_py(data, control, True)


UY_SQ = [1.0, 0.5, 0.2]
DVEC = [3.0, 1.0, 0.1]
# (id, uy_sq, dvec, lambda.start, r.start)
DAMPING = [
    ("typical", UY_SQ, DVEC, 1e5, 0.0),
    ("restart", UY_SQ, DVEC, 2.0, 0.5),
    ("nan-start", UY_SQ, DVEC, np.nan, np.nan),
    ("zero-singular-value", UY_SQ, [3.0, 1.0, 0.0], 1e5, 0.0),
    ("zero-target", [0.0, 0.0, 0.0], DVEC, 1e5, 0.0),
]


@pytest.mark.parametrize(
    "uy_sq, dvec, lambda_start, r_start",
    [d[1:] for d in DAMPING],
    ids=[d[0] for d in DAMPING],
)
@pytest.mark.parametrize("sk", [0, 10])
def test_damping_find_matches_r(uy_sq, dvec, lambda_start, r_start, sk):
    load_r()
    uy_sq, dvec = np.array(uy_sq), np.array(dvec)
    ftf = float(np.sqrt(np.sum(uy_sq * dvec**2)))
    args = (uy_sq, dvec, 1.2, 25, sk, ftf)

    ref = robjects.r("daarem:::DampingFind")(
        *(robjects.FloatVector(a) for a in args[:2]),
        *args[2:],
        lambda_start=lambda_start,
        r_start=r_start,
    )
    got = _damping_find(*args, lambda_start=lambda_start, r_start=r_start)

    assert_allclose(got["lambda"], ref.rx2("lambda")[0], rtol=1e-12)
    assert_allclose(got["rr"], ref.rx2("rr")[0], rtol=1e-12, atol=1e-300)


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
