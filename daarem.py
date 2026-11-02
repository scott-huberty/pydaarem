"""Damped Anderson acceleration with restarts and monotonicity control."""

from __future__ import annotations

import math
import warnings

import numpy as np

_CONTROL_DEFAULTS = {
    "maxiter": 2000,
    "order": 5,
    "tol": 1.0e-8,
    "mon.tol": 0.01,
    "cycl.mon.tol": 0.0,
    "kappa": 25,
    "alpha": 1.2,
    "resid.tol": 0.95,
    "convtype": "param",
    "intermed": False,
}


def daarem(par, fixptfn, objfn=None, *args, control=None, **kwargs):
    """Accelerate a fixed-point iteration with DAAREM.

    DAAREM is a damped Anderson acceleration scheme with periodic restarts
    and monotonicity control. It is intended as an off-the-shelf accelerator
    for smooth, monotone, slowly converging fixed-point iterations, including
    expectation-maximization (EM), minorization-maximization (MM), and related
    algorithms.

    Parameters
    ----------
    par : array_like, shape (n_parameters,)
        Starting values for the parameter vector.
    fixptfn : callable
        Fixed-point mapping ``G``. The callable must accept the current
        parameter vector as its first argument and return a vector with the
        same length. It defines the iteration ``x[k + 1] = G(x[k])``. For an
        EM algorithm, this function performs one E/M update.
    objfn : callable or None, optional
        Scalar merit function ``L`` that is maximized at the fixed point of
        ``G``. The callable must accept the parameter vector as its first
        argument and return a scalar. For an EM algorithm, this is typically
        the observed-data log-likelihood. Supplying ``objfn`` is optional but
        preferable because it enables monotonicity control. If ``None`` is
        passed, DAAREM uses the residual-norm acceptance rule from the R
        implementation's no-objective path; ``value.objfn``, ``objfevals``,
        and ``objfn.track`` are returned as ``None``, and ``residuals``
        contains the recorded residual norms.
    *args
        Additional positional arguments passed to both ``fixptfn`` and
        ``objfn``.
    control : dict, optional
        Algorithm controls. The accepted keys mirror the R implementation;
        no other keys are allowed.

        ``maxiter`` : int, default=2000
            Maximum number of iterations or fixed-point evaluations, following
            the R implementation's accounting for the selected algorithm path.
        ``order`` : int, default=5
            Order of the DAAREM acceleration scheme. Internally this is capped
            at ``ceil(len(par) / 2)``.
        ``tol`` : float, default=1e-8
            Convergence tolerance. With ``convtype="param"``, iteration stops
            when ``||G(x[k]) - x[k]|| < tol``. With ``convtype="objfn"``,
            iteration stops when ``abs(L(x[k + 1]) - L(x[k])) < tol``.
        ``mon.tol`` : float, default=0.01
            Nonnegative tolerance for objective monotonicity. With an
            objective function, an accelerated proposal is rejected when
            ``L(x[k + 1]) < L(x[k]) - mon.tol``.
        ``cycl.mon.tol`` : float, default=0.0
            Nonnegative tolerance for the cycle-level monotonicity check at
            restart boundaries.
        ``kappa`` : float, default=25
            Half-life parameter controlling how quickly the relative damping
            factor approaches one.
        ``alpha`` : float, default=1.2
            Parameter greater than one controlling the initial relative damping
            factor and its rate of relaxation.
        ``resid.tol`` : float, default=0.95
            Residual-change tolerance used when ``objfn`` is not supplied.
            A proposed accelerated step is accepted when its residual norm is
            at most the previous residual norm times ``1 + resid.tol**k``.
        ``convtype`` : {"param", "objfn"}, default="param"
            Convergence criterion used when ``objfn`` is supplied.
        ``intermed`` : bool, default=False
            If true, return intermediate iterates in ``p.intermed``. With an
            objective function, the first column is the objective value and the
            remaining columns are parameters.
    **kwargs
        Additional keyword arguments passed to both ``fixptfn`` and ``objfn``.

    Returns
    -------
    dict
        Dictionary with keys matching the R implementation.

        ``par`` : ndarray
            Final parameter vector.
        ``fpevals`` : int
            Number of fixed-point function evaluations or iterations, following
            the R implementation's accounting.
        ``value.objfn`` : float or None
            Objective function value at termination, or ``None`` when no
            objective function is supplied.
        ``objfevals`` : int or None
            Number of objective function evaluations, or ``None`` when no
            objective function is supplied.
        ``convergence`` : bool
            True if the algorithm converged before reaching ``maxiter``.
        ``objfn.track`` : ndarray or None
            Objective values recorded during iteration, or ``None`` when no
            objective function is supplied.
        ``residuals`` : ndarray, optional
            Residual norms recorded during iteration. Returned only when
            ``objfn`` is not supplied.
        ``p.intermed`` : ndarray or None
            Intermediate iterates when ``intermed`` is true, otherwise ``None``.

    Warns
    -----
    RuntimeWarning
        If the algorithm reaches the iteration limit without convergence.

    Notes
    -----
    The public API and control names follow the R package ``daarem``. Unknown
    control names and a ``convtype`` other than ``"param"`` or ``"objfn"``
    raise ``ValueError``. The R package also stops with an error in these
    cases.

    References
    ----------
    Henderson, N. C. and Varadhan, R. (2019). Damped Anderson acceleration
    with restarts and monotonicity control for accelerating EM and EM-like
    algorithms. *Journal of Computational and Graphical Statistics*, 28(4),
    834-846. https://doi.org/10.1080/10618600.2019.1594835

    Examples
    --------
    >>> import numpy as np
    >>> from pydaarem import daarem
    >>> def fixpt(x):
    ...     return np.cos(x)
    >>> out = daarem(np.array([1.0]), fixpt, control={"maxiter": 100})
    >>> out["par"].shape
    (1,)
    """
    control = {} if control is None else dict(control)
    unknown = [name for name in control if name not in _CONTROL_DEFAULTS]
    if unknown:
        raise ValueError(f"unknown names in control: {unknown}")
    opts = dict(_CONTROL_DEFAULTS)
    opts.update(control)

    par = np.asarray(par, dtype=float).reshape(-1)
    num_params = len(par)
    nlag = min(int(opts["order"]), int(np.ceil(num_params / 2)))

    if opts["convtype"] not in ("param", "objfn"):
        raise ValueError(f"convtype must be 'param' or 'objfn': {opts['convtype']!r}")
    check_par_resid = opts["convtype"] == "param"

    if objfn is None:
        ans = _daarem_base_noobjfn(
            par=par,
            fixptfn=fixptfn,
            maxiter=int(opts["maxiter"]),
            tol=float(opts["tol"]),
            resid_tol=float(opts["resid.tol"]),
            a1=float(opts["alpha"]),
            kappa=float(opts["kappa"]),
            nlag=nlag,
            intermed=bool(opts["intermed"]),
            args=args,
            kwargs=kwargs,
        )
    else:
        ans = _daarem_base_objfn(
            par=par,
            fixptfn=fixptfn,
            objfn=objfn,
            maxiter=int(opts["maxiter"]),
            tol=float(opts["tol"]),
            mon_tol=float(opts["mon.tol"]),
            cycl_mon_tol=float(opts["cycl.mon.tol"]),
            a1=float(opts["alpha"]),
            kappa=float(opts["kappa"]),
            nlag=nlag,
            check_par_resid=check_par_resid,
            intermed=bool(opts["intermed"]),
            args=args,
            kwargs=kwargs,
        )
    if not ans["convergence"]:
        warnings.warn("Algorithm did not converge", RuntimeWarning, stacklevel=2)
    return ans


def _call_vector(fn, x, args, kwargs):
    return np.asarray(
        fn(np.asarray(x, dtype=float), *args, **kwargs), dtype=float
    ).reshape(-1)


def _call_scalar(fn, x, args, kwargs):
    return float(fn(np.asarray(x, dtype=float), *args, **kwargs))


def _norm2(x):
    return float(np.sqrt(np.dot(x, x)))


def _with_value(value, x):
    """Put a scalar in front of a vector, as R does with ``c(value, x)``."""
    return np.concatenate([np.asarray([value], dtype=float), x])


def _drop_nan(values):
    """Remove NaN values, as R does with ``x[!is.na(x)]``."""
    return np.asarray([v for v in values if not math.isnan(v)], dtype=float)


def _damping_find(
    uy_sq,
    dvec,
    aa,
    kappa,
    sk,
    ftf,
    lambda_start=None,
    r_start=None,
    maxit=10,
):
    if lambda_start is None or math.isnan(lambda_start):
        lambda_start = 100.0
    if r_start is None or math.isnan(r_start):
        r_start = 0.0

    uy_sq = np.asarray(uy_sq, dtype=float).reshape(-1)
    dvec = np.asarray(dvec, dtype=float).reshape(-1)
    # R drops the components with a zero singular value. Set their terms to
    # zero instead, so that the array shapes do not depend on the data.
    nonzero = dvec > 0
    d_sq = dvec * dvec
    d_sq_safe = np.where(nonzero, d_sq, 1.0)

    pow_ = kappa - sk
    target = np.exp(-0.5 * np.log1p(aa**pow_))
    betahat_ls = np.where(nonzero, uy_sq / d_sq_safe, 0.0)
    betahat_ls_norm = float(np.sqrt(np.sum(betahat_ls)))
    vk = target * betahat_ls_norm

    if vk == 0:
        return {"lambda": float(lambda_start), "rr": float(r_start)}

    lambda_ridge = lambda_start - r_start / vk
    ll = (betahat_ls_norm * (betahat_ls_norm - vk)) / np.sum(
        np.where(nonzero, uy_sq / (d_sq_safe * d_sq_safe), 0.0)
    )
    uu = ftf / vk

    pow_low = pow_ + 0.5
    pow_up = pow_ - 0.5
    l_stop = np.exp(-0.5 * np.log1p(aa**pow_low))
    u_stop = np.exp(-0.5 * np.log1p(aa**pow_up))
    s_norm = 0.0
    phi_ratio = 0.0

    for _ in range(maxit):
        if lambda_ridge <= ll or lambda_ridge >= uu:
            lambda_ridge = max(0.0001 * uu, np.sqrt(ll * uu))

        d_lambda = (dvec / (d_sq + lambda_ridge)) ** 2
        d_prime = d_lambda / (d_sq + lambda_ridge)
        s_norm = float(np.sqrt(np.sum(uy_sq * d_lambda)))
        phi_val = s_norm - vk
        phi_der = -float(np.sum(uy_sq * d_prime)) / s_norm
        phi_ratio = phi_val / phi_der

        if s_norm <= u_stop * betahat_ls_norm and s_norm >= l_stop * betahat_ls_norm:
            break

        uu = uu if phi_val >= 0 else lambda_ridge
        ll = max(ll, lambda_ridge - phi_ratio)
        lambda_ridge = lambda_ridge - (s_norm * phi_ratio) / vk

    return {"lambda": float(lambda_ridge), "rr": float(s_norm * phi_ratio)}


def _anderson_step(Ftmp, Xtmp, fnew, a1, kappa, shrink_count, lambda_ridge, r_penalty):
    u, dvec, vt = np.linalg.svd(Ftmp, full_matrices=False)
    dvec_sq = dvec * dvec
    uy = u.T @ fnew
    uy_sq = uy * uy
    ftf = float(np.sqrt(np.sum(uy_sq * dvec_sq)))
    tmp_lam = _damping_find(
        uy_sq,
        dvec,
        a1,
        kappa,
        shrink_count,
        ftf,
        lambda_start=lambda_ridge,
        r_start=r_penalty,
    )
    lambda_ridge = tmp_lam["lambda"]
    r_penalty = tmp_lam["rr"]
    dd = (dvec * uy) / (dvec_sq + lambda_ridge)
    gamma_vec = vt.T @ dd
    xbar_delta = Xtmp @ gamma_vec
    fbar_delta = Ftmp @ gamma_vec
    return xbar_delta, fbar_delta, lambda_ridge, r_penalty


def _daarem_base_objfn(
    par,
    fixptfn,
    objfn,
    maxiter,
    tol,
    mon_tol,
    cycl_mon_tol,
    a1,
    kappa,
    nlag,
    check_par_resid,
    intermed,
    args,
    kwargs,
):
    f_cols, x_cols = [], []
    xold = par
    xnew = _call_vector(fixptfn, xold, args, kwargs)
    obj_funvals = [
        _call_scalar(objfn, xold, args, kwargs),
        _call_scalar(objfn, xnew, args, kwargs),
    ]
    obj_evals = 2

    fold = xnew - xold
    k = 1
    count = 0
    shrink_count = 0
    lambda_ridge = 100000.0
    r_penalty = 0.0
    conv = True
    ell_star = obj_funvals[1]
    p_inter = (
        [_with_value(obj_funvals[0], par), _with_value(obj_funvals[1], xnew)]
        if intermed
        else None
    )

    while k < maxiter:
        count += 1
        try:
            ftmp = _call_vector(fixptfn, xnew, args, kwargs)
        except Exception:
            break
        fnew = ftmp - xnew
        ss_resids = _norm2(fnew)
        if ss_resids < tol and check_par_resid:
            break

        f_cols.append(fnew - fold)
        x_cols.append(xnew - xold)
        xbar_delta, fbar_delta, lambda_ridge, r_penalty = _anderson_step(
            np.stack(f_cols, axis=1),
            np.stack(x_cols, axis=1),
            fnew,
            a1,
            kappa,
            shrink_count,
            lambda_ridge,
            r_penalty,
        )

        xbar = xnew - xbar_delta
        fbar = fnew - fbar_delta
        x_propose = xbar + fbar
        try:
            new_objective_val = _call_scalar(objfn, x_propose, args, kwargs)
            proposal_ok = not math.isnan(new_objective_val)
        except Exception:
            new_objective_val = math.nan
            proposal_ok = False
        obj_evals += 1

        if (
            proposal_ok
            and not math.isnan(obj_funvals[k])
            and not math.isnan(new_objective_val)
        ):
            if new_objective_val >= obj_funvals[k] - mon_tol:
                obj_funvals.append(new_objective_val)
                fold = fnew
                xold = xnew
                xnew = x_propose
                shrink_count += 1
            else:
                fold = fnew
                xold = xnew
                xnew = fold + xold
                obj_funvals.append(_call_scalar(objfn, xnew, args, kwargs))
                obj_evals += 1
        else:
            fold = fnew
            xold = xnew
            xnew = fold + xold
            obj_funvals.append(_call_scalar(objfn, xnew, args, kwargs))
            obj_evals += 1
            count = 0
            f_cols, x_cols = [], []

        # R uses only the history columns 1:count, so a restart (count = 0)
        # starts a new history.
        if count == nlag:
            count = 0
            f_cols, x_cols = [], []
            if obj_funvals[k + 1] < ell_star - cycl_mon_tol:
                shrink_count = max(shrink_count - nlag, -2 * kappa)
            ell_star = obj_funvals[k + 1]

        if abs(obj_funvals[k + 1] - obj_funvals[k]) < tol and not check_par_resid:
            break

        if intermed:
            p_inter.append(_with_value(obj_funvals[k + 1], xnew))
        k += 1

    value_obj = _call_scalar(objfn, xnew, args, kwargs)
    if k >= maxiter:
        conv = False
    if intermed and k < maxiter:
        p_inter.append(_with_value(value_obj, xnew))

    return {
        "par": xnew,
        "fpevals": k,
        "value.objfn": value_obj,
        "objfevals": obj_evals,
        "convergence": conv,
        "objfn.track": _drop_nan(obj_funvals),
        "p.intermed": None if p_inter is None else np.stack(p_inter),
    }


def _daarem_base_noobjfn(
    par,
    fixptfn,
    maxiter,
    tol,
    resid_tol,
    a1,
    kappa,
    nlag,
    intermed,
    args,
    kwargs,
):
    f_cols, x_cols = [], []
    rho = resid_tol

    xold = par
    xnew = _call_vector(fixptfn, xold, args, kwargs)
    fold = xnew - xold
    fnew = _call_vector(fixptfn, xnew, args, kwargs) - xnew
    ss_resids = _norm2(fnew)
    resid_vals = [_norm2(fold), ss_resids]

    p_inter = [par, xnew] if intermed else None
    fp_evals = 2
    k = 1
    n_aa = 0
    count = 0
    shrink_count = 0
    lambda_ridge = 100000.0
    r_penalty = 0.0
    conv = True

    while fp_evals < maxiter:
        count += 1
        f_cols.append(fnew - fold)
        x_cols.append(xnew - xold)
        xbar_delta, fbar_delta, lambda_ridge, r_penalty = _anderson_step(
            np.stack(f_cols, axis=1),
            np.stack(x_cols, axis=1),
            fnew,
            a1,
            kappa,
            shrink_count,
            lambda_ridge,
            r_penalty,
        )

        xbar = xnew - xbar_delta
        fbar = fnew - fbar_delta
        x_propose = xbar + fbar

        try:
            f_propose = _call_vector(fixptfn, x_propose, args, kwargs) - x_propose
        except Exception:
            break
        fp_evals += 1
        ss_propose = _norm2(f_propose)

        if ss_propose <= ss_resids * (1.0 + rho**k):
            fold = fnew
            xold = xnew
            xnew = x_propose
            fnew = f_propose
            shrink_count += 1
            ss_resids = ss_propose
            n_aa += 1
        else:
            fold = fnew
            xold = xnew
            xnew = fold + xold
            fnew = _call_vector(fixptfn, xnew, args, kwargs) - xnew
            ss_resids = _norm2(fnew)
            fp_evals += 1

        resid_vals.append(ss_resids)
        if ss_resids < tol and count == nlag:
            break

        if count == nlag:
            count = 0
            f_cols, x_cols = [], []

        if intermed:
            p_inter.append(xnew)
        k += 1

    if fp_evals >= maxiter:
        conv = False

    return {
        "par": xnew,
        "fpevals": fp_evals,
        "value.objfn": None,
        "objfevals": None,
        "convergence": conv,
        "objfn.track": None,
        "residuals": _drop_nan(resid_vals),
        "p.intermed": None if p_inter is None else np.stack(p_inter),
    }
