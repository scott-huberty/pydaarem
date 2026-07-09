"""Compare pydaarem to the R DAAREM mixture-proportions example.

This mirrors optimizers/examples/daarem/analysis/mixem.Rmd:

    - load mixdata.RData
    - fit mixture proportions with plain EM
    - fit with pydaarem
    - fit with the R DAAREM reference
    - save an objective-track plot and CSV for inspection
"""

from __future__ import annotations

import csv
import os
from pathlib import Path

import numpy as np
from rpy2 import robjects
from rpy2.robjects import numpy2ri

from pydaarem import daarem


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = Path("/Users/scotterik/amica_test_data/daarem/mixdata.RData")
R_DAAREM_ROOT = REPO_ROOT / "optimizers" / "daarem"
R_EXAMPLE_ROOT = REPO_ROOT / "optimizers" / "examples" / "daarem"
OUTPUT_DIR = Path(__file__).resolve().parent / "output"
MPL_CONFIG_DIR = OUTPUT_DIR / "matplotlib"
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))

import matplotlib.pyplot as plt


def load_mixdata():
    with (robjects.default_converter + numpy2ri.converter).context():
        env = robjects.Environment()
        robjects.r["load"](str(DATA_PATH), envir=env)
        return np.asarray(env["L"], dtype=float), np.asarray(env["x"], dtype=float)


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


def fit_em(L, x0, e, numiter):
    x = x0.copy()
    values = np.empty(numiter, dtype=float)
    for i in range(numiter):
        x = mixem_update(L, x, e)
        values[i] = mixobjective(L, x, e)
    return x, values


def fit_pydaarem(L, x0, e, numiter=200, order=10):
    out = daarem(
        x0,
        mixdaarem_update,
        mixdaarem_objective,
        L,
        e,
        control={
            "maxiter": numiter,
            "order": order,
            "tol": 0,
            "mon.tol": 0.05,
            "kappa": 20,
            "alpha": 1.2,
        },
    )
    return {
        "x": project_iterate(out["par"]),
        "value": np.asarray(out["objfn.track"][1:], dtype=float),
        "raw": out,
    }


def fit_r_daarem(numiter=200, order=10):
    script = f"""
    source("{R_DAAREM_ROOT / "R" / "DampingFind.R"}")
    source("{R_DAAREM_ROOT / "R" / "daarem_base_objfn.R"}")
    source("{R_DAAREM_ROOT / "R" / "daarem_base_noobjfn.R"}")
    source("{R_DAAREM_ROOT / "R" / "daarem.R"}")
    source("{R_EXAMPLE_ROOT / "code" / "misc.R"}")
    source("{R_EXAMPLE_ROOT / "code" / "mixem.R"}")
    load("{DATA_PATH}")
    e <- 1e-15 * apply(L, 1, max)
    x0 <- rep(1 / ncol(L), ncol(L))
    out <- suppressWarnings(
      daarem(x0, mixdaarem.update, mixdaarem.objective, L, e,
             control = list(maxiter = {numiter}, order = {order}, tol = 0,
                            mon.tol = 0.05, kappa = 20, alpha = 1.2)))
    list(
      x = project.iterate(out$par),
      value = out$objfn.track[-1],
      fpevals = out$fpevals,
      objfevals = out$objfevals,
      convergence = out$convergence
    )
    """
    out = robjects.r(script)
    with (robjects.default_converter + numpy2ri.converter).context():
        return {
            "x": np.asarray(out.rx2("x"), dtype=float),
            "value": np.asarray(out.rx2("value"), dtype=float),
            "fpevals": int(out.rx2("fpevals")[0]),
            "objfevals": int(out.rx2("objfevals")[0]),
            "convergence": bool(out.rx2("convergence")[0]),
        }


def write_tracks_csv(path, em_values, py_values, r_values):
    max_len = max(len(em_values), len(py_values), len(r_values))
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["iter", "em", "pydaarem", "r_daarem"])
        for i in range(max_len):
            writer.writerow(
                [
                    i + 1,
                    em_values[i] if i < len(em_values) else "",
                    py_values[i] if i < len(py_values) else "",
                    r_values[i] if i < len(r_values) else "",
                ]
            )


def plot_tracks(path, em_values, py_values, r_values, best_value):
    fig, ax = plt.subplots(figsize=(8, 5))
    it_em = np.arange(1, len(em_values) + 1)
    it_py = np.arange(1, len(py_values) + 1)
    it_r = np.arange(1, len(r_values) + 1)

    ax.plot(it_em, best_value - em_values, label="EM", color="darkorange", linewidth=1.8)
    ax.plot(it_py, best_value - py_values, label="pydaarem", color="dodgerblue", linewidth=1.8)
    ax.plot(it_r, best_value - r_values, label="R DAAREM", color="black", linestyle="--", linewidth=1.4)
    ax.set_yscale("log")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Distance from best objective")
    ax.set_title("Mixture proportions: EM vs DAAREM")
    ax.legend()
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    L, x_best = load_mixdata()
    n, m = L.shape
    e = 1e-15 * np.max(L, axis=1)
    x0 = np.repeat(1.0 / m, m)
    numiter = 200

    em_x, em_values = fit_em(L, x0, e, numiter)
    py_fit = fit_pydaarem(L, x0, e, numiter=numiter)
    r_fit = fit_r_daarem(numiter=numiter)

    best_value = mixobjective(L, x_best, e)
    em_final = mixobjective(L, em_x, e)
    py_final = mixobjective(L, py_fit["x"], e)
    r_final = mixobjective(L, r_fit["x"], e)

    write_tracks_csv(
        OUTPUT_DIR / "mixem_objective_tracks.csv",
        em_values,
        py_fit["value"],
        r_fit["value"],
    )
    plot_tracks(
        OUTPUT_DIR / "mixem_objective_comparison.png",
        em_values,
        py_fit["value"],
        r_fit["value"],
        best_value,
    )

    summary = [
        f"Loaded L with shape {n} x {m}",
        f"Best/reference objective from x in RData: {best_value:.12f}",
        f"EM final objective: {em_final:.12f}",
        f"pydaarem final objective: {py_final:.12f}",
        f"R DAAREM final objective: {r_final:.12f}",
        f"pydaarem - R final objective: {py_final - r_final:.12e}",
        f"max abs x difference pydaarem vs R: {np.max(np.abs(py_fit['x'] - r_fit['x'])):.12e}",
        f"pydaarem fpevals/objfevals/convergence: {py_fit['raw']['fpevals']} / {py_fit['raw']['objfevals']} / {py_fit['raw']['convergence']}",
        f"R fpevals/objfevals/convergence: {r_fit['fpevals']} / {r_fit['objfevals']} / {r_fit['convergence']}",
        f"Wrote {OUTPUT_DIR / 'mixem_objective_tracks.csv'}",
        f"Wrote {OUTPUT_DIR / 'mixem_objective_comparison.png'}",
    ]
    text = "\n".join(summary)
    (OUTPUT_DIR / "mixem_summary.txt").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
