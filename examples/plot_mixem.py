"""
Accelerate an EM algorithm for mixture proportions
==================================================

This example uses DAAREM to accelerate a simple expectation maximization (EM)
algorithm. The algorithm computes maximum-likelihood estimates of the mixture
proportions in a mixture model.

The example is a Python version of the
`mixem analysis <https://stephenslab.github.io/daarem/mixem.html>`_ in the
`stephenslab/daarem <https://github.com/stephenslab/daarem>`_ repository.
"""

# %%
# Load the data
# -------------
# The data is a conditional likelihood matrix ``L`` that comes from a simulated
# data set. Element ``L[i, k]`` is the likelihood of sample ``i`` under mixture
# component ``k``. The data also contains the mixture proportions ``x_best`` of
# a reference solution. :func:`~pydaarem.datasets.fetch_mixdata` downloads the
# file on the first call.

import time

import matplotlib.pyplot as plt
import numpy as np
from pydaarem import daarem
from pydaarem.datasets import fetch_mixdata

L, x_best = fetch_mixdata()
n, m = L.shape
print(f"Loaded {n} x {m} data matrix.")

# %%
# Define the EM update and the objective
# --------------------------------------
# The E step computes the posterior probability of each component for each
# sample. The M step sets the mixture proportions to the mean of these
# probabilities. The small value ``e`` prevents a division by zero. Each row of
# ``L`` gets its own value of ``e``, scaled by the largest value in the row.

e = 1e-15 * L.max(axis=1)


def mixem_update(x):
    """Do one EM update of the mixture proportions."""
    p = L * x + e[:, None]
    p /= p.sum(axis=1, keepdims=True)
    return p.mean(axis=0)


def mixobjective(x):
    """Compute the log-likelihood of the mixture proportions."""
    y = L @ x + e
    return np.sum(np.log(y)) if np.all(y > 0) else -np.inf


# %%
# DAAREM can propose iterates that are not on the probability simplex. Project
# each iterate onto the simplex before the EM update and before the objective.


def project_iterate(x):
    """Set negative proportions to zero and scale the sum to one."""
    x = np.maximum(x, 0)
    return x / x.sum()


def daarem_update(x):
    """Do one EM update of the projected mixture proportions."""
    return mixem_update(project_iterate(x))


def daarem_objective(x):
    """Compute the log-likelihood of the projected mixture proportions."""
    return mixobjective(project_iterate(x))


# %%
# Run the basic EM updates
# ------------------------
# Start from equal mixture proportions and run 200 EM updates.

numiter = 200
x0 = np.full(m, 1 / m)

start = time.perf_counter()
x_em = x0.copy()
em_values = np.empty(numiter)
for i in range(numiter):
    x_em = mixem_update(x_em)
    em_values[i] = mixobjective(x_em)
print(f"Computation took {time.perf_counter() - start:.2f} seconds.")
print(f"Log-likelihood at the EM estimate is {mixobjective(x_em):.12f}.")

# %%
# Run the accelerated EM updates
# ------------------------------
# Run the same number of updates with DAAREM. The control values are the same
# as in the R analysis. ``tol=0`` makes DAAREM run all 200 iterations, so
# DAAREM gives a warning that the algorithm did not converge.

start = time.perf_counter()
fit = daarem(
    x0,
    daarem_update,
    daarem_objective,
    control={
        "maxiter": numiter,
        "order": 10,
        "tol": 0,
        "mon.tol": 0.05,
        "kappa": 20,
        "alpha": 1.2,
    },
)
x_daarem = project_iterate(fit["par"])
daarem_values = fit["objfn.track"][1:]
print(f"Computation took {time.perf_counter() - start:.2f} seconds.")
print(f"Log-likelihood at the DAAREM estimate is {mixobjective(x_daarem):.12f}.")

# %%
# The DAAREM estimate has a much larger log-likelihood than the EM estimate.
#
# Plot the improvement in the solution
# ------------------------------------
# The vertical axis shows the difference between the log-likelihood of the
# reference solution and the log-likelihood at each iteration.

EM_COLOR = "#ef5f00"  # Radix orange 10
DAAREM_COLOR = "#0d9b8a"  # Radix teal 10
INK = "#60646c"  # Radix slate 11
GRID = "#e8e8ec"  # Radix slate 4

f_best = mixobjective(x_best)
iterations = np.arange(1, numiter + 1)
series = [
    ("EM", em_values, EM_COLOR),
    ("DAAREM", daarem_values, DAAREM_COLOR),
]

fig, ax = plt.subplots(figsize=(7, 4), layout="constrained")
for label, values, color in series:
    distance = f_best - values
    ax.plot(iterations, distance, color=color, linewidth=2, label=label)
    ax.annotate(
        label,
        xy=(iterations[-1], distance[-1]),
        xytext=(6, 0),
        textcoords="offset points",
        va="center",
        color=INK,
    )
ax.set_yscale("log")
ax.set_xlabel("Iteration", color=INK)
ax.set_ylabel("Distance from the reference solution", color=INK)
ax.tick_params(colors=INK)
ax.grid(True, which="major", color=GRID, linewidth=0.8)
ax.set_axisbelow(True)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
for side in ("left", "bottom"):
    ax.spines[side].set_color(GRID)
ax.legend(frameon=False, labelcolor=INK)
plt.show()

# %%
# DAAREM gets much closer to the reference solution than the basic EM
# algorithm. After approximately 100 iterations, the improvement becomes
# small.
