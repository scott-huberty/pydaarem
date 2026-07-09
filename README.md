# pydaarem

Python interpretation of DAAREM: damped Anderson acceleration with restarts and monotonicity control for fixed-point iterations.

## Install

From the repository root:

```bash
python -m pip install -e ./pydaarem
```

## Use

```python
from pydaarem import daarem

result = daarem(par, fixptfn, objfn, control={"maxiter": 1000})
```

The public controls mirror the R implementation: `maxiter`, `order`, `tol`, `mon.tol`, `cycl.mon.tol`, `kappa`, `alpha`, `resid.tol`, `convtype`, and `intermed`.
