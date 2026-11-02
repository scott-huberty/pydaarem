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

## Development

This project uses [pixi](https://pixi.sh) to manage the Python and R environments.

```bash
pixi run test                                  # install R daarem from CRAN, then run pytest
pixi run -e lint lint                          # run ruff and numpydoc on all files
pixi run -e lint pre-commit install            # run the same checks before each commit
pixi run -e docs docs                          # build the documentation in docs/_build/html
```

The documentation contains the API reference and the examples in `examples/`.
sphinx-gallery runs each example during the build.

The parity tests compare pydaarem with the R `daarem` package, version 0.7.

## License

pydaarem is a Python port of the R package [daarem](https://CRAN.R-project.org/package=daarem) (version 0.7) by Nicholas Henderson and Ravi Varadhan, and is distributed under the same license, the GNU General Public License version 2 (GPL-2.0-only). See `LICENSE`.
