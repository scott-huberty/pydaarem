# pydaarem

Python interpretation of DAAREM: damped Anderson acceleration with restarts and monotonicity control for fixed-point iterations.

pydaarem supports the [Python array API standard](https://data-apis.org/array-api/latest/). It computes with the array library and on the device of the start value (for example NumPy, PyTorch or CuPy), in float64, and it returns arrays of that library.

Documentation: https://scott-huberty.github.io/pydaarem/

## Install

```bash
pip install pydaarem
```

The example data sets in `pydaarem.datasets` need two more packages. To install them too:

```bash
pip install "pydaarem[datasets]"
```

## Use

```python
from pydaarem import daarem

result = daarem(par, fixptfn, objfn, control={"maxiter": 1000})
```

The controls and the result keys are the R names with each dot replaced by an underscore. The controls are `maxiter`, `order`, `tol`, `mon_tol`, `cycl_mon_tol`, `kappa`, `alpha`, `resid_tol`, `convtype`, and `intermed`.

## Development

This project uses [pixi](https://pixi.sh) to manage the Python and R environments.

```bash
pixi run test                                  # install R daarem from CRAN, then run pytest with coverage
pixi run -e lint lint                          # run ruff and numpydoc on all files
pixi run -e lint pre-commit install            # run the same checks before each commit
pixi run -e docs docs                          # build the documentation in docs/_build/html
```

The documentation contains the API reference and the examples in `examples/`.
sphinx-gallery runs each example during the build.

The parity tests compare pydaarem with the R `daarem` package, version 0.7.

## Release

1. Set the new version in `pyproject.toml`, and merge that change into `main`.
2. Tag the merge commit with the version and push the tag:

   ```bash
   git tag v0.1.0
   git push origin v0.1.0
   ```

The `Release` workflow builds the sdist and the wheel, checks that the tag matches the version, and publishes both on PyPI.

## License

pydaarem is a Python port of the R package [daarem](https://CRAN.R-project.org/package=daarem) (version 0.7) by Nicholas Henderson and Ravi Varadhan, and is distributed under the same license, the GNU General Public License version 2 (GPL-2.0-only). See `LICENSE`.
