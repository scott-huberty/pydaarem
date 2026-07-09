# Examples

## Mixture proportions example

This mirrors `optimizers/examples/daarem/analysis/mixem.Rmd`.

Run from the repository root:

```bash
/Users/scotterik/miniforge3/envs/pydaarem/bin/python pydaarem/examples/mixem_compare.py
```

The script writes outputs to `pydaarem/examples/output/`:

- `mixem_objective_comparison.png`
- `mixem_objective_tracks.csv`
- `mixem_summary.txt`

It compares three fits on `mixdata.RData`: plain EM, `pydaarem`, and the R DAAREM reference implementation.
