# Script Guide

Scripts are grouped by purpose: quick validation, figure generation, and longer training diagnostics.

| Script | Purpose | Typical command | Outputs |
|---|---|---|---|
| `run_smoke_tests.sh` | Fast confidence check for every executable component. | `bash scripts/run_smoke_tests.sh` | Console output only |
| `generate_figures.py` | Rebuild static explanatory figures used by the README. | `python scripts/generate_figures.py` | `figures/*.png` |
| `run_training_iterations.py` | Run longer inverse PINN, PIDL, direct-control, and PMP-informed diagnostics. | `python scripts/run_training_iterations.py` | `experiments/*.csv`, `experiments/training_summary.md`, `figures/training_iteration_diagnostics.png` |

## Runtime Notes

`run_smoke_tests.sh` is the fast check used by GitHub Actions.  `run_training_iterations.py` is longer because it is meant to show loss reduction over time.

Use `--iters` to change the length of each PINN/PIDL teaching run:

```bash
python scripts/run_training_iterations.py --iters 1000
```

The checked-in CSV files and figures are examples from one deterministic teaching run.  For research claims, rerun with multiple seeds and report uncertainty.
