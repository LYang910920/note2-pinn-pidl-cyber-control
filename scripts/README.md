# Script Guide

Scripts are grouped by purpose: quick validation, figure generation, and longer training diagnostics.

| Script | Purpose | Typical command | Outputs |
|---|---|---|---|
| `run_smoke_tests.sh` | Fast confidence check for every executable component. | `bash scripts/run_smoke_tests.sh` | Console output only |
| `generate_figures.py` | Rebuild static explanatory figures used by the README. | `python scripts/generate_figures.py` | `figures/*.png` |
| `run_training_iterations.py` | Run longer inverse PINN, PIDL, direct-control, PMP-informed, and baseline-comparison diagnostics. | `python scripts/run_training_iterations.py` | `experiments/*.csv`, `experiments/OUTPUT_PREVIEW.md`, `experiments/training_summary.md`, `figures/training_iteration_diagnostics.png`, `figures/baseline_comparison.png` |

## Runtime Notes

`run_smoke_tests.sh` is the fast check used by GitHub Actions.  `run_training_iterations.py` is longer because it is meant to show loss reduction over time.

Use `--iters` to change the length of each PINN/PIDL teaching run:

```bash
python scripts/run_training_iterations.py --iters 1000
```

The checked-in CSV files and figures are examples from one deterministic teaching run.  For research claims, rerun with multiple seeds and report uncertainty.

`run_training_iterations.py` writes two kinds of evidence:

| Evidence | Use it for |
|---|---|
| Training histories and `figures/training_iteration_diagnostics.png` | checking whether losses and residuals move toward a stable low-error regime |
| `baseline_comparison_metrics.csv` and `figures/baseline_comparison.png` | checking whether the trained method beats a simple, topic-specific baseline |

## What Each Script Needs

| Script | Needs | Expected result |
|---|---|---|
| `run_smoke_tests.sh` | installed dependencies from `requirements.txt` | exits with status 0 and unit-test summary |
| `generate_figures.py` | working PyTorch/Matplotlib environment | rewrites the PNG files under `figures/` |
| `run_training_iterations.py` | working PyTorch/Matplotlib environment | rewrites training CSVs, baseline metrics, output preview, summary markdown, and the diagnostic PNGs |
