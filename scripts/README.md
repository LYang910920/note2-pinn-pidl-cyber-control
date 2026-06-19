# Script Guide

Scripts are grouped by purpose: quick validation, figure generation, and longer training diagnostics.

| Script | Purpose | Typical command | Outputs |
|---|---|---|---|
| `run_smoke_tests.sh` | Fast confidence check for every executable component. | `bash scripts/run_smoke_tests.sh` | Console output only |
| `generate_figures.py` | Rebuild static explanatory figures used by the README. | `python scripts/generate_figures.py` | `figures/*.png` |
| `run_training_iterations.py` | Run longer inverse PINN, PIDL, direct-control, PMP-informed, and baseline-comparison diagnostics. | `python scripts/run_training_iterations.py` | `experiments/*.csv`, `experiments/OUTPUT_PREVIEW.md`, `experiments/training_diagnostic_glossary.md`, `experiments/training_summary.md`, `figures/training_iteration_diagnostics.png`, `figures/baseline_comparison.png` |

## Runtime Notes

`run_smoke_tests.sh` is the fast check used by GitHub Actions. If the virtual environment is not activated, pass the interpreter explicitly:

```bash
PYTHON=../.venv/bin/python bash scripts/run_smoke_tests.sh
```

`run_training_iterations.py` is longer because it is meant to show loss reduction over time.

Use `--iters` to change the length of each PINN/PIDL tutorial run:

```bash
python scripts/run_training_iterations.py --iters 1000
```

Use the heavier profile when you want a larger local/GPU run:

```bash
python scripts/run_training_iterations.py --profile gpu
```

The GPU-oriented profile increases network width/depth, collocation points, sparse observation points, and optimizer iterations. It uses the same scripts and output files as the default profile, so students can compare small and large runs directly.

The checked-in CSV files and figures are examples from one deterministic tutorial run.  For research claims, rerun with multiple seeds and report uncertainty.

`run_training_iterations.py` writes two kinds of evidence:

| Evidence | Use it for |
|---|---|
| Training histories and `figures/training_iteration_diagnostics.png` | checking whether losses and residuals move toward a stable low-error regime |
| `baseline_comparison_metrics.csv` and `figures/baseline_comparison.png` | checking whether the trained method beats a simple, topic-specific baseline |

`experiments/training_diagnostic_glossary.md` defines the plotted terms. Use it when comparing total loss, data loss, ODE residual loss, stationarity loss, mean control, and rollout objective; these are related but not interchangeable.

## What Each Script Needs

| Script | Needs | Expected result |
|---|---|---|
| `run_smoke_tests.sh` | installed dependencies from `requirements.txt` | exits with status 0 and unit-test summary |
| `generate_figures.py` | working PyTorch/Matplotlib environment | rewrites the PNG files under `figures/` |
| `run_training_iterations.py` | working PyTorch/Matplotlib environment | rewrites training CSVs, baseline metrics, output preview, summary markdown, and the diagnostic PNGs |
