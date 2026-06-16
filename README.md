# PINN and PIDL Cyber Control, Note 2

Executable companion for **Note 2: PINN/PIDL for Cyber Control**.  The repo keeps four teaching examples in one place: inverse PINNs, PIDL with a missing mechanism, direct neural optimal control, and PMP-informed PINNs.

If this is your first visit, start with `START_HERE.md`.

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
bash scripts/run_smoke_tests.sh
python scripts/generate_figures.py
```

For longer loss-curve diagnostics:

```bash
python scripts/run_training_iterations.py
```

## Repository Guide

| Need | Open |
|---|---|
| Short orientation | `START_HERE.md` |
| Lecture narrative | `docs/note2_pinn_pidl_cyber_control.pdf` |
| Source-code map | `src/README.md` |
| Script and output map | `scripts/README.md` |
| Training curves and CSVs | `experiments/README.md` |
| Extensions and scaling | `docs/EXTENDING.md` |
| License and attribution | `LICENSE`, `NOTICE.md` |

## Core Flow

```text
cyber ODE model
  -> sparse data and residual losses
  -> inverse PINN / PIDL
  -> neural control PINN
  -> PMP-informed optimality residuals
```

![Neural architectures](figures/neural_architectures.png)

## Main Outputs

| Output | Purpose |
|---|---|
| `figures/inverse_pinn_sparse_data.png` | sparse-observation inverse-learning setup |
| `figures/pidl_missing_mechanism.png` | known dynamics plus learned missing mechanism |
| `figures/training_iteration_diagnostics.png` | longer inverse PINN, PIDL, control PINN, and PMP-informed diagnostics |
| `experiments/*.csv` | logged histories behind the training plot |

## Validation

`bash scripts/run_smoke_tests.sh` runs the fast local check.  GitHub Actions repeats the smoke tests and regenerates figures on each push or pull request.

These examples are teaching code, not calibrated cyber-risk models.  For research use, add noisy-data studies, identifiability checks, multiple seeds, held-out trajectories, and uncertainty estimates.

## Related Repository

For the optimal-control and differential-game foundation behind PMP residuals and network-scale extensions, see https://github.com/LYang910920/network-control-differential-games.

## License And Copyright

Released under the MIT License.  See `LICENSE` for terms and `NOTICE.md` for copyright, dependency, and attribution notes.
