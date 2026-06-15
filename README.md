# PINN and PIDL Cyber Control, Note 2

This repository is the executable companion for **Note 2: PINN/PIDL for Cyber Control**.  It focuses on physics-informed neural networks, physics-informed deep learning, inverse parameter learning, and neural optimal-control examples for malware-style propagation models.

The scripts are compact by design: each file is a readable starting point for one modeling idea, with small smoke modes for quick verification.

## Start Here

If this is your first time opening the repo, read `START_HERE.md` first.  It gives a five-minute path, a file finder, and the recommended code-reading order.

## Repository Map

| Path | Purpose |
|---|---|
| `START_HERE.md` | First-stop guide for new readers. |
| `docs/note2_pinn_pidl_cyber_control.pdf` | Main lecture note for PINN/PIDL cyber-control methods. |
| `docs/README.md` | Reading path and lecture-structure guide. |
| `docs/implementation_companion.pdf` | Companion explanation for implementation choices. |
| `docs/code_run_guide.pdf` | General run guide from the original bundle. |
| `src/README.md` | Source-code map, state conventions, and model responsibilities. |
| `src/inverse_pinn_sir_malware.py` | Inverse PINN that learns malware propagation parameters from sparse `I(t)` data. |
| `src/pidl_unknown_mechanism.py` | PIDL example with known SIR dynamics plus a learned missing mechanism. |
| `src/control_pinn_malware.py` | Direct neural-control PINN for malware mitigation. |
| `src/pmp_informed_pinn_malware.py` | PMP-informed PINN using state, costate, and stationarity residuals. |
| `scripts/README.md` | Command guide for validation, figures, and longer diagnostics. |
| `scripts/generate_figures.py` | Generates explanatory figures in `figures/`. |
| `scripts/run_training_iterations.py` | Runs longer PINN/PIDL teaching diagnostics and writes CSV histories in `experiments/`. |
| `scripts/run_smoke_tests.sh` | Runs all fast checks for this repo. |
| `.github/workflows/smoke-tests.yml` | GitHub Actions workflow for dependency install, smoke tests, and figure generation. |
| `experiments/` | Small training-iteration CSV outputs and an explanation of each metric. |
| `tests/` | Small regression tests for data generation, constraints, and autograd connectivity. |
| `LICENSE` and `NOTICE.md` | MIT license, copyright, and attribution notes. |

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Run the full smoke check:

```bash
bash scripts/run_smoke_tests.sh
```

Generate the figures used in this README:

```bash
python scripts/generate_figures.py
```

Run longer training-iteration diagnostics:

```bash
python scripts/run_training_iterations.py
```

## Common Workflows

| Goal | Command or file |
|---|---|
| Check that everything runs | `bash scripts/run_smoke_tests.sh` |
| Rebuild README figures | `python scripts/generate_figures.py` |
| Rebuild loss diagnostics | `python scripts/run_training_iterations.py` |
| Increase training time | `python scripts/run_training_iterations.py --iters 1000` |
| Understand module responsibilities | `src/README.md` |
| Understand command outputs | `scripts/README.md` |

## Main Ideas

This repo keeps four related learning tasks separate:

1. **Inverse PINN**: fit hidden state trajectories and unknown propagation parameters under ODE residual constraints.
2. **PIDL**: keep the known cyber mechanism explicit and learn only a missing correction term.
3. **Direct control PINN**: optimize a state network and control network against dynamics residuals plus a cost objective.
4. **PMP-informed PINN**: train state, costate, and control networks against the PMP optimality system.

## Figures

`figures/neural_architectures.png` summarizes the two main neural structures: inverse/PIDL state learning and PMP-informed control PINNs.

![Neural architectures](figures/neural_architectures.png)

`figures/inverse_pinn_sparse_data.png` shows the sparse-observation setting used by the inverse PINN example.

![Inverse PINN sparse-data setup](figures/inverse_pinn_sparse_data.png)

`figures/pidl_missing_mechanism.png` shows the synthetic missing term `q S I^2` that the PIDL correction network is designed to recover.

![PIDL missing nonlinear mechanism](figures/pidl_missing_mechanism.png)

## Training Diagnostics

`scripts/run_training_iterations.py` writes four experiment tables and a summary:

| CSV | What to inspect |
|---|---|
| `experiments/inverse_pinn_training_history.csv` | Parameter estimates and whether data/ODE losses are both decreasing. |
| `experiments/pidl_training_history.csv` | Whether the learned missing mechanism is used without dominating the known dynamics. |
| `experiments/control_pinn_training_history.csv` | Objective, dynamics residual, and mean control across training. |
| `experiments/pmp_informed_pinn_training_history.csv` | State, costate, stationarity, and boundary residuals for the PMP system. |
| `experiments/training_summary.md` | First-versus-last loss reductions and interpretation. |

The default run is intentionally longer than smoke mode.  The inverse PINN, PIDL, and direct-control PINN losses should move toward a low-error regime.  For the PMP-informed PINN, the stationarity residual is the cleanest quick sanity signal because the costate boundary term can dominate the total loss early in training.

The combined plot is:

![Training iteration diagnostics](figures/training_iteration_diagnostics.png)

## Validation

The repo includes smoke tests for every executable script and unit tests for the core tensor constraints.  It also has a GitHub Actions workflow that installs dependencies, runs smoke tests, and regenerates figures on push or pull request.

This consolidated version keeps the Note 2 PDFs, source materials, generated figures, tests, CI workflow, and longer teaching diagnostics in one final repo.  The PMP-informed script computes `H_x` and `H_u` on the live autograd graph so stationarity residuals train the neural control model as intended.

These examples are teaching code, not calibrated cyber-risk models.  For publication-grade experiments, add noisy-data studies, identifiability checks, multiple seeds, held-out trajectories, and uncertainty estimates.

## License And Copyright

This repository is released under the MIT License.  See `LICENSE` for the full terms and `NOTICE.md` for copyright, dependency, and attribution notes.
