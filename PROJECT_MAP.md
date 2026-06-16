# Project Map

Use this page when you want the whole repository in one view.

## Big Picture

This repo connects four layers:

```text
lecture note
  -> PINN/PIDL model families
  -> inverse learning, missing-mechanism learning, and neural control
  -> figures, CSV diagnostics, and smoke tests
```

The main question is:

> How can physics-informed neural methods learn cyber dynamics, controls, and optimality conditions when observations are sparse or mechanisms are only partially known?

## Folder Roles

| Folder or file | Role | Read first when... |
|---|---|---|
| `START_HERE.md` | First-stop onboarding path. | You are new to the repo. |
| `README.md` | Main public-facing overview. | You want the summary, figures, and quick commands. |
| `docs/` | Lecture PDFs plus extension and cross-repo guides. | You want theory, learning path, or scale-up guidance. |
| `src/` | Core executable PINN/PIDL examples. | You want to change losses, dynamics, or networks. |
| `scripts/` | Reproducible commands for figures and diagnostics. | You want to regenerate outputs. |
| `experiments/` | Saved CSV histories and interpretation. | You want to inspect loss curves and learned quantities. |
| `figures/` | Generated visual outputs. | You want visual sanity checks. |
| `tests/` | Small regression checks. | You want to know what behavior is protected. |

## Code Flow

```text
src/inverse_pinn_sir_malware.py
  -> sparse data + ODE residual + learned beta/gamma

src/pidl_unknown_mechanism.py
  -> known dynamics + correction network

src/control_pinn_malware.py
  -> state network + control network + direct objective

src/pmp_informed_pinn_malware.py
  -> state network + costate network + control network + PMP residuals
```

Read this as:

1. Inverse PINN learns hidden states and unknown parameters.
2. PIDL learns only the part of the mechanism that is missing.
3. Direct control PINN optimizes a neural control without costates.
4. PMP-informed PINN trains against state, costate, and stationarity residuals.

## Command Flow

```text
pip install -r requirements.txt
  -> bash scripts/run_smoke_tests.sh
  -> python scripts/generate_figures.py
  -> python scripts/run_training_iterations.py
```

The smoke tests answer: does the code run?

The figure script answers: do the examples produce readable visual outputs?

The training diagnostic script answers: do the logged loss terms move in a sensible direction?

## Output Flow

| Command | Output |
|---|---|
| `bash scripts/run_smoke_tests.sh` | console output and unit-test summary |
| `python scripts/generate_figures.py` | `figures/inverse_pinn_sparse_data.png`, `figures/pidl_missing_mechanism.png`, `figures/neural_architectures.png` |
| `python scripts/run_training_iterations.py` | `experiments/*.csv`, `experiments/training_summary.md`, `figures/training_iteration_diagnostics.png` |

## If You Want To Extend It

Start with `docs/EXTENDING.md`.

For the larger optimal-control and differential-game foundation, use:

https://github.com/LYang910920/network-control-differential-games
