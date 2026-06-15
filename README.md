# PINN and PIDL Cyber Control, Note 2

This repository is the executable companion for **Note 2: PINN/PIDL for Cyber Control**.  It focuses on physics-informed neural networks, physics-informed deep learning, inverse parameter learning, and neural optimal-control examples for malware-style propagation models.

The scripts are compact by design: each file is a readable starting point for one modeling idea, with small smoke modes for quick verification.

## Repository Map

| Path | Purpose |
|---|---|
| `docs/note2_pinn_pidl_cyber_control.pdf` | Main lecture note for PINN/PIDL cyber-control methods. |
| `docs/implementation_companion.pdf` | Companion explanation for implementation choices. |
| `docs/code_run_guide.pdf` | General run guide from the original bundle. |
| `src/inverse_pinn_sir_malware.py` | Inverse PINN that learns malware propagation parameters from sparse `I(t)` data. |
| `src/pidl_unknown_mechanism.py` | PIDL example with known SIR dynamics plus a learned missing mechanism. |
| `src/control_pinn_malware.py` | Direct neural-control PINN for malware mitigation. |
| `src/pmp_informed_pinn_malware.py` | PMP-informed PINN using state, costate, and stationarity residuals. |
| `scripts/generate_figures.py` | Generates explanatory figures in `figures/`. |
| `scripts/run_smoke_tests.sh` | Runs all fast checks for this repo. |
| `tests/` | Small regression tests for data generation, constraints, and autograd connectivity. |

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

## Main Ideas

This repo keeps four related learning tasks separate:

1. **Inverse PINN**: fit hidden state trajectories and unknown propagation parameters under ODE residual constraints.
2. **PIDL**: keep the known cyber mechanism explicit and learn only a missing correction term.
3. **Direct control PINN**: optimize a state network and control network against dynamics residuals plus a cost objective.
4. **PMP-informed PINN**: train state, costate, and control networks against the PMP optimality system.

## Figures

`figures/inverse_pinn_sparse_data.png` shows the sparse-observation setting used by the inverse PINN example.

![Inverse PINN sparse-data setup](figures/inverse_pinn_sparse_data.png)

`figures/pidl_missing_mechanism.png` shows the synthetic missing term `q S I^2` that the PIDL correction network is designed to recover.

![PIDL missing nonlinear mechanism](figures/pidl_missing_mechanism.png)

## Validation

The repo includes smoke tests for every executable script and unit tests for the core tensor constraints.  The PMP-informed script has been adjusted so `H_x` and `H_u` are computed on the live autograd graph; this ensures stationarity residuals actually train the neural control model.

These examples are teaching code, not calibrated cyber-risk models.  For publication-grade experiments, add noisy-data studies, identifiability checks, multiple seeds, held-out trajectories, and uncertainty estimates.
