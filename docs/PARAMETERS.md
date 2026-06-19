# Parameter And Hyperparameter Reference

Use this page before changing losses, networks, collocation points, or the cyber dynamics.

## Quick Commands

| Need | Command |
|---|---|
| Print method profiles and hyperparameters | `python src/experiment_profiles.py` |
| Fast smoke check | `bash scripts/run_smoke_tests.sh` |
| Rebuild figures | `python scripts/generate_figures.py` |
| Run longer diagnostics | `python scripts/run_training_iterations.py` |
| Run node-SIPRS inverse PINN smoke | `python src/node_siprs_inverse_pinn.py --smoke --device cpu` |
| Run heavier GPU-oriented diagnostics | `python scripts/run_training_iterations.py --profile gpu` |

## Shared Model Parameters

Most examples use SIR-style malware dynamics with:

| Parameter | Default |
|---|---:|
| time horizon `T` | `20.0` |
| propagation rate `beta` | `0.8` |
| recovery/removal rate `gamma` | `0.2` |
| initial state | `[S,I,R]=[0.95,0.05,0.0]` |
| maximum control `umax` | `1.0` in control examples |

## Long Diagnostic Hyperparameters

These are the values used by `python scripts/run_training_iterations.py`.

| Method | Neural/training hyperparameters | Loss/objective weights |
|---|---|---|
| Inverse PINN | `iters=600`, `width=24`, `depth=2`, `n_data=16`, `n_collocation=70`, `lr=1e-3`, `seed=21` | `w_ic=10.0`, `w_ode=1.0` |
| PIDL missing mechanism | `iters=600`, `width=24`, `n_data=18`, `n_collocation=70`, `lr=1e-3`, `seed=22` | `w_ic=10.0`, `w_res=1.0`, `w_corr=1e-3` |
| Direct control PINN | `iters=600`, `width=24`, `n_collocation=70`, `lr=1e-3`, `seed=23` | `A=10.0`, `B=1.0`, `AT=10.0`, `w_res=10.0`, `w_ic=10.0` |
| PMP-informed PINN | `iters=600`, `width=24`, `n_collocation=70`, `lr=1e-3`, `seed=24` | `A=10.0`, `B=1.0`, `AT=10.0`, `w_state=10.0`, `w_costate=1.0`, `w_stat=1.0`, `w_bc=10.0` |

## Node-SIPRS Inverse PINN Parameters

| Parameter | Default |
|---|---:|
| graph nodes | `8` |
| compartments | `[S,I,P,R]` |
| time horizon/grid | `T=6.0`, `grid=61` |
| true rates | `beta=0.82`, `gamma=0.18`, `omega_p=0.03`, `omega_r=0.02` |
| known controls | `patch=0.08`, `clean=0.04` |
| sparse observations | `4` observed nodes, `14` observed time points, infected compartment only |
| training defaults | `iters=500`, `width=32`, `depth=2`, `collocation=32`, `lr=1e-3` |
| smoke profile | `nodes=6`, `grid=25`, `observed_nodes=3`, `observed_times=8`, `collocation=12`, `iters=12` |

## GPU-Oriented Diagnostic Profile

Use this after the smoke tests and default profile pass:

```bash
python scripts/run_training_iterations.py --profile gpu
```

| Method | Larger neural/training settings |
|---|---|
| Inverse PINN | `iters=2000`, `width=128`, `depth=5`, `n_data=40`, `n_collocation=400` |
| PIDL missing mechanism | `iters=2000`, `width=128`, `depth=4`, `n_data=50`, `n_collocation=400` |
| Direct control PINN | `iters=2000`, `width=128`, `depth=4`, `n_collocation=400` |
| PMP-informed PINN | `iters=2000`, `width=128`, `depth=4`, `n_collocation=400` |

## Script CLI Defaults

These are the standalone script defaults if you run each file directly without `--smoke`.

| Script | Important defaults |
|---|---|
| `src/inverse_pinn_sir_malware.py` | `iters=5000`, `width=64`, `depth=4`, `n_data=30`, `n_collocation=200`, `lr=1e-3`, `w_ic=10.0`, `w_ode=1.0` |
| `src/pidl_unknown_mechanism.py` | `iters=5000`, `width=64`, `n_data=40`, `n_collocation=200`, `lr=1e-3`, `w_ic=10.0`, `w_res=1.0`, `w_corr=1e-3` |
| `src/control_pinn_malware.py` | `iters=5000`, `T=20.0`, `width=64`, `n_collocation=200`, `lr=1e-3`, `beta=0.8`, `gamma=0.2`, `umax=1.0`, `A=10.0`, `B=1.0`, `AT=10.0` |
| `src/pmp_informed_pinn_malware.py` | `iters=5000`, `T=20.0`, `width=64`, `n_collocation=200`, `lr=1e-3`, `beta=0.8`, `gamma=0.2`, `umax=1.0`, `A=10.0`, `B=1.0`, `AT=10.0` |
| `src/node_siprs_inverse_pinn.py` | `iters=500`, `nodes=8`, `grid=61`, `observed_nodes=4`, `observed_times=14`, `collocation=32`, `width=32`, `lr=1e-3` |

## What To Change First

| Goal | First file/function |
|---|---|
| Change hidden states or parameters | `src/inverse_pinn_sir_malware.py::generate_data`, `sir_rhs`, `StateNet` |
| Learn a different missing mechanism | `src/pidl_unknown_mechanism.py::known_rhs`, `CorrectionNet` |
| Add more controls or constraints | `src/control_pinn_malware.py::ControlNet`, `rhs` |
| Adapt PMP conditions to a paper model | `src/pmp_informed_pinn_malware.py::f_state`, `hamiltonian` |
| Move from aggregate to graph/node observations | `src/node_siprs_inverse_pinn.py::generate_truth`, `NodeSIPRSStateNet`, `train` |
| Change diagnostic run length or architecture | `scripts/run_training_iterations.py` |
