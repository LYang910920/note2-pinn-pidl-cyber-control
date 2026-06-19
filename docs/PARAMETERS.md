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

## Terms Used In This Repo

| Term | Meaning in Note 2 |
|---|---|
| `trajectory` | A time-indexed state path such as `[S(t),I(t),R(t)]` or node-level `[S_i(t),I_i(t),P_i(t),R_i(t)]`. PINN/PIDL state networks approximate trajectories. |
| `rollout` | A forward simulation through the original ODE or graph simulator using a fixed parameter set or control policy. It is used for validation because it evaluates behavior outside the training loss. |
| `wrong-parameter rollout` | A deliberately misspecified baseline: it uses the correct SIR model form but inaccurate beta/gamma values, then rolls the ODE forward to show what happens without inverse parameter learning. |
| `baseline` | A simple method evaluated on the same topic and metric, such as sparse interpolation, known-SIR-only dynamics, no control, fixed control, or a rollout-optimized control. |
| `rollout objective` | The malware-control objective computed after simulating the controlled ODE forward: infected burden plus control cost and terminal infection penalty. Lower is better within the same control topic. |
| `robustness` | Sensitivity to noise, missing states, sparse observations, parameter mismatch, or held-out trajectories. It is reported through held-out error, multiple seeds, and baseline comparisons, not by training loss alone. |
| `PINN` | Physics-informed neural network: a neural state/control approximation trained with data loss plus differential-equation residuals. |
| `PIDL` | Physics-informed deep learning: a broader setup that can combine known dynamics, learned correction terms, differentiable simulation, priors, and residual losses. |
| `collocation points` | Time or state-time points where the equation residual is evaluated, even if no data are observed there. |
| `residual` | The mismatch between the neural derivative and the model right-hand side; small residual means the learned trajectory approximately obeys the assumed dynamics. |
| `PMP-informed` | A PINN/PIDL variant that includes Hamiltonian, costate, and stationarity residuals from Pontryagin's maximum principle. |
| `held-out` | Data, time points, graph seeds, or trajectories not used in fitting; used to check whether the learned model generalizes beyond training observations. |

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

This smoke route is a graph/node bridge. It uses the Foundation SIPRS simulator to generate truth, observes only infected probabilities on a subset of nodes and times, and evaluates hidden-state recovery on held-out time points.

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
| main validation metric | `heldout_state_mse` on unobserved time points plus `beta_abs_error`, `gamma_abs_error`, and `mass_error` |

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
