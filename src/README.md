# Source Code Guide

The `src/` folder contains four independent teaching examples.  Each file can be run directly with `--smoke` for a fast execution check.

## State Convention

All examples use SIR-style malware propagation:

| Symbol | Meaning |
|---|---|
| `S` | susceptible or vulnerable devices |
| `I` | infected, compromised, or active-malware devices |
| `R` | recovered, cleaned, patched, or protected devices |
| `u` | mitigation control intensity in optimal-control examples |
| `lambda` | costate variable in the PMP-informed example |

State networks use `softmax` so `S + I + R = 1` by construction.  Control networks use `sigmoid` so `0 <= u <= umax`.

## Module Map

| File | Main purpose | Useful entry points |
|---|---|---|
| `experiment_profiles.py` | Student-facing map from method to loss terms, edit points, and paper extensions. | `PROFILES`, `get_profile`, `describe_profiles` |
| `inverse_pinn_sir_malware.py` | Learn hidden states and unknown `beta`, `gamma` from sparse infected observations. | `generate_data`, `StateNet`, `train` |
| `pidl_unknown_mechanism.py` | Combine known SIR dynamics with a learned correction term. | `generate`, `CorrectionNet`, `train` |
| `control_pinn_malware.py` | Train state and control networks by direct optimal-control loss minimization. | `StateNet`, `ControlNet`, `rhs`, `train` |
| `pmp_informed_pinn_malware.py` | Train state, costate, and control networks with PMP residuals. | `hamiltonian`, `time_derivative`, `train` |

## Inputs And Outputs

| Component | Needs | Produces |
|---|---|---|
| Experiment profiles | method name, script, losses, first functions to edit | quick commands and paper-extension route |
| Inverse PINN | sparse infected observations, collocation points, initial condition | state network, learned `beta`/`gamma`, loss history |
| PIDL | sparse observations, known dynamics, correction-network architecture | state network, correction network, residual and correction diagnostics |
| Direct control PINN | dynamics, cost weights, initial condition, collocation points | state/control networks and objective/residual history |
| PMP-informed PINN | Hamiltonian terms, boundary conditions, collocation points | state/costate/control networks and PMP residual history |

## How The Pieces Fit

1. `experiment_profiles.py` tells you which method, losses, and functions to modify first.
2. Inverse PINN asks: can sparse observations identify hidden states and parameters?
3. PIDL asks: if part of the dynamics is known, can a network learn only the missing mechanism?
4. Direct control PINN asks: can a neural control reduce malware while satisfying the ODE?
5. PMP-informed PINN asks: can the network satisfy the optimality system, not only the state dynamics?

## First Extension Step

Run:

```bash
python src/experiment_profiles.py
```

Pick the closest profile, then change one part at a time: state variables, known dynamics, observations, loss terms, control bounds, Hamiltonian, or architecture.

## Teaching-Code Boundaries

These are compact examples for learning and adaptation.  For research-grade studies, add multiple seeds, held-out trajectories, uncertainty estimates, and identifiability checks.

For network-scale extensions, read `docs/EXTENDING.md` before changing code.
For visible parameter and neural-training settings, read `docs/PARAMETERS.md` or run `python src/experiment_profiles.py`.
