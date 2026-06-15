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
| `inverse_pinn_sir_malware.py` | Learn hidden states and unknown `beta`, `gamma` from sparse infected observations. | `generate_data`, `StateNet`, `train` |
| `pidl_unknown_mechanism.py` | Combine known SIR dynamics with a learned correction term. | `generate`, `CorrectionNet`, `train` |
| `control_pinn_malware.py` | Train state and control networks by direct optimal-control loss minimization. | `StateNet`, `ControlNet`, `rhs`, `train` |
| `pmp_informed_pinn_malware.py` | Train state, costate, and control networks with PMP residuals. | `hamiltonian`, `time_derivative`, `train` |

## How The Pieces Fit

1. Inverse PINN asks: can sparse observations identify hidden states and parameters?
2. PIDL asks: if part of the dynamics is known, can a network learn only the missing mechanism?
3. Direct control PINN asks: can a neural control reduce malware while satisfying the ODE?
4. PMP-informed PINN asks: can the network satisfy the optimality system, not only the state dynamics?

## Teaching-Code Boundaries

These are compact examples for learning and adaptation.  For research-grade studies, add multiple seeds, held-out trajectories, uncertainty estimates, and identifiability checks.
