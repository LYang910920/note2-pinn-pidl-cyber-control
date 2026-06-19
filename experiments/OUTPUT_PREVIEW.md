# Output Preview

Use this page as the first stop after running `python scripts/run_training_iterations.py`.

Profile used: `teaching` with `600` optimizer iterations per method.

## 1. What Each Experiment Checks

| Experiment | Main question | Key diagnostics |
|---|---|---|
| Sparse-data inverse PINN | Can sparse infected observations recover hidden S/I/R states and beta/gamma? | total loss, data loss, ODE residual, learned beta/gamma |
| PIDL missing mechanism | Can a small correction network represent the missing nonlinear propagation term? | residual loss, correction regularizer, mean learned correction |
| Direct control PINN | Can state/control networks reduce malware while satisfying the ODE? | objective, state residual, initial-condition loss, mean control |
| PMP-informed PINN | Can state, costate, and control networks satisfy PMP optimality conditions? | state residual, costate residual, stationarity loss, boundary loss |

## 2. Training Diagnostic Panels

Open `figures/training_iteration_diagnostics.png`.

| Panel | What to check |
|---|---|
| Sparse-data inverse PINN | total loss and ODE residual should decrease together |
| PIDL missing mechanism | residual loss should fall while the correction remains interpretable |
| Direct control PINN | objective should fall without losing dynamics consistency |
| PMP-informed PINN | stationarity and state/costate residuals should move toward a low-error regime |

## 3. Baseline Comparison Panels

Open `figures/baseline_comparison.png`.

| Panel | What is being compared | Main metric |
|---|---|---|
| Sparse-data inverse PINN | learned inverse PINN vs sparse interpolation and wrong-parameter SIR | full S/I/R mean squared error |
| PIDL missing mechanism | learned correction vs known SIR without the missing term | full S/I/R mean squared error |
| Controlled malware mitigation: objective | no control, fixed controls, direct PINN, PMP-informed PINN, and rollout-optimized neural control | infected burden plus control cost |
| Controlled malware mitigation: epidemic burden | same policies as above | integral of compromised devices over time |

Best rollout-control objective in this run: **Rollout-optimized neural control** with objective **6.153e+00**.

The direct-control and PMP-informed PINN panels above are training diagnostics. The baseline rollout panels use the original ODE simulator, so they deliberately show whether a learned control remains strong after it is rolled forward outside the training loss. A wrong-parameter SIR rollout means the equation form is correct but beta/gamma are deliberately inaccurate; it is a simple baseline for inverse parameter learning.

## 4. First-Versus-Last Snapshot

| Diagnostic | Start | End | End/start |
|---|---:|---:|---:|
| Inverse PINN total loss | 1.286e+00 | 1.255e-03 | 9.756e-04 |
| PIDL total loss | 2.543e+00 | 3.564e-03 | 1.401e-03 |
| Direct control PINN total loss | 9.360e+01 | 2.636e-01 | 2.816e-03 |
| PMP-informed stationarity loss | 2.971e-01 | 5.581e-03 | 1.879e-02 |

## 5. Files To Open First

| Category | File |
|---|---|
| Summary | `experiments/training_summary.md` |
| Diagnostic glossary | `experiments/training_diagnostic_glossary.md` |
| Learning curves | `figures/training_iteration_diagnostics.png` |
| Baseline comparison | `figures/baseline_comparison.png` |
| Baseline metrics | `experiments/baseline_comparison_metrics.csv` |
| Inverse PINN CSV | `experiments/inverse_pinn_training_history.csv` |
| PIDL CSV | `experiments/pidl_training_history.csv` |
| Direct control CSV | `experiments/control_pinn_training_history.csv` |
| PMP-informed CSV | `experiments/pmp_informed_pinn_training_history.csv` |
