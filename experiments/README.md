# Training Iteration Experiments

Run:

```bash
python scripts/run_training_iterations.py
```

The script performs longer, CPU-friendly PINN/PIDL teaching runs and writes:

| File | Meaning |
|---|---|
| `inverse_pinn_training_history.csv` | Inverse PINN total, data, ODE, initial-condition losses plus beta/gamma estimates. |
| `pidl_training_history.csv` | PIDL total, data, residual, correction regularizer, and mean learned correction. |
| `control_pinn_training_history.csv` | Direct control PINN objective, residual loss, initial-condition loss, and mean control. |
| `pmp_informed_pinn_training_history.csv` | PMP-informed total, state, costate, stationarity, and boundary losses. |
| `training_summary.md` | First-versus-last loss reductions and interpretation. |

The companion plot is saved as `figures/training_iteration_diagnostics.png`.

The default configuration is longer than smoke mode and is intended to show losses approaching a stable low-error regime. For the PMP-informed example, inspect stationarity and state residuals alongside the total loss.
