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
| `OUTPUT_PREVIEW.md` | Categorized first-stop summary of each experiment, diagnostic panel, and output file. |
| `training_summary.md` | First-versus-last loss reductions and interpretation. |

Start with `OUTPUT_PREVIEW.md` for a compact result map. The companion plot is saved as `figures/training_iteration_diagnostics.png`.

The default configuration is longer than smoke mode and is intended to show losses approaching a stable low-error regime. For the PMP-informed example, inspect stationarity and state residuals alongside the total loss.

## How To Read The CSV Files

Each row is a logged checkpoint rather than every optimizer step.  Use `OUTPUT_PREVIEW.md` for the first categorized pass, then inspect the full curve in `figures/training_iteration_diagnostics.png`.

| Question | Column to inspect |
|---|---|
| Did the inverse PINN learn a plausible fit? | `loss`, `data_loss`, `ode_loss`, `beta`, and `gamma` |
| Did PIDL use the correction term sensibly? | `residual_loss`, `correction_regularizer`, and `mean_correction` |
| Did direct control reduce the objective while respecting dynamics? | `objective`, `residual_loss`, and `mean_control` |
| Did the PMP-informed model learn optimality conditions? | `state_loss`, `costate_loss`, `stationarity_loss`, and `boundary_loss` |
