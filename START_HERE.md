# Start Here

This page is the compact map. You can ignore most files at first.

## Big Picture

```text
tutorial note
  -> PINN/PIDL model families
  -> inverse learning, missing mechanisms, neural control
  -> figures and training diagnostics
```

## Five-Minute Path

1. Open `docs/note2_pinn_pidl_cyber_control.pdf` for the tutorial narrative.
2. Run `bash scripts/run_smoke_tests.sh` to verify the environment.
3. Run `python scripts/generate_figures.py` to recreate the figures.
4. Run `python scripts/run_training_iterations.py` for longer loss curves and baseline comparisons.
5. Read `docs/EXTENDING.md` when you want to scale the model.

## Folder Map

| Path | Purpose |
|---|---|
| `docs/` | tutorial note, implementation notes, extension guide |
| `src/` | executable PINN/PIDL examples |
| `scripts/` | commands for figures, smoke tests, and diagnostics |
| `experiments/` | CSV histories, baseline metrics, output preview, and training summary |
| `figures/` | generated plots used by the README |
| `tests/` | small regression tests |

## Code Reading Order

1. `src/inverse_pinn_sir_malware.py`: sparse observations, ODE residuals, inverse parameters.
2. `src/pidl_unknown_mechanism.py`: known mechanism plus learned correction.
3. `src/control_pinn_malware.py`: direct state/control optimization.
4. `src/pmp_informed_pinn_malware.py`: state, costate, control, and Hamiltonian residuals.

For command details, use `scripts/README.md`. For module details, use `src/README.md`.
