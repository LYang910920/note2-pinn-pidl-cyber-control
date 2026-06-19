# Start Here

This page is the compact map. You can ignore most files at first.

## Big Picture

```text
Foundation repository
  -> notation, ODE models, shared cybercontrol helpers
Companion Note 2
  -> PINN/PIDL model families
  -> inverse learning, missing mechanisms, neural control
  -> figures and training diagnostics
```

## Three-Repository Order

| Step | Repository | What to use it for |
|---:|---|---|
| 0 | `network-control-differential-games` | Foundation notation, shared package, continuous/impulse/hybrid worked examples, and degree-level/node-level scalability. |
| 1 | `note1-cyber-control-games` | PMP/FBSM, sampled-data MDPs, DDQN, CTDE/MADRL, and cyber game-learning diagnostics. |
| 2 | `note2-pinn-pidl-cyber-control` | This note: PINN/PIDL, inverse learning, neural control, PMP-informed residuals, and sparse-data validation. |

## Five-Minute Path

1. Open `docs/note2_pinn_pidl_cyber_control.pdf` for the tutorial narrative.
2. Run `python src/experiment_profiles.py` to see the student-facing method profiles.
3. Open `docs/PARAMETERS.md` before changing PINN/PIDL hyperparameters.
4. For a heavier local/GPU diagnostic, run `python scripts/run_training_iterations.py --profile gpu` after the smoke tests pass.
5. Read `docs/PAPER_WORKFLOW.md` when turning an example into a paper section.
6. Run `bash scripts/run_smoke_tests.sh` to verify the environment.
7. Run `python scripts/generate_figures.py` to recreate the figures.
8. Run `python scripts/run_training_iterations.py` for longer loss curves and baseline comparisons.
9. Read `docs/EXTENDING.md` when you want to scale the model.

## Folder Map

| Path | Purpose |
|---|---|
| `docs/` | tutorial note, implementation notes, extension guide |
| `docs/PARAMETERS.md` | model parameters, loss weights, collocation settings, and neural hyperparameters |
| `docs/PAPER_WORKFLOW.md` | paper workflow for inverse PINN, PIDL, neural control, and PMP-informed PINN |
| `src/` | executable PINN/PIDL examples |
| `scripts/` | commands for figures, smoke tests, and diagnostics |
| `experiments/` | CSV histories, baseline metrics, output preview, and training summary |
| `figures/` | generated plots used by the README |
| `tests/` | small regression tests |

## Code Reading Order

1. `src/shared_setup.py`: local helper that finds the shared foundation package in a sibling workspace.
2. `src/experiment_profiles.py`: named method profiles and first functions to edit.
3. `src/inverse_pinn_sir_malware.py`: sparse observations, ODE residuals, inverse parameters.
4. `src/pidl_unknown_mechanism.py`: known mechanism plus learned correction.
5. `src/control_pinn_malware.py`: direct state/control optimization.
6. `src/pmp_informed_pinn_malware.py`: state, costate, control, and Hamiltonian residuals.

For command details, use `scripts/README.md`. For module details, use `src/README.md`.
