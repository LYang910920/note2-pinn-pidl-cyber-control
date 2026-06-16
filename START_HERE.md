# Start Here

This repository is designed as a practical companion to Note 2.  Start with the concept you need, then open the matching script.

## Five-Minute Path

1. Open `docs/note2_pinn_pidl_cyber_control.pdf` for the lecture narrative.
2. Read `PROJECT_MAP.md` for the big-picture repository structure.
3. Read `README.md` for the high-level map and figures.
4. Run `bash scripts/run_smoke_tests.sh` to verify the environment.
5. Run `python scripts/generate_figures.py` to recreate the explanatory figures.
6. Run `python scripts/run_training_iterations.py` when you want longer PINN/PIDL loss curves.
7. Read `docs/EXTENDING.md` when you want to move beyond the small teaching models.

## Find The Right File

| If you want to... | Open |
|---|---|
| Understand the equations first | `docs/note2_pinn_pidl_cyber_control.pdf` |
| See the whole repo at a glance | `PROJECT_MAP.md` |
| See the recommended reading order | `docs/README.md` |
| Understand the Python modules | `src/README.md` |
| Know which command generates which output | `scripts/README.md` |
| Inspect loss curves and training metrics | `experiments/README.md` and `experiments/training_summary.md` |
| Connect this repo to the differential-games repo | `docs/LEARNING_PATH.md` |
| Extend to complex cyber or network models | `docs/EXTENDING.md` |
| Check the license and copyright assumptions | `LICENSE` and `NOTICE.md` |

## Recommended Code Reading Order

1. `src/inverse_pinn_sir_malware.py`: sparse observations, ODE residuals, inverse parameters.
2. `src/pidl_unknown_mechanism.py`: known mechanism plus learned correction.
3. `src/control_pinn_malware.py`: direct state/control optimization.
4. `src/pmp_informed_pinn_malware.py`: state, costate, control, and Hamiltonian residuals.

## Common First Changes

| Change | Where |
|---|---|
| Shorten or lengthen training | CLI `--iters`, or `scripts/run_training_iterations.py --iters 1000` |
| Change collocation density | CLI `--n-collocation` |
| Change sparse observation count | CLI `--n-data` for inverse/PIDL examples |
| Change malware dynamics | `beta`, `gamma`, and correction parameters in `src/` |
| Change optimal-control penalties | `A`, `B`, `AT`, and loss weights in control scripts |
| Move to degree-level, node-level, or graph-neural PINNs | `docs/EXTENDING.md` plus `network-control-differential-games` |
