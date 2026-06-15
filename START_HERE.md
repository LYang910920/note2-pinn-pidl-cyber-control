# Start Here

This repository is designed as a practical companion to Note 2.  Start with the concept you need, then open the matching script.

## Five-Minute Path

1. Open `docs/note2_pinn_pidl_cyber_control.pdf` for the lecture narrative.
2. Read `README.md` for the high-level map and figures.
3. Run `bash scripts/run_smoke_tests.sh` to verify the environment.
4. Run `python scripts/generate_figures.py` to recreate the explanatory figures.
5. Run `python scripts/run_training_iterations.py` when you want longer PINN/PIDL loss curves.

## Find The Right File

| If you want to... | Open |
|---|---|
| Understand the equations first | `docs/note2_pinn_pidl_cyber_control.pdf` |
| See the recommended reading order | `docs/README.md` |
| Understand the Python modules | `src/README.md` |
| Know which command generates which output | `scripts/README.md` |
| Inspect loss curves and training metrics | `experiments/README.md` and `experiments/training_summary.md` |
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
