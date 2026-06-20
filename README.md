# Physics-Informed Cyber Control

Executable tutorial code for PINNs, PIDL, neural optimal control, and PMP-informed residual learning in cyber-control models. This is the third repository in the tutorial family. It uses the foundation package `cybercontrol` for shared ODEs, graph SIPRS dynamics, Torch helper blocks, integration, plotting, and CSV utilities.

## Repository Family

| Order | Repository | Role |
|---:|---|---|
| 0 | [network-control-differential-games](https://github.com/LYang910920/network-control-differential-games) | Foundation notation, shared `cybercontrol` package, continuous/impulse/hybrid examples, degree-vs-node scalability, and reference smoke runs. |
| 1 | [note1-cyber-control-games](https://github.com/LYang910920/note1-cyber-control-games) | FBSM baselines, sampled-data MDP conversion, DDQN defense, compact CTDE, and node-SIPRS MAPPO. |
| 2 | `note2-pinn-pidl-cyber-control` | Inverse PINN, PIDL, direct neural control, PMP-informed PINN, and node-SIPRS inverse-learning smoke examples. |

## 5-Minute Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e "../network-control-differential-games[torch,dev]"
python -m pip install -e ".[dev]"
bash scripts/run_smoke_tests.sh
python scripts/generate_figures.py
```

If this repository is cloned without the sibling foundation repo:

```bash
python -m pip install "cybercontrol[torch] @ git+https://github.com/LYang910920/network-control-differential-games.git"
python -m pip install -e ".[dev]"
```

For bounded diagnostics and baseline comparisons:

```bash
python scripts/run_training_iterations.py
```

## Code Map

| Need | Start here |
|---|---|
| Tutorial PDF | `docs/note2_pinn_pidl_cyber_control.pdf` |
| Run and implementation guide | `docs/code_run_guide.pdf`, `docs/implementation_companion.pdf` |
| Parameters and neural hyperparameters | `docs/PARAMETERS.md` |
| Paper workflow and extensions | `docs/PAPER_WORKFLOW.md`, `docs/EXTENDING.md` |
| Inverse PINN | `src/inverse_pinn_sir_malware.py` |
| PIDL missing-mechanism example | `src/pidl_unknown_mechanism.py` |
| Neural control and PMP-informed PINN | `src/control_pinn_malware.py`, `src/pmp_informed_pinn_malware.py` |
| Node-SIPRS inverse PINN smoke test | `src/node_siprs_inverse_pinn.py` |
| Static figures and bounded diagnostics | `scripts/generate_figures.py`, `scripts/run_training_iterations.py` |

## Representative Experiments

The inverse PINN starts from sparse infected-state observations and learns hidden state curves plus propagation parameters under ODE residual constraints.

![Inverse PINN sparse-data setup](docs/assets/inverse_pinn_sparse_data.png)

The PIDL example keeps the known SIR mechanism explicit and uses a correction network for the synthetic missing nonlinear term.

![PIDL missing nonlinear mechanism](docs/assets/pidl_missing_mechanism.png)

The baseline comparison evaluates learned methods against method-specific alternatives. A rollout means the original ODE or graph simulator is run forward under a parameter set or control policy.

![Baseline comparison for learned methods](docs/assets/baseline_comparison.png)

## Extension Route

1. Read `docs/PARAMETERS.md` before changing collocation points, width/depth, loss weights, or training length.
2. Pick one method file and preserve the meaning of its logged loss terms.
3. Keep common ODE, Torch, graph, plotting, and CSV helpers in `cybercontrol`; add Note 2 code only for PINN/PIDL method logic.
4. Run `bash scripts/run_smoke_tests.sh` after each structural change.
5. Use `python scripts/run_training_iterations.py` for bounded diagnostics. Outputs go to ignored `artifacts/experiments/` and `artifacts/figures/`.

## Validation

```bash
python -m compileall -q src tests scripts
python -m pytest -q
bash scripts/run_smoke_tests.sh
python scripts/generate_figures.py
```

GitHub Actions runs the smoke tests on pushes and pull requests. The examples are tutorial baselines and need additional seed, noise, identifiability, and uncertainty studies before paper-level claims.

## Citation and License

See `LICENSE` and `NOTICE.md`. When using the repository in a paper or report, cite the related publication and the foundation repository when its shared package is used.
