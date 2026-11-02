# Physics-informed learning for cyber dynamics

Infer propagation rates and hidden states from sparse observations, then ask whether those estimates reproduce the dynamics when used in an ODE solver. Continue to learned mechanism corrections, neural control and node-level inference.

**First run:** one inverse-SIR experiment comparing a PINN with ordinary ODE parameter fitting. **Read next:** [the walkthrough](docs/GUIDE.md). The [lecture notes](docs/note2_pinn_pidl_cyber_control.pdf) develop PINN, PIDL, neural optimal control and PMP-based training.

## Install and run

Use Python 3.11 or 3.12 and install Git. From this repository:

```bash
python -m venv .venv
# macOS / Linux:
source .venv/bin/activate
# Windows PowerShell instead:
# .venv\Scripts\Activate.ps1

python -m pip install -e .
python -m cyberpinn
```

Installation automatically fetches the tested shared `cybercontrol` revision. The default runs on a CPU and uses generated data. A GPU, external dataset and LaTeX are not required. On Windows or Linux without a GPU, CPU-only PyTorch can be installed first:

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e .
```

The example writes `results/fit.png`, `learning.png`, the observations, trajectories, training history and a comparison table. PINN and ODE fitting see exactly the same infection observations. A third curve integrates the ODE using the PINN's estimated rates, separating a fitted neural trajectory from the dynamics implied by its parameters.

```bash
python -m cyberpinn --iters 3000 --noise 0.01 --observations 20 --output-dir results/noisy
```

The synthetic truth is available for teaching evaluation only. The reported held-out-time error measures interpolation between observed times, not forecasting beyond the training interval or proof of parameter identifiability.

## Read the code

| Question | Start here |
|---|---|
| What does the complete experiment do? | [`cli.py`](src/cyberpinn/cli.py): `run_example` |
| How are observations and physical losses constructed? | [`inverse.py`](src/cyberpinn/inverse.py): `observation_data`, `train` |
| How is the ODE baseline fitted? | [`cli.py`](src/cyberpinn/cli.py): `fit_ode` |
| How do I adapt the model or add control? | [Walkthrough](docs/GUIDE.md) |

The [foundation tutorial](https://github.com/LYang910920/network-control-differential-games) supplies the common dynamics and numerical tools. [Note 1](https://github.com/LYang910920/note1-cyber-control-games) studies learned feedback decisions using those dynamics.

For development: `python -m pip install -e ".[dev]"`, then `pytest -q`. Source attribution and licensing are in [LICENSE](LICENSE) and [NOTICE.md](NOTICE.md).
