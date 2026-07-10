# Reproducibility

## Install

With the sibling foundation repository:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e "../network-control-differential-games[torch]"
python -m pip install -e ".[dev]" --no-deps
```

After the foundation `0.2.0` branch is merged, the Git dependency in
`pyproject.toml` supports a standalone checkout:

```bash
python -m pip install -e ".[dev]"
```

## Public commands

```bash
python -m cyberpinn smoke
python -m cyberpinn medium --device auto --output-dir artifacts/medium
python -m cyberpinn figures
python -m cyberpinn docs
python -m cyberpinn all
```

`auto` chooses MPS, then CUDA, then CPU. The node inverse runner falls back to
CPU when an MPS build cannot execute the required higher-order residual slicing;
the selected backend is recorded rather than silently described as MPS.

The medium profile uses seeds `31, 43, 59`. It runs all four aggregate methods
for 150 updates and paired dense/factorized node-SIPS inverse studies for 300
updates in the full factorial combination of noise levels `0` and `0.02` with
4 or 3 observed nodes. Noise and observation density are therefore reported
as separate factors.
Factorized models are evaluated on an unseen 10-node graph. Outputs are
`medium_metrics.csv` and `medium_manifest.json`.

## Validation

```bash
python -m compileall -q src tests scripts
python -m ruff check src tests scripts
python -m pytest -q
python -m cyberpinn smoke
python -m cyberpinn figures
python -m cyberpinn docs
```

Tests cover residual shapes, finite values, SIPS mass conservation,
NumPy/Torch parity, deterministic seeds, noise projection, held-out masks,
identifiability gauge, matched architecture size and unseen-node transfer.

Inspect each non-smoke run for:

- pre-update training losses and separately labeled post-training metrics;
- data, residual, IC/BC, constraint and regularization losses;
- observed, held-out-time and held-out-node state error;
- parameter and effective-transmission error;
- mass/nonnegativity error and independent rollout objective;
- seed, noise, sparsity, architecture, hardware and runtime.

Do not select a method from total loss alone because loss weights differ by
method.

## Figures, PDF and literature

`python -m cyberpinn figures` regenerates experiment plots and shared vector
diagrams. Curated assets live in `docs/assets/`; generated runs stay under
ignored `artifacts/`.

The main source is `docs/source/note2_pinn_pidl_cyber_control.tex`.
`python -m cyberpinn docs` runs `latexmk` and copies the current PDF to
`docs/note2_pinn_pidl_cyber_control.pdf`.

`docs/literature/literature_matrix.csv` records evidence status. Rows marked
`requested` cannot support detailed architecture claims until an owner-supplied,
legally obtained PDF is reviewed. Such PDFs stay in ignored
`literature_pdfs/`.
