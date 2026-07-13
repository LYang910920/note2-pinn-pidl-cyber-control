# Reproducibility

## Commands

```bash
python -m pip install -e "../network-control-differential-games[torch]"
python -m pip install -e ".[dev]"
python -m cyberpinn smoke
python -m cyberpinn medium --device auto --output-dir artifacts/medium
python -m cyberpinn figures
python -m cyberpinn docs
```

`auto` tries MPS, then CUDA, then CPU through the Foundation device resolver. The
resolved device is logged. Importing the package does not select a device, change
thread counts, or seed global RNGs.

## Smoke and Medium Profiles

`smoke` checks tensor shapes, residual execution, short optimization paths, and
node-SIPS mass error. It does not establish estimation quality.

`medium` uses three seeds and records:

- aggregate inverse PINN, PIDL, neural-control, and PMP-informed diagnostics;
- dense and factorized node-time architectures;
- zero and nonzero observation noise;
- two observed-node counts;
- held-out node/time masks;
- transfer to an unseen node count;
- runtime, device, parameter count, state/parameter/residual/mass errors.

The profile is bounded for local validation. A paper experiment should increase
optimization budgets only through an explicit configuration and retain failed
seeds in its report.

## Required Checks

1. Compare generated truth with an independent numerical solver.
2. Verify SIPS mass conservation and NumPy/Torch residual parity.
3. Log data, residual, initial-condition, constraint, and regularization losses separately.
4. Evaluate unobserved times and nodes, not only training points.
5. Compare heterogeneous estimation with a matched homogeneous misspecification baseline.
6. Report effective transmission error under the susceptibility/infectivity gauge.
7. Repeat across noise, sparsity, parameter seeds, and graph settings.

Low residual and data losses can coexist with inaccurate parameters. Report the
full metric set and any run that did not converge.

## Outputs

Generated CSV, JSON, checkpoints, and temporary figures belong under ignored
`artifacts/`. `medium_manifest.json` records the seeds and experiment factors;
`evaluation_masks.csv` records exactly which node/time points were observed.

## PDF Build

```bash
cd docs/source
latexmk -pdf -interaction=nonstopmode note2_pinn_pidl_cyber_control.tex
```
