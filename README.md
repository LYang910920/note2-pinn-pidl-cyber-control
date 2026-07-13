# Physics-Informed Cyber Control

Inverse PINN, PIDL, neural control, PMP-informed learning and heterogeneous
node-SIPS inference. Shared ODE/SIPS equations, graph operations, neural blocks,
device selection, metrics and plotting come from the Foundation package
`cybercontrol`.

It answers three questions:

- What is observed, what is unknown and what may be non-identifiable?
- How are data, residual, initial/boundary and constraint losses assembled?
- Which held-out and independent-rollout checks separate fit from evidence?

The examples use synthetic data. A small residual does not establish parameter
truth, global optimality or deployment safety.

## Repository Family

![Repository family](docs/assets/diagrams/repository_family.png)

| Order | Repository | Responsibility |
|---:|---|---|
| 0 | [Network Control and Differential Games](https://github.com/LYang910920/network-control-differential-games) | Shared equations, heterogeneity, graphs, FBSM, neural utilities and plotting |
| 1 | [Cyber Control and Game Learning](https://github.com/LYang910920/note1-cyber-control-games) | Sampled environments, DDQN/PPO, CTDE, MAPPO and game evaluation |
| 2 | **Physics-Informed Cyber Control** | Inverse PINN, PIDL, neural control and PMP-informed learning |

New readers should follow the Foundation
[Learning Path](https://github.com/LYang910920/network-control-differential-games/blob/main/docs/LEARNING_PATH.md)
before selecting a neural architecture.

## First Run

Python 3.10 or newer is required. With sibling checkouts, install the reviewed
Foundation and this repository:

```bash
python -m pip install -e "../network-control-differential-games[torch]"
python -m pip install -e ".[dev]" --no-deps
```

Then run the bounded smoke check, normally well under one minute:

```bash
python -m cyberpinn smoke
```

## Expected Output

| Output | Interpretation |
|---|---|
| `artifacts/smoke_summary.json` | Logged-row counts, node-SIPS mass error, Git and hardware provenance for the short training paths |
| terminal exit code `0` | Data generation, residual construction and training paths completed without an exception |

The smoke losses are execution checks, not parameter-recovery results.

## Read These Two Files

1. [`src/cyberpinn/node_problem.py`](src/cyberpinn/node_problem.py): synthetic
   heterogeneous SIPS truth, observations, masks and homogeneous baseline.
2. [`src/cyberpinn/node_inverse.py`](src/cyberpinn/node_inverse.py): dense and
   factorized estimators, residuals, gauges and held-out evaluation.

Exact loss terms, tensor shapes and architecture defaults are in
[`docs/METHODS_AND_API.md`](docs/METHODS_AND_API.md).

## Change One Thing

Set `NodeInverseTrainConfig.noise` to `0.02` in
[`configs.py`](src/cyberpinn/configs.py). Keep the observation mask and seed
fixed, then compare held-out state, effective-transmission and residual errors.
Noise may affect these metrics differently; do not judge the run by total loss
alone.

## Medium Experiment

Run the three-seed noise/sparsity and architecture profile:

```bash
python -m cyberpinn medium --device auto --output-dir artifacts/medium
```

Inspect:

- `artifacts/medium/medium_metrics.csv` for state, parameter/effective-rate,
  residual, mass and baseline errors;
- `artifacts/medium/evaluation_masks.csv` for observed and held-out nodes/times;
- `artifacts/medium/medium_manifest.json` for seeds, architectures, hardware and
  runtime settings.

The node estimator compares a dense model with a factorized node-time model at
a matched trainable-parameter budget. Its gauge removes one scale ambiguity but
does not make all rates identifiable.

![Node-time encoder-decoder](docs/assets/diagrams/node_time_encoder_decoder.png)

![PINN loss assembly](docs/assets/diagrams/pinn_loss.png)

## Next Experiment

Train on one graph seed and node count, then evaluate a new graph seed and an
unseen size while preserving a known initial state. Add noise/sparsity sweeps and
compare homogeneous misspecification, community-specific and factorized models.
The full protocol is in
[`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md).

## Extension Route

1. Define observations, unknowns, units and identifiability before the network.
2. Reserve held-out times, nodes, trajectories or graphs before training.
3. Compare classical/known-mechanism and homogeneous baselines.
4. Log each loss component and validate through the original simulator.
5. Keep graph equations and reusable neural blocks in the Foundation.

The repository-specific paper checklist is
[`docs/FROM_MODEL_TO_PAPER.md`](docs/FROM_MODEL_TO_PAPER.md).

## Validation

The maintained install, test, figure and PDF commands are in
[`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md). Tests cover residual
dimensions, NumPy/Torch parity, SIPS mass, deterministic seeds and held-out
evaluation masks.

## Citation and License

Code, documentation and generated figures are MIT-licensed unless a file says
otherwise. See [`LICENSE`](LICENSE) and [`NOTICE.md`](NOTICE.md). Cite the
Foundation for `cybercontrol` and the relevant method source recorded under
`docs/literature/`.
