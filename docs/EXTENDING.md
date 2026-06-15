# Extending Note 2

This repository starts with compact PINN/PIDL examples.  The extension path is to keep the same loss decomposition while replacing the small SIR system with richer dynamics, data, and neural architectures.

## What You Need

| Need | Where it appears now |
|---|---|
| State variables and constraints | `StateNet` classes in `src/` |
| Known dynamics | `sir_rhs`, `known_rhs`, `rhs`, or `f_state` |
| Sparse observations | `generate_data`, `generate`, and `t_data`/`I_data` construction |
| Unknown parameters | `beta_raw`, `gamma_raw`, and `positive` transforms |
| Missing mechanisms | `CorrectionNet` in `pidl_unknown_mechanism.py` |
| Optimality residuals | `hamiltonian`, `H_x`, and `H_u` in `pmp_informed_pinn_malware.py` |

## What You Get

The current scripts produce:

| Output | Meaning |
|---|---|
| `figures/*.png` | Visual checks for sparse data, missing mechanisms, and architecture diagrams |
| `experiments/*.csv` | Logged loss terms and learned quantities |
| `experiments/training_summary.md` | First-versus-last interpretation of the diagnostics |
| GitHub Actions smoke tests | Basic confidence that scripts and figures still run |

## Extension Path

1. **Change the state.** Add compartments, node-level states, uncertain parameters, budget states, or exogenous features.
2. **Preserve constraints.** Use `softmax`, sigmoid transforms, projection layers, or penalty terms to keep states and controls meaningful.
3. **Replace the dynamics.** Start by changing `sir_rhs` or `f_state`. Then update residual losses and tests.
4. **Add data carefully.** Keep a small synthetic case before using real data. Add noise, missing observations, and held-out trajectories one at a time.
5. **Separate loss terms.** Log data loss, residual loss, boundary loss, stationarity loss, and regularizers separately.
6. **Scale architecture last.** Move from MLPs to Fourier features, residual networks, GNNs, or domain-specific encoders only after the loss terms behave correctly.

## Scaling To Network Models

The current examples use population-level SIR states.  Large cyber or social-network models can use:

| Representation | Use when | Implementation direction |
|---|---|---|
| Degree-level PINN | nodes can be grouped by degree | learn one trajectory per degree class |
| Node-level PINN | every node has its own state | use vector-valued state networks and sparse adjacency operations |
| Graph neural PINN | topology should affect propagation or control | combine GNN layers with residual losses |
| Operator-learning model | many network instances are needed | learn a map from graph/features to trajectories or controls |
| Hybrid neural control | continuous dynamics plus impulses | combine PINN residuals with jump or event losses |

For the network optimal-control foundation, compare with:

https://github.com/LYang910920/network-control-differential-games

Useful pieces from that repository:

| Concept | Where to look there | How it helps here |
|---|---|---|
| Degree-k dynamics | `examples/lecture/code/simple_degree_k_control.py` | Shows how to move from scalar SIR to degree-class arrays |
| Node-level games | `examples/lecture/code/network_control_examples.py` | Shows state and control variables indexed by graph nodes |
| PMP and Hamiltonian updates | `docs/lecture_note.pdf` and lecture code | Gives target residuals for PMP-informed neural training |
| Hybrid impulse examples | `examples/lecture/code/network_control_examples.py` | Suggests jump losses for PINN/PIDL hybrid systems |

## Research-Grade Checklist

Before treating a run as evidence rather than a teaching demo:

1. Run multiple random seeds and report spread.
2. Add held-out time windows or held-out trajectories.
3. Compare against classical solvers, no-control baselines, and simple rule-based policies.
4. Test identifiability of learned parameters or missing mechanisms.
5. Track constraint violations and residual terms separately.
6. Report data noise, observation frequency, and collocation sampling strategy.
7. Keep dataset, dependency, and third-party license notes with the experiment.
