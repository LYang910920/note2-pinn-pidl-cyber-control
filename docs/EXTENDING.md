# Extending Note 2

Keep the same loss decomposition while replacing the small teaching model:

```text
state network -> dynamics residual -> data/boundary loss -> diagnostics
```

## Extension Path

Start by running:

```bash
python src/experiment_profiles.py
```

That command prints the method profiles. Pick the closest one before editing code, because it records the intended method, key loss terms, first functions to edit, and paper-extension route.

1. **Change the state.** Add compartments, node-level states, uncertain parameters, budget states, or exogenous features.
2. **Preserve constraints.** Use `softmax`, sigmoid transforms, projection layers, or penalty terms to keep states and controls meaningful.
3. **Replace the dynamics.** Start by changing `sir_rhs` or `f_state`. Then update residual losses and tests.
4. **Add data carefully.** Keep a small synthetic case before using real data. Add noise, missing observations, and held-out trajectories one at a time.
5. **Separate loss terms.** Log data loss, residual loss, boundary loss, stationarity loss, and regularizers separately.
6. **Scale architecture last.** Move from MLPs to Fourier features, residual networks, GNNs, or domain-specific encoders only after the loss terms behave correctly.

## Scaling To Network Models

For larger models, move from population-level SIR states to degree-level PINNs, node-level vector states, graph-neural PINNs, operator-learning models, or hybrid neural control with jump losses. Keep the first version small and testable before scaling architecture or data.

## From Tutorial Code To Paper Models

| Paper-model ingredient | First teaching hook | What to preserve while extending |
|---|---|---|
| More compartments or hidden states | `StateNet`, `sir_rhs`, `f_state` | state constraints and residual dimensions |
| Sparse, noisy, or partial observations | `generate_data`, data loss blocks | separate data, initial/boundary, and residual losses |
| Unknown cyber mechanisms | `pidl_unknown_mechanism.py` | known physics term plus learned correction term |
| Multiple controls or constraints | `ControlNet`, `rhs`, `hamiltonian` | bounded control transform and explicit objective terms |
| PMP-informed paper model | `f_state`, `hamiltonian`, costate residual block | live autograd graph for `H_x` and `H_u` |
| Network or graph PINNs | profile `paper_extension` notes | a small synthetic case before large data or GNN architecture |

## Related Learning Path

Use these repositories together:

| Step | Repository | Focus |
|---|---|---|
| 1 | `network-control-differential-games` | PMP, FBSM, degree/node-level network control, differential games |
| 2 | `note1-cyber-control-games` | ODE-to-RL conversion, DDQN, CTDE/MADRL |
| 3 | `note2-pinn-pidl-cyber-control` | PINN/PIDL inverse learning and neural optimal control |

Companion repository: https://github.com/LYang910920/network-control-differential-games

## Research-Grade Checklist

Before treating a run as evidence rather than a teaching demo:

1. Run multiple random seeds and report spread.
2. Add held-out time windows or held-out trajectories.
3. Compare against classical solvers, no-control baselines, and simple rule-based policies.
4. Test identifiability of learned parameters or missing mechanisms.
5. Track constraint violations and residual terms separately.
6. Report data noise, observation frequency, and collocation sampling strategy.
7. Keep dataset, dependency, and third-party license notes with the experiment.
