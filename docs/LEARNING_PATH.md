# Cross-Repository Learning Path

These repositories are meant to work together as a small learning sequence.

## Recommended Order

| Step | Repository | Focus |
|---|---|---|
| 1 | `network-control-differential-games` | PMP, FBSM, degree-level and node-level network optimal control, differential games |
| 2 | `note1-cyber-control-games` | ODE-to-RL conversion, DDQN defense, CTDE/MADRL attacker-defender learning |
| 3 | `note2-pinn-pidl-cyber-control` | PINN/PIDL inverse learning, neural control, PMP-informed neural residuals |

## How Note 2 Connects To The Differential-Games Repository

The differential-games repository gives the classical optimal-control and network-game foundation:

https://github.com/LYang910920/network-control-differential-games

This Note 2 repository asks how neural residual methods can learn dynamics, controls, and optimality systems when data are sparse or mechanisms are partially unknown.

| Differential-games concept | Note 2 counterpart |
|---|---|
| State equation | PINN ODE residual |
| Unknown or calibrated parameters | inverse PINN parameter learning |
| Missing propagation mechanism | PIDL correction network |
| Hamiltonian and stationarity | PMP-informed PINN residuals |
| Degree/node-level network states | candidate extension targets in `docs/EXTENDING.md` |

## How Note 2 Connects To Note 1

Note 1 is useful when control is learned through sampled interaction.  Note 2 is useful when the model itself, parameters, or optimality residuals should be learned from data.

| If Note 1 gives you... | Note 2 can help with... |
|---|---|
| A simulator with uncertain parameters | inverse PINN calibration |
| A reward-driven neural policy | direct control PINN comparison |
| A PMP/FBSM baseline | PMP-informed neural residual training |
| A hybrid cyber environment | future PINN/PIDL extensions with jump losses |
