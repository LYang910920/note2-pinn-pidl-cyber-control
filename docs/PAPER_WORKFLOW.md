# How To Write A Paper From Note 2

Note 2 connects cyber ODE models to inverse PINNs, PIDL, direct neural control,
PMP-informed PINNs, and a small node-level SIPRS graph inverse-learning bridge.

## Model-To-Code Map

| Paper notation | Code location | Meaning |
| --- | --- | --- |
| `x=[S,I,R]` | `StateNet` wrappers in `src/*.py` | State trajectory on the simplex. |
| `beta`, `gamma` | `beta_raw`, `gamma_raw`, `positive(...)` | Positive learned or fixed propagation parameters. |
| `f(x)` | `cybercontrol.models.sir_rhs_torch` | Known SIR dynamics. |
| `f(x,u)` | `cybercontrol.models.controlled_sir_rhs_torch` | Controlled SIR dynamics. |
| `g_phi(x)` | `CorrectionNet` | Learned PIDL correction term. |
| `u_phi(t)` | `ControlNet` | Bounded continuous control network. |
| `lambda_psi(t)` | `costate = MLP(...)` | PMP-informed costate network. |
| `d/dt` | `cybercontrol.torch_utils.time_derivative` | Autograd time derivative for residual losses. |
| `x_i=[S_i,I_i,P_i,R_i]` | `src/node_siprs_inverse_pinn.py` | Node-level SIPRS state on a graph. |
| `\widetilde A_{ij}` | `toy_adjacency`, `node_siprs_rhs_torch` | Node `j` contributes infection pressure to node `i`. |

## Recommended Paper Path

1. Define the cyber ODE and available observations.
2. Start with inverse PINN if the question is hidden-state or parameter estimation.
3. Use PIDL if some cyber mechanism is known and only a correction should be learned.
4. Use direct neural control if the goal is to search over a control network.
5. Use PMP-informed PINN when explicit Hamiltonian/PMP structure matters.
6. Use node-SIPRS inverse PINN when the question is sparse graph/node observation.
7. Compare each method against baselines from the same problem, not unrelated tasks.

## Minimum Experiments

| Figure/table | What it should show |
| --- | --- |
| Sparse-data setup | Which state is observed, sampling density, and hidden states. |
| Loss convergence | Data, residual, boundary, objective, and stationarity losses separately. |
| State fit | True vs learned `S`, `I`, `R` where synthetic ground truth is available. |
| Parameter/error table | Parameter L1 error, state MSE, infected-state MSE, and residual error. |
| Control comparison | No control, fixed controls, direct PINN control, PMP-informed PINN, and random controls. |
| Graph inverse-learning table | observed-node/time sparsity, held-out state MSE, beta/gamma error, residual error, mass error. |
| Ablation | Remove data, residual, boundary, known-physics, or PMP terms one at a time. |

## Claim Discipline

- Say "estimate" for inverse PINN results unless identifiability and multiple-seed checks are added.
- Say "known-mechanism plus learned correction" for PIDL rather than black-box discovery.
- For `node_siprs_inverse_pinn.py`, say "small graph inverse-learning bridge"; add graph encoders, multiple graph seeds, and held-out graph sizes before claiming graph generalization.
- The PMP-informed example currently uses an interior stationarity residual. Add projected/KKT residuals before claiming bounded-control optimality at active bounds.
- Report time horizon, collocation count, observed-data count, noise level, network width/depth, learning rate, loss weights, seed count, and optimizer iterations.

For a broader cross-method workflow, see the foundation repository's
`docs/from_model_to_paper.md`.
