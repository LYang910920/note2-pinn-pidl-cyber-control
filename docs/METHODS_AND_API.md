# Methods and API

## Common PINN Structure

Let `x_theta(t)` be the neural state and `f` the declared ODE. Automatic
differentiation gives the residual

```text
r_theta(t) = d x_theta(t) / dt - f(x_theta(t), u(t), p).
```

The objective combines only terms that are defined for the task:

```text
L = w_data * L_data + w_ode * L_ode + w_ic * L_initial
    + w_constraint * L_constraint + w_reg * L_regularization.
```

Observed data points constrain measured states. Collocation points constrain the
equation and need not coincide with observations. Loss components are logged
separately because a low total can hide a poorly fitted term.

## Method Boundaries

| Method | Unknowns | Main evidence |
|---|---|---|
| Forward PINN | state trajectory | agreement with an independent ODE solver |
| Inverse PINN | state and selected parameters | held-out state error, parameter error, residual |
| PIDL | state and missing RHS correction | ablation against known-only and fully learned alternatives |
| Neural control | bounded control and state | independent rollout objective and constraint checks |
| PMP-informed PINN | state, costate, control | state/costate/stationarity/boundary residuals |
| Node-SIPS inverse | node states and heterogeneous rates | held-out nodes/times, effective-rate and mass errors |

The Foundation repository owns the ODE, SIPS residual, integration, projection,
and shared neural components. This repository owns method-specific losses,
configurations, training, and evaluation.

## Heterogeneous Node-SIPS Inverse Problem

Each node has `[S, I, P]`, with `S + I + P = 1`. The force of infection is

```text
lambda[i] = susceptibility[i]
            * sum_j A[i,j] * infectivity[j] * I[j].
```

Truth uses community-correlated physical rates, not only heterogeneous initial
states. The inverse model estimates a small identifiable parameterization:
community-specific rates or positive feature-conditioned maps. The product of
susceptibility and infectivity has a scaling gauge, so evaluation includes the
effective edgewise transmission matrix and uses a documented normalization.

`cyberpinn.node_problem` generates truth with the canonical Foundation RHS.
`cyberpinn.node_inverse` trains dense or factorized node-time models and evaluates
held-out nodes, times, and unseen graph size.

## Typed Settings

All settings are dataclasses in `cyberpinn.configs`.
The documented medium study is assembled in `cyberpinn.experiments`; model,
residual, and optimization code remains in the method modules above.

| Config | Important defaults |
|---|---|
| `InverseConfig` | 5,000 iterations, 30 data, 200 collocation, width 64, depth 4 |
| `PIDLConfig` | 5,000 iterations, correction regularization `1e-3` |
| `ControlConfig` | horizon 20, 200 collocation, bounded `u <= 1` |
| `PMPConfig` | state, costate, stationarity, and boundary loss weights |
| `NodeSIPSDataConfig` | 8 nodes, 2 communities, 61 times, 4 observed nodes |
| `NodeInverseTrainConfig` | 500 iterations, dense/factorized architecture, strength 0.35 |

The public medium profile uses shorter bounded settings and records each resolved
configuration, seed, device, runtime, architecture, and evaluation mask.

## Interpretation Terms

- **Residual:** mismatch between the neural derivative and declared equation at
  collocation points.
- **Rollout:** an independent numerical integration using learned parameters or
  control, rather than another network evaluation.
- **Held-out time/node:** a point excluded from the data loss and used for
  evaluation.
- **Homogeneous misspecification:** a scalar-rate model applied to heterogeneous
  truth; it measures model bias, not a competing estimate of every local rate.
- **Identifiability:** whether the available observations and model structure can
  uniquely determine the requested parameters.
- **Unknown-count proxy:** the number of free trainable outputs or parameters used
  only as a complexity diagnostic; it is not an identifiability proof.

Training loss is an optimization diagnostic. Parameter truth requires synthetic
ground truth or independently validated data, and residual consistency alone does
not resolve non-identifiability.
