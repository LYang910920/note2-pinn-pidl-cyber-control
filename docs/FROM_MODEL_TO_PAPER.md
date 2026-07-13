# From Model to Paper

## 1. Define the Scientific Question

Choose forward prediction, inverse estimation, missing-mechanism learning, control,
or optimality-system approximation. State the requested outputs and which are
observable or identifiable.

## 2. Validate the Mechanistic Model

Write the state order, units, graph, parameters, controls, invariants, and boundary
conditions. Generate a reference trajectory with an independent solver and test
the NumPy/Torch equations before fitting a network.

## 3. Define Data and Residual Points

Record observed states, nodes, times, noise, and missingness. Draw collocation
points independently and document their distribution. Keep held-out nodes, times,
trajectories, or graphs out of every training loss.

## 4. Choose an Identifiable Parameterization

Do not fit one global rate pair to heterogeneous truth and interpret it as local
recovery. Use community rates or a positive feature-conditioned map, regularize it,
and report gauge or correlation ambiguities. Compare with the homogeneous model to
measure misspecification.

## 5. Train and Diagnose

Start with data and initial-condition terms, introduce residual and constraint
terms, inspect gradient/loss scales, and refine collocation points if justified.
Track every component, not just the weighted total.

## 6. Validate Independently

Use ODE rollouts, held-out observations, parameter/effective-rate errors,
conservation, and residual maps. For control, compare the learned control with FBS
or direct optimization under the same objective and bounds. For PMP-informed
methods, report state, costate, stationarity, and boundary residuals separately.

## 7. Design Ablations

Vary data density, noise, collocation count, loss weights, architecture, graph
seed/size, and heterogeneity strength. For PIDL, compare known-only, correction,
and more flexible learned-mechanism models.

## 8. State Limits

Separate interpolation, parameter recovery, mechanism discovery, and control
performance claims. Residual consistency is not parameter truth; synthetic
recovery is not field calibration; and a small configured inverse problem does not
establish identifiability for a larger model.
