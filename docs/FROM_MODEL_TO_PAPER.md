# From Inverse Problem to Paper

## 1. Define the inference target

State which compartments are observed, at which nodes and times, with what noise
model. Separate unknown states, physical parameters, observation parameters and
missing mechanisms. Explain why the chosen parameters are identifiable from the
available measurements or state the ambiguity explicitly.

## 2. Generate and verify a synthetic benchmark

Use an independent ODE solver and save the full truth before sampling sparse
observations. Check mass, nonnegativity and adjacency orientation. Reserve
held-out times, nodes, trajectories or graphs before training.

## 3. Establish misspecification baselines

Include a matched homogeneous model when truth is heterogeneous, a classical
parameter fit when feasible, and an architecture-matched neural baseline. For
PIDL, compare known physics only, correction only and the combined model. For
control, compare FBSM or direct optimization and no/constant control.

## 4. Train with interpretable losses

Log each data, residual, IC/BC, constraint, regularization, stationarity and
terminal component. Record sampling counts and optimizer updates. Use positive
parameter maps, state-simplex outputs and declared gauges where the equations
have scale symmetries.

## 5. Validate independently

Run the learned parameter/control output through the original simulator. Report
held-out state error, effective-rate error, residual maps, mass error and control
objective. A small PINN residual does not prove parameter truth. A small
Hamiltonian residual does not prove global optimality.

## 6. Test noise, sparsity and generalization

Use at least three seeds, multiple noise levels, observation densities, held-out
times and unobserved nodes. For graph models add a different graph seed/family
and unseen size. Report failures and sensitivity to loss weights.

## 7. Write bounded conclusions

Distinguish state reconstruction, parameter recovery, mechanism discovery and
control quality. Report when states are accurate but individual rates are not.
Do not transfer synthetic identifiability to operational data without a validated
measurement model.

![Research evidence pipeline](assets/diagrams/model_to_paper.png)

The literature evidence notes under `docs/literature/` explain why independent
rollout, adjoint/FBSM comparison, geometry-aware encoders and separate
verification are included. Abstract-only papers are not used for unverified
layer dimensions or benchmark claims.
