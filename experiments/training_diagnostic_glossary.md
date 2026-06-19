# Note 2 Training Diagnostic Glossary

Use this page while reading the training-diagnostic figures and CSV histories.

| Term | Meaning | How to read it |
|---|---|---|
| `iteration` | One optimizer, FBSM, or residual-update step. | Use for solver or neural-training progress on the x-axis. |
| `rollout` | A forward simulation under a fixed policy, control, or parameter set. | Use for validation outside the training loss. |
| `loss` | The scalar objective minimized by an optimizer. | Must be read with its component losses. |
| `data loss` | Mismatch between the neural prediction and observed data. | Shows data fit; it can improve while dynamics get worse if residual terms are weak. |
| `ODE residual loss` | Mismatch between the neural time derivative and the ODE right-hand side. | A PINN/PIDL equation-consistency check at collocation points. |
| `residual loss` | Mismatch between a neural derivative and the model right-hand side. | Small values indicate equation consistency, not necessarily optimality. |
| `initial-condition loss` | Mismatch between the neural state and the known initial state. | Anchors the trajectory at t=0. |
| `boundary loss` | Mismatch at required initial or terminal boundary conditions. | Important for PMP-informed models with terminal costate conditions. |
| `costate loss` | Mismatch in the PMP costate differential equation. | Read with state and stationarity losses; one low residual alone is not enough. |
| `stationarity loss` | Hamiltonian first-order condition residual, such as H_u. | Evidence for PMP consistency in interior-control regions. |
| `objective` | The control or learning target being minimized, such as infected burden plus control cost. | Lower is better only within the same model and metric. |
| `rollout objective` | Objective recomputed after simulating the original dynamics under a learned or fixed control. | Validation outside the training residual. |
| `correction regularizer` | Penalty that keeps a learned PIDL correction term small or smooth. | Prevents the correction network from replacing the known mechanism. |
| `mean control` | Average control intensity over the training or validation time grid. | Useful for checking whether an objective is won by excessive intervention. |
| `collocation point` | A time or state-time point where a residual is enforced without requiring observed data. | Controls where PINN/PIDL physics is checked. |
| `held-out metric` | Error on data, times, trajectories, or graph seeds not used in fitting. | A generalization check rather than a training loss. |
| `baseline comparison` | Same-model comparison with no-control, fixed, random, or simple learned policies. | Use before making a stronger method claim. |
