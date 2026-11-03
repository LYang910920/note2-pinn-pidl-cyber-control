# From sparse observations to a dynamical model

## 1. Run one inference experiment

After installation, run `python -m cyberpinn`. The default uses 24 infection observations over 20 time units and trains a small inverse PINN for 1,500 steps on a CPU. It compares three quantities: the PINN's state trajectory, an ODE trajectory generated from the PINN's estimated parameters, and an ODE least-squares fit to the same observations.

Read `src/cyberpinn/cli.py::run_example` from top to bottom. It constructs observations, trains the PINN, fits the ODE baseline, evaluates held-out times and plots. Read `inverse.py` next for the data construction, network and loss. More advanced architectures remain separate because they solve different problems; none is required for this first example.

## 2. The inverse problem

The synthetic system is

\[
\dot S=-\beta SI,\qquad
\dot I=\beta SI-\gamma I,\qquad
\dot R=\gamma I,
\]

with known initial state (0.95,0.05,0), true beta=0.8 and gamma=0.2. These are illustrative teaching rates per time unit, not fitted cyber-incident measurements. Only selected I values are observed. S and R are latent after the known initial condition; beta and gamma are estimated.

`observation_data` returns four separate objects: the complete time grid, the clean synthetic trajectory, observed indices and measured infection values. Measurement noise is generated with a local seeded generator and added only to the measured values. It does not alter the ground-truth latent states. The initial state, physical model and selected times are shared by both inference methods.

## 3. Network, derivatives and loss

`StateNet` maps physical time to three nonnegative state fractions summing to one. Its input is scaled from [0,20] to [-1,1]. Automatic differentiation is applied with respect to the original physical time, so the scale factor is included through the chain rule. Changing the horizon requires changing this scale and the collocation grid consistently.

The unknown rates use a positive transformation of unconstrained trainable parameters. The loss is

\[
\mathcal L=\mathcal L_{\rm data}
 +w_{\rm ic}\mathcal L_{\rm initial}
 +w_{\rm ode}\mathcal L_{\rm residual}.
\]

The data term compares predicted I with the sparse measurements. The initial-condition term compares all three states at t=0 with the known initial state. The ODE residual compares automatic time derivatives with the shared SIR RHS at collocation points. The simplex architecture enforces mass and nonnegativity, while the initial condition and differential equations are imposed softly through losses.

`inverse.py::train` keeps these three terms explicit. Positive parameters and simplex outputs do not themselves guarantee accurate rates or trajectories. Inspect the individual losses rather than only their weighted sum. The parameter values in the saved history are recorded after each logged optimizer update; the accompanying losses are those used for that update.

## 4. Why the ODE comparisons matter

`fit_ode` uses SciPy `least_squares` and `solve_ivp`, estimates only beta and gamma, and minimizes error at exactly the same observed I values as the PINN. Both rates are restricted to [0.01,3] per time unit for this teaching baseline. These bounds are modeling assumptions and should be justified or changed for another application.

When a low-dimensional mechanism is known, ordinary ODE fitting is an informative baseline and may be more accurate or efficient. PINNs become useful for different reasons, such as recovering latent fields, handling partial observations with physical constraints, or combining imperfect mechanisms and data. A short example should show those questions clearly rather than promise neural superiority.

A neural state curve can fit the observations while its estimated rates imply a different ODE trajectory. The **PINN rates / ODE replay** curve checks this distinction. A discrepancy suggests incomplete physical convergence, competing loss terms, model mismatch or insufficient information; it should be understood before using the estimated model for control.

## 5. Read the outputs correctly

`fit.png` shows truth, measured I, the PINN trajectory, the mechanistic fit and the parameter-replay trajectory. `learning.png` shows the separate unweighted loss components. `observations.csv` contains only the measured times and infection values; `trajectories.csv` contains all three states for each method. `comparison.csv` reports beta, gamma, error against the observed values and error against clean synthetic I at unobserved grid times. `training.csv` contains the learning history.

The unobserved grid times lie inside the fitted interval. This is temporal interpolation, not extrapolation. Physics collocation still covers the interval. For forecasting, restrict all training observations and collocation to a training interval, then test on a later interval using a clearly specified prediction procedure.

Small residuals and low interpolation error do not prove identifiability. With real observations, test sensitivity to initial guesses, noise, observation times and competing parameterizations. When several parameter combinations produce nearly the same measured trajectory, report the resulting uncertainty rather than a single rate estimate as a uniquely recovered mechanism.

## 6. Change the observations or mechanism

```bash
python -m cyberpinn --iters 3000 --observations 20 --noise 0.01 --seed 4 --output-dir results/noisy
```

More training steps change numerical effort; changing observed times, noise or the mechanism changes the inference problem. Keep observation sets fixed when comparing methods. For real data, replace `observation_data` with a loader returning measurement times, observed components and their masks. Do not substitute unobserved states with zeros and treat them as measured values.

When changing the equations, update the NumPy/Torch RHS consistently and verify their agreement on sample states. Align the state ordering, initial conditions, unknown parameter transformations, observation operator, residual and ODE baseline. Use a centered finite-difference or analytic check on derivatives when introducing a new mechanism. Rescale residual components when units or magnitudes differ substantially, and state the resulting weights.

When the initial condition is uncertain, estimate it within the physical constraints or use an appropriate observation likelihood. Do not quietly impose the synthetic initial state on a real dataset. A learned correction can absorb parameter error as well as missing physics, so interpretation requires additional assumptions and comparisons.

## 7. Extend to PIDL, control and networks

`pidl.py` adds a learned mechanism correction to a physical model. Read the correction's location, sign constraints and regularization before interpreting it as a discovered process. Compare with the uncorrected mechanism using the same observations, and assess predictive behavior away from the fitted samples.

`control.py` learns a state/control representation through an objective and physical residual. `pmp.py` additionally represents costates and Hamiltonian conditions. These solve a different problem from parameter identification. To validate a proposed control, replay it in an independent ODE solver and compare its actual objective with meaningful baselines. Low training loss alone is not a global-optimality certificate. Control bounds and terminal conditions must agree with the mathematical problem.

For node-level inference, start at `node_problem.py` and `node_inverse.py`. Each node has S, I, P states; the incoming-influence matrix couples nodes. Specify which nodes and times are observed. Temporal holdout, node holdout and joint node-time holdout test different generalization questions. The dense and factorized architectures in `architectures.py` should be compared with appropriate capacity budgets. Normalizations used to separate susceptibility and infectivity define a parameterization and do not create new information in the observations.

The broader existing preset study is optional:

```bash
python -m cyberpinn study --device cpu --output-dir results/study
```

This runs several methods and seeds; the introductory `--iters` and `--observations` flags describe the first example, not every method in that preset. Start with the single example and move to the study after its data, losses and replay curves are understood.

## 8. Shared code, tests and lecture PDF

Ordinary installation uses the exact tested foundation revision in `pyproject.toml`. To edit both repositories in one separate environment, install the foundation first with `python -m pip install -e ".[torch,dev]"`, then install this note with `python -m pip install --no-deps -e ../note2-pinn-pidl-cyber-control`. This keeps the editable foundation rather than replacing it with the Git dependency.

```bash
python -m pip install -e ".[dev]"
pytest -q
python -m cyberpinn check
python -m cyberpinn docs
```

Tests cover state mass, positive/bounded outputs, live gradients, observation handling and the student workflow. They do not require a short PINN run to outperform ODE fitting. Only the PDF build requires LaTeX/latexmk and the source checkout. Existing lecture commands `smoke`, `figures` and `medium` map to `check`, `example` and `study`.

## Research connections and constrained residuals

The lecture restores the formal cyber-control papers and explains how their mechanisms motivate inverse problems, rather than attributing PINN methods to publications that develop different algorithms. It connects APT evaluation and control, opinion/rumor games, impulsive defense, propaganda awareness, malware-opinion coupling, and feedback-loop parameter learning with MPC.

For bounded PMP training, `pmp.py` uses `u - clamp(u - H_u/B, 0, umax)`, so valid boundary minima satisfy the residual. In `pidl.py`, the additional contact-driven flux is `S * I * softplus(N(x))`; it conserves mass and vanishes at the susceptible and infected boundaries. The lecture derives these properties, and `tests/test_lecture_conditions.py` checks them and their live gradients.
