# Pontryagin Neural Operator: Full-Text Evidence

## 1. Bibliographic record

- Citekey: `zhang2024pno`
- Title: *Pontryagin Neural Operator for Solving General-Sum Differential Games with Parametric State Constraints*
- Authors: Lei Zhang, Mukesh Ghimire, Zhe Xu, Wenlong Zhang and Yi Ren
- Year and venue: 2024, Learning for Dynamics and Control, PMLR 242:1728-1740
- Publisher and official URL: PMLR, <https://proceedings.mlr.press/v242/zhang24f.html>
- DOI: none reported by PMLR
- Peer-review status: peer-reviewed conference proceedings
- Final/preprint relationship: the reviewed file is the official proceedings version

## 2. Retrieval and text status

- Full-text source: official PMLR PDF
- Access basis: open full text
- Local PDF filename: `zhang2024_pontryagin_neural_operator.pdf`
- SHA-256: `9a5a4c55fafa4f868a592847b818e3f9f09736663df06e5cd6bafc8a3e0d0cc0`
- Page count: 13
- Text extraction method: native PDF text; no OCR
- Figures/tables inspected manually: Algorithm 1, Figures 1-5 and Tables 1-3

## 3. Repository question

- Question: which full-text result supports an advanced route from a
  PMP-informed residual model to parameter-conditioned feedback learning?
- Target repository: `note2-pinn-pidl-cyber-control`
- Target section/file/API: the PMP-informed PINN discussion in
  `docs/source/note2_pinn_pidl_cyber_control.tex`

## 4. Research problem and contribution

The paper addresses parameterized two-player general-sum differential games
with state penalties. It combines a DeepONet value representation with HJI,
terminal, value-trajectory and costate-trajectory consistency losses. The
costate losses use forward/backward characteristic rollouts rather than an
offline supervised boundary-value data set.

Evidence: pp. 1-2 and 4-6; Equations (5)-(6); Figure 1.

## 5. Mathematical model

- State: joint state `x=(x_1,x_2)` with `dot{x_i}=f_i(x_i,u_i)`; the vehicle
  experiment uses `(d_1,v_1,d_2,v_2)` plus time.
- Parameters: player-specific state-penalty zones `theta=(theta_1,theta_2)`.
- Controls: continuous compact convex control sets; accelerations in the
  intersection experiment.
- Objective/payoff: each player minimizes a running-plus-terminal cost subject
  to Nash inequalities.
- Constraints/invariants: differentiable state penalties represent collision
  zones; no compartment-mass invariant is involved.
- Semantics: continuous deterministic complete-information game, not sampled
  or impulsive control.

Evidence: pp. 3-4, Equations (1)-(4), and pp. 6-7, Equations (7)-(9).

## 6. Solution or learning method

PNO approximates parameterized values and costates, then derives feedback
policies through the Hamiltonian. Algorithm 1 pretrains terminal conditions,
samples HJI and trajectory points, integrates states forward and costates and
values backward, updates the networks, expands the time window, and resamples
high-residual states. The paper provides empirical evidence, not a convergence,
optimality or safety theorem for the learned operator.

Evidence: pp. 5-7, Equation (6), Algorithm 1 and the accompanying remarks.

## 7. Neural architecture, when applicable

- Inputs: a branch encoding of the parameterized penalty field and a trunk
  input `(x,t)`.
- Encoder/decoder: DeepONet branch coefficients and trunk basis values form a
  scalar value by inner product; a separate network predicts costates.
- Depth/width: three hidden layers of 64 units, tanh with adaptive activations.
- Normalization: inputs scaled to `[-1,1]`.
- Missing details: branch/trunk basis dimension, exact loss weights and total
  parameter count are not reported.

Evidence: p. 4, Equation (5); p. 5, Figure 1; pp. 8-9.

## 8. Training details, when applicable

- Optimizer: Adam; initial learning rate `2e-5`, with an incompletely specified
  adaptive schedule.
- Boundary pretraining: 50,000 iterations.
- Main training: 300 epochs, 3,000 gradient steps per epoch; trajectories are
  resampled every ten epochs.
- Samples: 1,000 PNO trajectories, about 62,000 trajectory points and 60,000
  trunk states.
- Hardware/runtime: A100 40 GB; reported PNO training time 27 hours.
- Seeds, minibatch size and numerical loss weights are not reported.

Evidence: pp. 5-9, Equation (6), Algorithm 1.

## 9. Experiments

The experiment is a two-vehicle uncontrolled-intersection game over 25
player-type pairs. Training uses four corner parameter configurations and
evaluation covers all 25. The main comparison is a supervised hybrid neural
operator based on boundary-value trajectories. Collision percentage is
reported over 600 trajectories per parameter pair, without multi-seed
uncertainty. PNO reports lower collision rates in most configurations; tanh,
sin and ReLU are also compared.

Evidence: pp. 6-10; Figures 2-5; Tables 1-3.

## 10. Limitations and failure modes

The paper states that the sampling-performance relation lacks a PAC analysis
and leaves basis sparsity and sampling complexity for future work. The evidence
comes from one low-dimensional vehicle game, with no cyber model, model
mismatch, observation noise or multi-seed uncertainty. Reproduction also lacks
loss weights, seeds, branch/trunk dimensions and a complete learning-rate
schedule.

Evidence: p. 6 remarks and footnote; pp. 9-10.

## 11. Code/data availability

- Code URL: <https://github.com/dayanbatuofu/PNO>
- Data URL and license: not stated in the paper
- Reproduction status: code and experiments were not run in this review

## 12. Transfer assessment

- Directly reusable: the distinction between an ordinary adjoint residual and
  value/costate trajectory-consistency losses.
- Adaptable: parameter-conditioned value and costate networks for a newly
  specified cyber feedback game.
- Context only: the 3-by-64 architecture, vehicle data volume and runtime.
- Incompatible: the current `cyberpinn.pmp` example is time-only,
  single-player and open-loop; it is not a PNO implementation.
- Required future evidence: held-out parameters and initial states, independent
  rollouts, HJI/costate residuals, bounded actions, unilateral deviations,
  multi-seed uncertainty and loss/sampling ablations.

## 13. Decision

- Score: 16/21 (fit 2, technical evidence 3, transferability 2, evidence quality
  3, reproducibility 2, evaluation quality 2, novel information 2)
- Decision: integrate, documentation only
- Target change: define PNO as an advanced feedback-learning direction and
  distinguish it from the maintained open-loop PMP-informed example
- Allowed claim: the paper proposes a DeepONet-based operator with HJI and
  self-supervised forward/backward value-costate consistency losses
- Claims that must not be made: this repository reproduces PNO; PNO proves
  convergence or safety; intersection performance transfers to cyber defense
- Reviewer and date: Codex full-text review, 2026-07-13

## 14. Evidence ledger

| Repository claim or design decision | Paper evidence | Location (page / Eq. / Alg. / Fig. / Table) | Confidence |
|---|---|---|---|
| PNO is a feedback-learning route beyond the current open-loop PMP example | Parameterized HJI values and feedback through value gradients/costates | pp. 4-5, Eqs. (3)-(6) | high |
| Costate consistency is more than an adjoint residual | Forward/backward value and costate losses | p. 5, Eq. (6), Fig. 1 | high |
| Parameter conditioning uses a neural operator | Branch coefficients and state-time trunk bases | p. 4, Eq. (5) | high |
| A code transfer needs a new method and experiment | Two-player feedback game, HJI losses and characteristic rollouts | pp. 4-8, Alg. 1 | high |
| Claims remain empirical | No PAC proof, no uncertainty analysis and one vehicle benchmark | pp. 6, 9-10 | high |
