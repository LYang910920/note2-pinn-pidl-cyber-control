# Physics-Informed Control Evidence Notes

## Full-text-reviewed evidence

- The Pontryagin neural operator uses costate consistency to learn families of
  constrained games. It is an advanced feedback-learning direction, distinct
  from the foundation's open-loop FBSM solver.

## Metadata or preview-level leads

- Mowlavi and Nabi and the physics-informed PointNet paper remain in
  `PDF_REQUESTS.md`. Their official metadata motivates independent rollout and
  geometry-aware extensions, but no unverified architecture or benchmark detail
  is used here.
- The official PINC abstract describes conditioning on initial state and control
  over shorter intervals and chaining interval predictions. Note 2 lists this as
  an extension route; the full article remains requested and the current code
  makes no long-horizon PINC performance claim.
- LyZNet with Control is available but has not yet received a full-text evidence
  pass. Its title and abstract motivate separating residual training from formal
  verification; they do not make the current neural controller certified.

## Claim boundary

The node-SIPS inverse experiment is synthetic. Held-out-time, node and graph-size
errors do not identify real cyber rates without a justified observation and
measurement model.
