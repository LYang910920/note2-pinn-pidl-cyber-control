# Training Summary

These diagnostics use longer laptop-friendly runs than the smoke tests.  They are intended to show whether each loss is moving toward a stable low-error regime.

| Diagnostic | Start | End | End/start |
|---|---:|---:|---:|
| Inverse PINN total loss | 1.286e+00 | 1.255e-03 | 9.756e-04 |
| PIDL total loss | 2.543e+00 | 3.564e-03 | 1.401e-03 |
| Direct control PINN total loss | 9.360e+01 | 2.636e-01 | 2.816e-03 |
| PMP-informed stationarity loss | 2.187e-01 | 3.364e-03 | 1.539e-02 |

The PMP-informed total loss can decrease more slowly because the costate boundary term and Hamiltonian residuals compete early in training.  In this teaching run, the stationarity residual is the most important quick sanity signal.
