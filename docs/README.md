# Note 2 Reading Path

Start with `note2_pinn_pidl_cyber_control.pdf`.  It explains how PINN/PIDL methods are used for cyber propagation, sparse-data learning, and neural optimal control.

Recommended order:

1. Read the PINN motivation and notation sections first.
2. Review the cyber propagation models before the inverse-learning examples.
3. Read the inverse PINN section to understand sparse observations and parameter learning.
4. Read the PIDL section to see how known mechanisms and learned corrections are combined.
5. Read the direct-control PINN before the PMP-informed PINN.
6. Use `implementation_companion.pdf` when mapping losses and residuals to code.
7. Use `code_run_guide.pdf` for run commands and troubleshooting.
8. Use `EXTENDING.md` before adapting the examples to larger graph or cyber-security models.

Source files are in `latex/`.  The source is included for inspection and adaptation; the checked-in PDF is the version intended for reading.

For a practical first pass, use `START_HERE.md` before diving into the LaTeX source.  It points to the code, scripts, figures, and experiment outputs by task.

The runnable code mirrors this tutorial order:

| Tutorial idea | Code |
|---|---|
| Sparse-data inverse PINN | `src/inverse_pinn_sir_malware.py` |
| PIDL with known dynamics plus missing mechanism | `src/pidl_unknown_mechanism.py` |
| Direct neural-control PINN | `src/control_pinn_malware.py` |
| PMP-informed state/costate/control PINN | `src/pmp_informed_pinn_malware.py` |

For extension and cross-repository guidance, read `EXTENDING.md`.
