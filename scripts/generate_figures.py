"""Generate static figures for the Note 2 README and tutorial notes.

Copyright (c) 2026 Luxing Yang.
Licensed under the MIT License. See LICENSE in the repository root.

The figures show the sparse-data setup, the synthetic missing mechanism, and
the neural architectures.  Training diagnostics live in
`scripts/run_training_iterations.py`.
"""

from functools import partial
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from shared_setup import ensure_foundation_package

ensure_foundation_package()
from cybercontrol.plotting import add_arrow, add_box
from inverse_pinn_sir_malware import generate_data
from pidl_unknown_mechanism import generate

diagram_box = partial(add_box, width=1.9, height=0.58)


def plot_sparse_inverse_data(output_dir: Path) -> None:
    t, x = generate_data(n_grid=240)
    idx = torch.linspace(0, len(t) - 1, 24).long()

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(t[:, 0], x[:, 0], label="Susceptible")
    ax.plot(t[:, 0], x[:, 1], label="Compromised")
    ax.plot(t[:, 0], x[:, 2], label="Recovered")
    ax.scatter(t[idx, 0], x[idx, 1], s=22, color="black", label="Sparse I(t) observations")
    ax.set_title("Inverse PINN setup: sparse observations with hidden states")
    ax.set_xlabel("Time")
    ax.set_ylabel("Population share")
    ax.legend(loc="best")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "inverse_pinn_sparse_data.png", dpi=180)
    plt.close(fig)


def plot_pidl_missing_mechanism(output_dir: Path) -> None:
    t, x = generate(n=240)
    q = 1.2
    correction = q * x[:, 0] * x[:, 1] * x[:, 1]

    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    axes[0].plot(t[:, 0], x[:, 0], label="Susceptible")
    axes[0].plot(t[:, 0], x[:, 1], label="Compromised")
    axes[0].plot(t[:, 0], x[:, 2], label="Recovered")
    axes[0].set_ylabel("Population share")
    axes[0].set_title("PIDL synthetic system with a missing nonlinear mechanism")
    axes[0].legend(loc="best")
    axes[0].grid(alpha=0.25)

    axes[1].plot(t[:, 0], correction, color="black", label="q S I^2")
    axes[1].set_xlabel("Time")
    axes[1].set_ylabel("Missing term")
    axes[1].legend(loc="best")
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "pidl_missing_mechanism.png", dpi=180)
    plt.close(fig)


def plot_neural_architectures(output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))

    ax = axes[0]
    diagram_box(ax, (0.1, 2.75), "time t", fc="#e8f1ff")
    diagram_box(ax, (2.1, 2.75), "state network\nx_theta(t)", fc="#fff4df")
    diagram_box(ax, (4.5, 2.75), "S(t), I(t), R(t)\nsoftmax simplex", width=2.2, fc="#e9f7ef")
    diagram_box(ax, (7.1, 3.25), "data loss\nobserved I(t)", width=1.9, fc="#f4ecff")
    diagram_box(ax, (7.1, 2.15), "ODE residual\nx' - f(x)", width=2.0, fc="#ffecec")
    add_arrow(ax, (2.0, 3.04), (2.1, 3.04))
    add_arrow(ax, (4.0, 3.04), (4.5, 3.04))
    add_arrow(ax, (6.6, 3.04), (7.1, 3.54))
    add_arrow(ax, (6.6, 3.04), (7.1, 2.44))
    ax.set_title("Inverse PINN / PIDL state learner")
    ax.set_xlim(0, 9.7)
    ax.set_ylim(1.2, 4.2)
    ax.axis("off")

    ax = axes[1]
    diagram_box(ax, (0.1, 3.25), "time t", fc="#e8f1ff")
    diagram_box(ax, (2.0, 3.25), "state net\nx_theta(t)", fc="#fff4df")
    diagram_box(ax, (2.0, 2.25), "costate net\nlambda_psi(t)", fc="#fff4df")
    diagram_box(ax, (2.0, 1.25), "control net\nu_phi(t)", fc="#fff4df")
    diagram_box(ax, (4.6, 2.25), "Hamiltonian\nH(x,u,lambda)", width=2.1, fc="#e9f7ef")
    diagram_box(ax, (7.2, 3.1), "state residual", fc="#ffecec")
    diagram_box(ax, (7.2, 2.25), "costate residual", fc="#ffecec")
    diagram_box(ax, (7.2, 1.4), "stationarity\nH_u = 0", fc="#ffecec")
    add_arrow(ax, (1.9, 3.54), (2.0, 3.54))
    add_arrow(ax, (3.9, 3.54), (4.6, 2.85))
    add_arrow(ax, (3.9, 2.54), (4.6, 2.54))
    add_arrow(ax, (3.9, 1.54), (4.6, 2.25))
    add_arrow(ax, (6.7, 2.54), (7.2, 3.39))
    add_arrow(ax, (6.7, 2.54), (7.2, 2.54))
    add_arrow(ax, (6.7, 2.54), (7.2, 1.69))
    ax.set_title("PMP-informed control PINN")
    ax.set_xlim(0, 9.5)
    ax.set_ylim(0.7, 4.2)
    ax.axis("off")

    fig.tight_layout()
    fig.savefig(output_dir / "neural_architectures.png", dpi=180)
    plt.close(fig)


def main() -> None:
    output_dir = ROOT / "figures"
    output_dir.mkdir(exist_ok=True)
    plot_sparse_inverse_data(output_dir)
    plot_pidl_missing_mechanism(output_dir)
    plot_neural_architectures(output_dir)
    print(f"Wrote figures to {output_dir}")


if __name__ == "__main__":
    main()
