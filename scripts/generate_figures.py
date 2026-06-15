from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from inverse_pinn_sir_malware import generate_data
from pidl_unknown_mechanism import generate


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


def main() -> None:
    output_dir = ROOT / "figures"
    output_dir.mkdir(exist_ok=True)
    plot_sparse_inverse_data(output_dir)
    plot_pidl_missing_mechanism(output_dir)
    print(f"Wrote figures to {output_dir}")


if __name__ == "__main__":
    main()
