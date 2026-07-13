"""Generate static figures for the Note 2 README and guide notes.

Copyright (c) 2026 Luxing Yang.
Licensed under the MIT License. See LICENSE in the repository root.

The figures show the sparse-data setup and the synthetic missing mechanism.
Training metrics are written by ``python -m cyberpinn medium``.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

ROOT = Path(__file__).resolve().parents[1]

from cybercontrol.plotting import (
    panel_label,
    publication_style,
    save_publication_figure,
    style_axis,
)
from cyberpinn.inverse import generate_data
from cyberpinn.pidl import generate


def plot_sparse_inverse_data(output_dir: Path) -> None:
    t, x = generate_data(n_grid=240)
    idx = torch.linspace(0, len(t) - 1, 24).long()

    with publication_style():
        fig, ax = plt.subplots(figsize=(7.16, 3.6))
    ax.plot(t[:, 0], x[:, 0], label="Susceptible", linestyle="-")
    ax.plot(t[:, 0], x[:, 1], label="Compromised", linestyle="--")
    ax.plot(t[:, 0], x[:, 2], label="Recovered", linestyle="-.")
    ax.scatter(t[idx, 0], x[idx, 1], s=22, color="black", label="Sparse I(t) observations")
    panel_label(ax, "Sparse inverse-PINN observations")
    style_axis(ax, xlabel="Time", ylabel="Population share", legend=True)
    fig.tight_layout()
    save_publication_figure(
        fig,
        output_dir / "inverse_pinn_sparse_data",
        metadata={
            "model": "SIR malware inverse PINN",
            "data": "sparse infected-state observations",
            "caption_hint": "Sparse observed I(t) with hidden S(t) and R(t).",
        },
    )
    plt.close(fig)


def plot_pidl_missing_mechanism(output_dir: Path) -> None:
    t, x = generate(n=240)
    q = 1.2
    correction = q * x[:, 0] * x[:, 1] * x[:, 1]

    with publication_style():
        fig, axes = plt.subplots(2, 1, figsize=(7.16, 4.6), sharex=True)
    axes[0].plot(t[:, 0], x[:, 0], label="Susceptible", linestyle="-")
    axes[0].plot(t[:, 0], x[:, 1], label="Compromised", linestyle="--")
    axes[0].plot(t[:, 0], x[:, 2], label="Recovered", linestyle="-.")
    panel_label(axes[0], "(a) PIDL synthetic state")
    style_axis(axes[0], ylabel="Population share", legend=True)

    axes[1].plot(t[:, 0], correction, color="black", label=r"$qSI^2$")
    panel_label(axes[1], r"(b) unknown RHS correction $qSI^2$")
    style_axis(axes[1], xlabel="Time", ylabel="RHS correction magnitude", legend=True)
    fig.tight_layout()
    save_publication_figure(
        fig,
        output_dir / "pidl_missing_mechanism",
        metadata={
            "model": "PIDL missing-mechanism example",
            "unknown_term": "q S I^2",
            "caption_hint": "Known SIR physics plus a synthetic missing RHS correction q S I^2.",
        },
    )
    plt.close(fig)


def main() -> None:
    output_dir = ROOT / "docs" / "assets"
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_sparse_inverse_data(output_dir)
    plot_pidl_missing_mechanism(output_dir)
    print(f"Wrote figures to {output_dir}")


if __name__ == "__main__":
    main()
