"""Run longer Note 2 diagnostics and save reader-friendly artifacts.

Copyright (c) 2026 Luxing Yang.
Licensed under the MIT License. See LICENSE in the repository root.

Smoke tests answer "does every script execute?"  This script answers "do the
loss curves move in a sensible direction over time?"
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from control_pinn_malware import train as train_control_pinn
from inverse_pinn_sir_malware import train as train_inverse_pinn
from pidl_unknown_mechanism import train as train_pidl
from pmp_informed_pinn_malware import train as train_pmp_pinn


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def inverse_args(iters: int) -> SimpleNamespace:
    """Configuration for the sparse-data inverse PINN teaching run."""
    return SimpleNamespace(
        smoke=False,
        iters=iters,
        width=24,
        depth=2,
        n_data=16,
        n_collocation=70,
        noise=0.0,
        lr=1e-3,
        w_ic=10.0,
        w_ode=1.0,
        log_every=max(1, iters // 24),
        seed=21,
        return_history=True,
    )


def pidl_args(iters: int) -> SimpleNamespace:
    """Configuration for the PIDL missing-mechanism teaching run."""
    return SimpleNamespace(
        smoke=False,
        iters=iters,
        n_data=18,
        n_collocation=70,
        width=24,
        lr=1e-3,
        w_ic=10.0,
        w_res=1.0,
        w_corr=1e-3,
        log_every=max(1, iters // 24),
        seed=22,
        return_history=True,
    )


def control_args(iters: int) -> SimpleNamespace:
    """Configuration for the direct neural-control PINN teaching run."""
    return SimpleNamespace(
        smoke=False,
        iters=iters,
        T=20.0,
        n_collocation=70,
        width=24,
        lr=1e-3,
        beta=0.8,
        gamma=0.2,
        umax=1.0,
        A=10.0,
        B=1.0,
        AT=10.0,
        w_res=10.0,
        w_ic=10.0,
        log_every=max(1, iters // 24),
        seed=23,
        return_history=True,
    )


def pmp_args(iters: int) -> SimpleNamespace:
    """Configuration for the PMP-informed PINN teaching run."""
    return SimpleNamespace(
        smoke=False,
        iters=iters,
        T=20.0,
        n_collocation=70,
        width=24,
        lr=1e-3,
        beta=0.8,
        gamma=0.2,
        umax=1.0,
        A=10.0,
        B=1.0,
        AT=10.0,
        w_state=10.0,
        w_costate=1.0,
        w_stat=1.0,
        w_bc=10.0,
        log_every=max(1, iters // 24),
        seed=24,
        return_history=True,
    )


def plot_training_diagnostics(output_path: Path, histories: dict[str, list[dict]]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.ravel()

    axes[0].semilogy([r["iteration"] for r in histories["inverse"]], [r["loss"] for r in histories["inverse"]], label="total")
    axes[0].semilogy([r["iteration"] for r in histories["inverse"]], [r["ode_loss"] for r in histories["inverse"]], label="ODE")
    axes[0].set_title("Sparse-data inverse PINN: total vs ODE residual loss")
    axes[0].set_ylabel("Loss (log scale)")

    pidl_it = [r["iteration"] for r in histories["pidl"]]
    axes[1].semilogy(pidl_it, [r["loss"] for r in histories["pidl"]], label="total")
    axes[1].semilogy(pidl_it, [r["residual_loss"] for r in histories["pidl"]], label="residual")
    axes[1].set_title("PIDL missing mechanism: residual loss and correction size")
    axes[1].set_ylabel("Loss (log scale)")
    ax_pidl = axes[1].twinx()
    ax_pidl.plot(pidl_it, [r["mean_correction"] for r in histories["pidl"]], color="#e45756", label="mean correction")
    ax_pidl.set_ylabel("Mean learned correction")

    control_it = [r["iteration"] for r in histories["control"]]
    axes[2].semilogy(control_it, [r["loss"] for r in histories["control"]], label="total")
    axes[2].semilogy(control_it, [r["objective"] for r in histories["control"]], label="objective")
    axes[2].semilogy(control_it, [r["residual_loss"] for r in histories["control"]], label="state residual")
    axes[2].set_title("Direct control PINN: objective and dynamics residual")
    axes[2].set_ylabel("Loss / objective (log scale)")
    ax_control = axes[2].twinx()
    ax_control.plot(control_it, [r["mean_control"] for r in histories["control"]], color="#e45756", label="mean control")
    ax_control.set_ylabel("Mean control intensity")

    axes[3].semilogy([r["iteration"] for r in histories["pmp"]], [r["loss"] for r in histories["pmp"]], label="total")
    axes[3].semilogy([r["iteration"] for r in histories["pmp"]], [r["state_loss"] for r in histories["pmp"]], label="state")
    axes[3].semilogy([r["iteration"] for r in histories["pmp"]], [r["costate_loss"] for r in histories["pmp"]], label="costate")
    axes[3].semilogy([r["iteration"] for r in histories["pmp"]], [r["stationarity_loss"] for r in histories["pmp"]], label="stationarity")
    axes[3].set_title("PMP-informed PINN: optimality residuals")
    axes[3].set_ylabel("Residual loss (log scale)")

    for ax in axes:
        ax.set_xlabel("Iteration")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    axes[1].legend(loc="upper left", fontsize=8)
    axes[2].legend(loc="upper left", fontsize=8)
    ax_pidl.legend(loc="upper right", fontsize=8)
    ax_control.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    output_path.parent.mkdir(exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def reduction(start: float, end: float) -> float:
    return end / max(abs(start), 1e-12)


def write_summary(path: Path, histories: dict[str, list[dict]]) -> None:
    inv = histories["inverse"]
    pidl = histories["pidl"]
    control = histories["control"]
    pmp = histories["pmp"]
    text = f"""# Training Summary

These diagnostics use longer laptop-friendly runs than the smoke tests.  They are intended to show whether each loss is moving toward a stable low-error regime.

| Diagnostic | Start | End | End/start |
|---|---:|---:|---:|
| Inverse PINN total loss | {inv[0]["loss"]:.3e} | {inv[-1]["loss"]:.3e} | {reduction(inv[0]["loss"], inv[-1]["loss"]):.3e} |
| PIDL total loss | {pidl[0]["loss"]:.3e} | {pidl[-1]["loss"]:.3e} | {reduction(pidl[0]["loss"], pidl[-1]["loss"]):.3e} |
| Direct control PINN total loss | {control[0]["loss"]:.3e} | {control[-1]["loss"]:.3e} | {reduction(control[0]["loss"], control[-1]["loss"]):.3e} |
| PMP-informed stationarity loss | {pmp[0]["stationarity_loss"]:.3e} | {pmp[-1]["stationarity_loss"]:.3e} | {reduction(pmp[0]["stationarity_loss"], pmp[-1]["stationarity_loss"]):.3e} |

The PMP-informed total loss can decrease more slowly because the costate boundary term and Hamiltonian residuals compete early in training.  In this teaching run, the stationarity residual is the most important quick sanity signal.
"""
    path.write_text(text)


def write_output_preview(path: Path, histories: dict[str, list[dict]]) -> None:
    """Write a short categorized guide to the generated Note 2 outputs."""
    inv = histories["inverse"]
    pidl = histories["pidl"]
    control = histories["control"]
    pmp = histories["pmp"]
    text = f"""# Output Preview

Use this page as the first stop after running `python scripts/run_training_iterations.py`.

## 1. What Each Experiment Checks

| Experiment | Main question | Key diagnostics |
|---|---|---|
| Sparse-data inverse PINN | Can sparse infected observations recover hidden S/I/R states and beta/gamma? | total loss, data loss, ODE residual, learned beta/gamma |
| PIDL missing mechanism | Can a small correction network represent the missing nonlinear propagation term? | residual loss, correction regularizer, mean learned correction |
| Direct control PINN | Can state/control networks reduce malware while satisfying the ODE? | objective, state residual, initial-condition loss, mean control |
| PMP-informed PINN | Can state, costate, and control networks satisfy PMP optimality conditions? | state residual, costate residual, stationarity loss, boundary loss |

## 2. Training Diagnostic Panels

Open `figures/training_iteration_diagnostics.png`.

| Panel | What to check |
|---|---|
| Sparse-data inverse PINN | total loss and ODE residual should decrease together |
| PIDL missing mechanism | residual loss should fall while the correction remains interpretable |
| Direct control PINN | objective should fall without losing dynamics consistency |
| PMP-informed PINN | stationarity and state/costate residuals should move toward a low-error regime |

## 3. First-Versus-Last Snapshot

| Diagnostic | Start | End | End/start |
|---|---:|---:|---:|
| Inverse PINN total loss | {inv[0]["loss"]:.3e} | {inv[-1]["loss"]:.3e} | {reduction(inv[0]["loss"], inv[-1]["loss"]):.3e} |
| PIDL total loss | {pidl[0]["loss"]:.3e} | {pidl[-1]["loss"]:.3e} | {reduction(pidl[0]["loss"], pidl[-1]["loss"]):.3e} |
| Direct control PINN total loss | {control[0]["loss"]:.3e} | {control[-1]["loss"]:.3e} | {reduction(control[0]["loss"], control[-1]["loss"]):.3e} |
| PMP-informed stationarity loss | {pmp[0]["stationarity_loss"]:.3e} | {pmp[-1]["stationarity_loss"]:.3e} | {reduction(pmp[0]["stationarity_loss"], pmp[-1]["stationarity_loss"]):.3e} |

## 4. Files To Open First

| Category | File |
|---|---|
| Summary | `experiments/training_summary.md` |
| Learning curves | `figures/training_iteration_diagnostics.png` |
| Inverse PINN CSV | `experiments/inverse_pinn_training_history.csv` |
| PIDL CSV | `experiments/pidl_training_history.csv` |
| Direct control CSV | `experiments/control_pinn_training_history.csv` |
| PMP-informed CSV | `experiments/pmp_informed_pinn_training_history.csv` |
"""
    path.write_text(text)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run training-iteration experiments for Note 2.")
    parser.add_argument("--iters", type=int, default=600, help="Iteration count for each PINN/PIDL diagnostic.")
    args = parser.parse_args()

    exp_dir = ROOT / "experiments"
    fig_dir = ROOT / "figures"

    _, _, _, inverse_history = train_inverse_pinn(inverse_args(args.iters))
    _, _, pidl_history = train_pidl(pidl_args(args.iters))
    _, _, control_history = train_control_pinn(control_args(args.iters))
    _, _, _, pmp_history = train_pmp_pinn(pmp_args(args.iters))

    histories = {
        "inverse": inverse_history,
        "pidl": pidl_history,
        "control": control_history,
        "pmp": pmp_history,
    }
    write_csv(exp_dir / "inverse_pinn_training_history.csv", inverse_history)
    write_csv(exp_dir / "pidl_training_history.csv", pidl_history)
    write_csv(exp_dir / "control_pinn_training_history.csv", control_history)
    write_csv(exp_dir / "pmp_informed_pinn_training_history.csv", pmp_history)
    write_output_preview(exp_dir / "OUTPUT_PREVIEW.md", histories)
    write_summary(exp_dir / "training_summary.md", histories)
    plot_training_diagnostics(fig_dir / "training_iteration_diagnostics.png", histories)

    print(f"Wrote experiment CSV files to {exp_dir}")
    print(f"Wrote training diagnostic figure to {fig_dir / 'training_iteration_diagnostics.png'}")


if __name__ == "__main__":
    main()
