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
    axes[0].set_title("Inverse PINN")

    axes[1].semilogy([r["iteration"] for r in histories["pidl"]], [r["loss"] for r in histories["pidl"]], label="total")
    axes[1].plot([r["iteration"] for r in histories["pidl"]], [r["mean_correction"] for r in histories["pidl"]], label="mean correction")
    axes[1].set_title("PIDL missing mechanism")

    axes[2].semilogy([r["iteration"] for r in histories["control"]], [r["loss"] for r in histories["control"]], label="total")
    axes[2].plot([r["iteration"] for r in histories["control"]], [r["mean_control"] for r in histories["control"]], label="mean control")
    axes[2].set_title("Direct control PINN")

    axes[3].semilogy([r["iteration"] for r in histories["pmp"]], [r["loss"] for r in histories["pmp"]], label="total")
    axes[3].semilogy([r["iteration"] for r in histories["pmp"]], [r["stationarity_loss"] for r in histories["pmp"]], label="stationarity")
    axes[3].set_title("PMP-informed PINN")

    for ax in axes:
        ax.set_xlabel("Iteration")
        ax.grid(alpha=0.25)
        ax.legend()

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
    write_summary(exp_dir / "training_summary.md", histories)
    plot_training_diagnostics(fig_dir / "training_iteration_diagnostics.png", histories)

    print(f"Wrote experiment CSV files to {exp_dir}")
    print(f"Wrote training diagnostic figure to {fig_dir / 'training_iteration_diagnostics.png'}")


if __name__ == "__main__":
    main()
