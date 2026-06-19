"""Run longer Note 2 diagnostics and save reader-friendly artifacts.

Copyright (c) 2026 Luxing Yang.
Licensed under the MIT License. See LICENSE in the repository root.

Smoke tests answer "does every script execute?"  This script answers two
heavier questions: do the loss curves move in a sensible direction, and do
trained methods beat simple, topic-specific baselines?
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import textwrap
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from shared_setup import ensure_foundation_package

ensure_foundation_package()
from cybercontrol.io import write_csv
from cybercontrol.torch_utils import rk4_step_torch
from control_pinn_malware import ControlNet, rhs as control_rhs, train as train_control_pinn
from inverse_pinn_sir_malware import generate_data as generate_inverse_data, train as train_inverse_pinn
from pidl_unknown_mechanism import generate as generate_pidl_data
from pidl_unknown_mechanism import known_rhs as pidl_known_rhs
from pidl_unknown_mechanism import train as train_pidl
from pmp_informed_pinn_malware import train as train_pmp_pinn

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BASELINE_FIELDS = [
    "topic",
    "method",
    "plot_label",
    "primary_metric",
    "primary_value",
    "state_mse",
    "infected_mse",
    "parameter_l1_error",
    "correction_mse",
    "objective",
    "cumulative_infected",
    "peak_infected",
    "final_infected",
    "mean_control",
    "notes",
]

TRAINING_PROFILES = {
    "teaching": {
        "iters": 600,
        "inverse": {"width": 24, "depth": 2, "n_data": 16, "n_collocation": 70},
        "pidl": {"width": 24, "depth": 2, "n_data": 18, "n_collocation": 70},
        "control": {"width": 24, "depth": 2, "n_collocation": 70},
        "pmp": {"width": 24, "depth": 2, "n_collocation": 70},
    },
    "gpu": {
        "iters": 2000,
        "inverse": {"width": 128, "depth": 5, "n_data": 40, "n_collocation": 400},
        "pidl": {"width": 128, "depth": 4, "n_data": 50, "n_collocation": 400},
        "control": {"width": 128, "depth": 4, "n_collocation": 400},
        "pmp": {"width": 128, "depth": 4, "n_collocation": 400},
    },
}


def resolve_training_profile(name: str, iters_override: int | None) -> dict:
    """Return one named PINN/PIDL profile with an optional iteration override."""
    source = TRAINING_PROFILES[name]
    profile = {
        "name": name,
        "iters": source["iters"],
        "inverse": dict(source["inverse"]),
        "pidl": dict(source["pidl"]),
        "control": dict(source["control"]),
        "pmp": dict(source["pmp"]),
    }
    if iters_override is not None:
        profile["iters"] = iters_override
    return profile


def baseline_row(**kwargs) -> dict:
    """Create a consistent row for baseline-comparison metrics."""
    row = {key: "" for key in BASELINE_FIELDS}
    row.update(kwargs)
    return row


def to_numpy(x) -> np.ndarray:
    """Move a tensor or array-like value to a NumPy array."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def mse(a, b) -> float:
    """Mean squared error as a plain Python float."""
    return float(np.mean((np.asarray(a) - np.asarray(b)) ** 2))


def inverse_args(profile: dict) -> SimpleNamespace:
    """Configuration for the sparse-data inverse PINN tutorial run."""
    cfg = profile["inverse"]
    return SimpleNamespace(
        smoke=False,
        iters=profile["iters"],
        width=cfg["width"],
        depth=cfg["depth"],
        n_data=cfg["n_data"],
        n_collocation=cfg["n_collocation"],
        noise=0.0,
        lr=1e-3,
        w_ic=10.0,
        w_ode=1.0,
        log_every=max(1, profile["iters"] // 24),
        seed=21,
        return_history=True,
    )


def pidl_args(profile: dict) -> SimpleNamespace:
    """Configuration for the PIDL missing-mechanism tutorial run."""
    cfg = profile["pidl"]
    return SimpleNamespace(
        smoke=False,
        iters=profile["iters"],
        n_data=cfg["n_data"],
        n_collocation=cfg["n_collocation"],
        width=cfg["width"],
        depth=cfg["depth"],
        lr=1e-3,
        w_ic=10.0,
        w_res=1.0,
        w_corr=1e-3,
        log_every=max(1, profile["iters"] // 24),
        seed=22,
        return_history=True,
    )


def control_args(profile: dict) -> SimpleNamespace:
    """Configuration for the direct neural-control PINN tutorial run."""
    cfg = profile["control"]
    return SimpleNamespace(
        smoke=False,
        iters=profile["iters"],
        T=20.0,
        n_collocation=cfg["n_collocation"],
        width=cfg["width"],
        depth=cfg["depth"],
        lr=1e-3,
        beta=0.8,
        gamma=0.2,
        umax=1.0,
        A=10.0,
        B=1.0,
        AT=10.0,
        w_res=10.0,
        w_ic=10.0,
        log_every=max(1, profile["iters"] // 24),
        seed=23,
        return_history=True,
    )


def pmp_args(profile: dict) -> SimpleNamespace:
    """Configuration for the PMP-informed PINN tutorial run."""
    cfg = profile["pmp"]
    return SimpleNamespace(
        smoke=False,
        iters=profile["iters"],
        T=20.0,
        n_collocation=cfg["n_collocation"],
        width=cfg["width"],
        depth=cfg["depth"],
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
        log_every=max(1, profile["iters"] // 24),
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


def roll_wrong_parameter_sir(beta: float, gamma: float, T: float = 20.0, n_grid: int = 400) -> np.ndarray:
    """Roll out a deliberately misspecified SIR baseline for inverse PINN comparison."""
    _, x = generate_inverse_data(beta_true=beta, gamma_true=gamma, T=T, n_grid=n_grid)
    return to_numpy(x)


def evaluate_inverse_baselines(model, beta: float, gamma: float, args: SimpleNamespace) -> list[dict]:
    """Compare the inverse PINN against simple sparse-data baselines."""
    t_all, x_all = generate_inverse_data(noise=args.noise)
    t_np = to_numpy(t_all[:, 0])
    x_true = to_numpy(x_all)
    idx = torch.linspace(0, len(t_all) - 1, args.n_data).long()
    t_sparse = to_numpy(t_all[idx, 0])
    i_sparse = to_numpy(x_all[idx, 1])

    model.eval()
    with torch.no_grad():
        x_pinn = to_numpy(model(t_all.to(DEVICE)))

    i_interp = np.interp(t_np, t_sparse, i_sparse)
    x_interp = np.column_stack([1.0 - i_interp, i_interp, np.zeros_like(i_interp)])
    x_wrong = roll_wrong_parameter_sir(beta=0.55, gamma=0.35, n_grid=len(t_np))

    rows = [
        baseline_row(
            topic="Sparse-data inverse PINN",
            method="Inverse PINN (data + ODE residual)",
            plot_label="Inverse PINN",
            primary_metric="full_state_mse",
            primary_value=mse(x_pinn, x_true),
            state_mse=mse(x_pinn, x_true),
            infected_mse=mse(x_pinn[:, 1], x_true[:, 1]),
            parameter_l1_error=abs(beta - 0.8) + abs(gamma - 0.2),
            notes="Learns hidden S/R states and beta/gamma from sparse I(t).",
        ),
        baseline_row(
            topic="Sparse-data inverse PINN",
            method="Naive sparse-data interpolation",
            plot_label="Sparse interp.",
            primary_metric="full_state_mse",
            primary_value=mse(x_interp, x_true),
            state_mse=mse(x_interp, x_true),
            infected_mse=mse(i_interp, x_true[:, 1]),
            notes="Interpolates observed I(t), then uses S=1-I and R=0.",
        ),
        baseline_row(
            topic="Sparse-data inverse PINN",
            method="Wrong-parameter SIR rollout",
            plot_label="Wrong SIR params",
            primary_metric="full_state_mse",
            primary_value=mse(x_wrong, x_true),
            state_mse=mse(x_wrong, x_true),
            infected_mse=mse(x_wrong[:, 1], x_true[:, 1]),
            parameter_l1_error=abs(0.55 - 0.8) + abs(0.35 - 0.2),
            notes="Uses the right model form but inaccurate beta/gamma.",
        ),
    ]
    return rows


def roll_known_pidl_baseline(T: float = 20.0, n_grid: int = 400, beta: float = 0.8, gamma: float = 0.2) -> np.ndarray:
    """Roll out the known SIR mechanism with the hidden PIDL correction removed."""
    dt = T / (n_grid - 1)
    x = torch.zeros(n_grid, 3)
    x[0] = torch.tensor([0.95, 0.05, 0.0])
    beta_t = torch.tensor(beta)
    gamma_t = torch.tensor(gamma)
    for k in range(n_grid - 1):
        rhs = lambda y: pidl_known_rhs(y, beta_t, gamma_t)
        x[k + 1] = rk4_step_torch(x[k], dt, rhs, project_simplex=True)
    return to_numpy(x)


def evaluate_pidl_baselines(state_net, corr_net) -> list[dict]:
    """Compare PIDL with the known-mechanism-only baseline."""
    t_all, x_all = generate_pidl_data(n=400)
    x_true = to_numpy(x_all)
    x_known = roll_known_pidl_baseline(n_grid=len(t_all))

    state_net.eval()
    corr_net.eval()
    with torch.no_grad():
        x_pidl = to_numpy(state_net(t_all.to(DEVICE)))
        correction_pred = to_numpy(corr_net(x_all.to(DEVICE)))[:, 0]
    correction_true = to_numpy(1.2 * x_all[:, 0] * x_all[:, 1] * x_all[:, 1])

    return [
        baseline_row(
            topic="PIDL missing mechanism",
            method="PIDL learned correction",
            plot_label="PIDL correction",
            primary_metric="full_state_mse",
            primary_value=mse(x_pidl, x_true),
            state_mse=mse(x_pidl, x_true),
            infected_mse=mse(x_pidl[:, 1], x_true[:, 1]),
            correction_mse=mse(correction_pred, correction_true),
            notes="Uses known SIR dynamics plus a learned nonlinear correction.",
        ),
        baseline_row(
            topic="PIDL missing mechanism",
            method="Known SIR only (no missing correction)",
            plot_label="Known SIR only",
            primary_metric="full_state_mse",
            primary_value=mse(x_known, x_true),
            state_mse=mse(x_known, x_true),
            infected_mse=mse(x_known[:, 1], x_true[:, 1]),
            notes="Removes the hidden q S I^2 term and cannot represent behavior change.",
        ),
    ]


def controlled_rhs_np(x: np.ndarray, u: float, beta: float, gamma: float) -> np.ndarray:
    """Controlled SIR malware dynamics for independent rollout evaluation."""
    S, I, R = x
    return np.array([-beta * S * I - u * S, beta * S * I - gamma * I, gamma * I + u * S], dtype=float)


def get_control_value(control_source, t: float) -> float:
    """Evaluate a constant, Python callable, or PyTorch control network at time t."""
    if isinstance(control_source, (float, int)):
        return float(control_source)
    if isinstance(control_source, torch.nn.Module):
        device = next(control_source.parameters()).device
        with torch.no_grad():
            value = control_source(torch.tensor([[t]], dtype=torch.float32, device=device)).cpu().item()
        return float(value)
    return float(control_source(t))


def rollout_control_objective(control_source, args: SimpleNamespace, n_grid: int = 400) -> dict:
    """Evaluate a control policy by rolling the original ODE forward."""
    t = np.linspace(0.0, args.T, n_grid)
    dt = args.T / (n_grid - 1)
    x = np.zeros((n_grid, 3), dtype=float)
    u = np.zeros(n_grid, dtype=float)
    x[0] = np.array([0.95, 0.05, 0.0], dtype=float)

    for k in range(n_grid - 1):
        u[k] = np.clip(get_control_value(control_source, float(t[k])), 0.0, args.umax)
        f = lambda y: controlled_rhs_np(y, u[k], args.beta, args.gamma)
        k1 = f(x[k])
        k2 = f(x[k] + 0.5 * dt * k1)
        k3 = f(x[k] + 0.5 * dt * k2)
        k4 = f(x[k] + dt * k3)
        y = x[k] + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
        y = np.clip(y, 0.0, 1.0)
        x[k + 1] = y / max(y.sum(), 1e-12)
    u[-1] = np.clip(get_control_value(control_source, float(t[-1])), 0.0, args.umax)

    running = args.A * x[:, 1] + 0.5 * args.B * u * u
    objective = float(np.trapz(running, t) + args.AT * x[-1, 1])
    return {
        "objective": objective,
        "cumulative_infected": float(np.trapz(x[:, 1], t)),
        "peak_infected": float(np.max(x[:, 1])),
        "final_infected": float(x[-1, 1]),
        "mean_control": float(np.mean(u)),
    }


def train_rollout_optimized_control(args: SimpleNamespace, iters: int = 450, n_steps: int = 120):
    """Train a small neural open-loop controller through differentiable RK4 rollout.

    This is included as a comparison point for the control topic: it optimizes
    the same original ODE rollout metric used for the fixed-control baselines.
    """
    torch.manual_seed(args.seed + 200)
    control = ControlNet(width=args.width, depth=getattr(args, "depth", 2), umax=args.umax).to(DEVICE)
    opt = torch.optim.Adam(control.parameters(), lr=args.lr)
    t_grid = torch.linspace(0.0, args.T, n_steps, device=DEVICE).view(-1, 1)
    dt = args.T / (n_steps - 1)
    x0 = torch.tensor([[0.95, 0.05, 0.0]], device=DEVICE)

    for _ in range(iters):
        opt.zero_grad()
        x = x0
        running_terms = []
        for k in range(n_steps):
            u = control(t_grid[k : k + 1])
            running_terms.append(args.A * x[:, 1:2] + 0.5 * args.B * u * u)
            if k == n_steps - 1:
                break
            f = lambda y: control_rhs(y, u, args.beta, args.gamma)
            k1 = f(x)
            k2 = f(x + 0.5 * dt * k1)
            k3 = f(x + 0.5 * dt * k2)
            k4 = f(x + dt * k3)
            x = x + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
            x = torch.clamp(x, 0.0, 1.0)
            x = x / torch.clamp(x.sum(dim=1, keepdim=True), min=1e-12)
        running = torch.cat(running_terms, dim=0)
        loss = args.T * running.mean() + args.AT * x[:, 1].mean()
        loss.backward()
        opt.step()
    return control


def evaluate_control_baselines(direct_control, pmp_control, rollout_control, args: SimpleNamespace) -> list[dict]:
    """Compare learned controls against no-control and fixed-control baselines."""
    fixed_candidates = [(u, rollout_control_objective(u, args)) for u in np.linspace(0.0, args.umax, 21)]
    best_u, best_fixed = min(fixed_candidates, key=lambda item: item[1]["objective"])

    methods = [
        ("No control", "No control", 0.0, "Baseline with no patching or mitigation."),
        (
            "Fixed moderate control u=0.30",
            "Fixed u=0.30",
            0.30,
            "Simple static control chosen before seeing the trajectory.",
        ),
        (
            f"Best fixed-control grid baseline u={best_u:.2f}",
            "Best fixed grid",
            float(best_u),
            "Strong fixed-control baseline selected by grid search.",
        ),
        (
            "Direct control PINN",
            "Direct PINN",
            direct_control,
            "Control network learned with objective plus ODE-residual loss.",
        ),
        (
            "PMP-informed PINN",
            "PMP-informed PINN",
            pmp_control,
            "Control network learned through Hamiltonian stationarity residuals.",
        ),
        (
            "Rollout-optimized neural control",
            "Rollout neural",
            rollout_control,
            "Extension experiment trained directly against the ODE rollout objective.",
        ),
    ]

    rows = []
    for method, plot_label, control_source, notes in methods:
        metrics = best_fixed if method.startswith("Best fixed-control") else rollout_control_objective(control_source, args)
        rows.append(
            baseline_row(
                topic="Controlled malware mitigation",
                method=method,
                plot_label=plot_label,
                primary_metric="rollout_objective",
                primary_value=metrics["objective"],
                objective=metrics["objective"],
                cumulative_infected=metrics["cumulative_infected"],
                peak_infected=metrics["peak_infected"],
                final_infected=metrics["final_infected"],
                mean_control=metrics["mean_control"],
                notes=notes,
            )
        )
    return rows


def numeric(row: dict, key: str) -> float:
    value = row.get(key, "")
    return float(value) if value != "" else float("nan")


def annotate_bars(ax, bars, values, fmt="{:.2g}") -> None:
    for bar, value in zip(bars, values):
        if np.isnan(value):
            continue
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            fmt.format(value),
            ha="center",
            va="bottom",
            fontsize=7.5,
            rotation=0,
        )


def plot_metric_bars(ax, rows: list[dict], metric: str, title: str, ylabel: str, log: bool = False) -> None:
    labels = [textwrap.fill(row["plot_label"], 12) for row in rows]
    values = [numeric(row, metric) for row in rows]
    colors = ["#2f5d8c", "#db8f34", "#4f9d69", "#8a5fbf", "#c94c4c", "#4c6f79"][: len(rows)]
    bars = ax.bar(labels, values, color=colors, edgecolor="#222222", linewidth=0.6)
    if log:
        positive = [value for value in values if value > 0 and not np.isnan(value)]
        if positive:
            ax.set_yscale("log")
            ax.set_ylim(max(min(positive) * 0.35, 1e-7), max(positive) * 4.0)
    annotate_bars(ax, bars, values, fmt="{:.2e}" if log else "{:.2f}")
    ax.set_title(title, fontsize=11)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", labelsize=8)
    ax.grid(axis="y", alpha=0.25)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)


def plot_baseline_comparison(output_path: Path, rows: list[dict]) -> None:
    """Create the reader-facing baseline comparison figure."""
    inverse_rows = [row for row in rows if row["topic"] == "Sparse-data inverse PINN"]
    pidl_rows = [row for row in rows if row["topic"] == "PIDL missing mechanism"]
    control_rows = [row for row in rows if row["topic"] == "Controlled malware mitigation"]

    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5))
    plot_metric_bars(
        axes[0, 0],
        inverse_rows,
        "state_mse",
        "Sparse-data inverse PINN: hidden-state error vs baselines",
        "Full S/I/R MSE (lower is better)",
        log=True,
    )
    plot_metric_bars(
        axes[0, 1],
        pidl_rows,
        "state_mse",
        "PIDL missing mechanism: learned correction vs known-model baseline",
        "Full S/I/R MSE (lower is better)",
        log=True,
    )
    plot_metric_bars(
        axes[1, 0],
        control_rows,
        "objective",
        "Controlled malware mitigation: rollout objective",
        "Objective: infected burden + control cost",
        log=False,
    )
    plot_metric_bars(
        axes[1, 1],
        control_rows,
        "cumulative_infected",
        "Controlled malware mitigation: epidemic burden",
        "Integral of compromised devices",
        log=False,
    )

    fig.suptitle("Note 2 Baseline Comparisons: learned methods against simple alternatives", fontsize=15)
    caption = (
        "Caption: inverse PINN is compared with sparse interpolation and a wrong-parameter SIR rollout; "
        "PIDL is compared with the known SIR model without the missing term; control methods are evaluated "
        "by rolling the original controlled malware model forward under each policy. Lower bars are better."
    )
    fig.text(0.5, 0.01, textwrap.fill(caption, 150), ha="center", va="bottom", fontsize=9)
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    output_path.parent.mkdir(exist_ok=True)
    fig.savefig(output_path, dpi=190)
    plt.close(fig)


def best_by_topic(rows: list[dict], topic: str) -> dict:
    topic_rows = [row for row in rows if row["topic"] == topic]
    return min(topic_rows, key=lambda row: numeric(row, "primary_value"))


def write_summary(path: Path, histories: dict[str, list[dict]], baseline_rows: list[dict], profile: dict) -> None:
    inv = histories["inverse"]
    pidl = histories["pidl"]
    control = histories["control"]
    pmp = histories["pmp"]
    best_inverse = best_by_topic(baseline_rows, "Sparse-data inverse PINN")
    best_pidl = best_by_topic(baseline_rows, "PIDL missing mechanism")
    control_best = min(
        [row for row in baseline_rows if row["topic"] == "Controlled malware mitigation"],
        key=lambda row: numeric(row, "objective"),
    )
    text = f"""# Training Summary

These diagnostics use the `{profile["name"]}` profile with `{profile["iters"]}` optimizer iterations per method. The default profile stays laptop-friendly; the GPU profile increases width/depth and collocation points for a more demanding local run.

## Profile Parameters

| Method | Width/depth | Data points | Collocation points |
|---|---:|---:|---:|
| Inverse PINN | {profile["inverse"]["width"]}/{profile["inverse"]["depth"]} | {profile["inverse"]["n_data"]} | {profile["inverse"]["n_collocation"]} |
| PIDL missing mechanism | {profile["pidl"]["width"]}/{profile["pidl"]["depth"]} | {profile["pidl"]["n_data"]} | {profile["pidl"]["n_collocation"]} |
| Direct control PINN | {profile["control"]["width"]}/{profile["control"]["depth"]} | n/a | {profile["control"]["n_collocation"]} |
| PMP-informed PINN | {profile["pmp"]["width"]}/{profile["pmp"]["depth"]} | n/a | {profile["pmp"]["n_collocation"]} |

## Loss Movement

| Diagnostic | Start | End | End/start |
|---|---:|---:|---:|
| Inverse PINN total loss | {inv[0]["loss"]:.3e} | {inv[-1]["loss"]:.3e} | {reduction(inv[0]["loss"], inv[-1]["loss"]):.3e} |
| PIDL total loss | {pidl[0]["loss"]:.3e} | {pidl[-1]["loss"]:.3e} | {reduction(pidl[0]["loss"], pidl[-1]["loss"]):.3e} |
| Direct control PINN total loss | {control[0]["loss"]:.3e} | {control[-1]["loss"]:.3e} | {reduction(control[0]["loss"], control[-1]["loss"]):.3e} |
| PMP-informed stationarity loss | {pmp[0]["stationarity_loss"]:.3e} | {pmp[-1]["stationarity_loss"]:.3e} | {reduction(pmp[0]["stationarity_loss"], pmp[-1]["stationarity_loss"]):.3e} |

The PMP-informed total loss can decrease more slowly because the costate boundary term and Hamiltonian residuals compete early in training.  In this tutorial run, the stationarity residual is the most important quick sanity signal.

## Baseline Comparison Snapshot

The second figure asks a different question: after training, how do the learned methods compare with simple alternatives?

| Topic | Best method in this run | Metric | Value |
|---|---|---|---:|
| Sparse-data inverse PINN | {best_inverse["method"]} | {best_inverse["primary_metric"]} | {numeric(best_inverse, "primary_value"):.3e} |
| PIDL missing mechanism | {best_pidl["method"]} | {best_pidl["primary_metric"]} | {numeric(best_pidl, "primary_value"):.3e} |
| Controlled malware mitigation | {control_best["method"]} | rollout objective | {numeric(control_best, "objective"):.3e} |

Open `figures/baseline_comparison.png` for the visual comparison and `experiments/baseline_comparison_metrics.csv` for the exact numbers.
"""
    path.write_text(text)


def write_output_preview(path: Path, histories: dict[str, list[dict]], baseline_rows: list[dict], profile: dict) -> None:
    """Write a short categorized guide to the generated Note 2 outputs."""
    inv = histories["inverse"]
    pidl = histories["pidl"]
    control = histories["control"]
    pmp = histories["pmp"]
    control_rows = [row for row in baseline_rows if row["topic"] == "Controlled malware mitigation"]
    best_control = min(control_rows, key=lambda row: numeric(row, "objective"))
    text = f"""# Output Preview

Use this page as the first stop after running `python scripts/run_training_iterations.py`.

Profile used: `{profile["name"]}` with `{profile["iters"]}` optimizer iterations per method.

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

## 3. Baseline Comparison Panels

Open `figures/baseline_comparison.png`.

| Panel | What is being compared | Main metric |
|---|---|---|
| Sparse-data inverse PINN | learned inverse PINN vs sparse interpolation and wrong-parameter SIR | full S/I/R mean squared error |
| PIDL missing mechanism | learned correction vs known SIR without the missing term | full S/I/R mean squared error |
| Controlled malware mitigation: objective | no control, fixed controls, direct PINN, PMP-informed PINN, and rollout-optimized neural control | infected burden plus control cost |
| Controlled malware mitigation: epidemic burden | same policies as above | integral of compromised devices over time |

Best rollout-control objective in this run: **{best_control["method"]}** with objective **{numeric(best_control, "objective"):.3e}**.

The direct-control and PMP-informed PINN panels above are training diagnostics. The baseline rollout panels use the original ODE simulator, so they deliberately show whether a learned control remains strong after it is rolled forward outside the training loss.

## 4. First-Versus-Last Snapshot

| Diagnostic | Start | End | End/start |
|---|---:|---:|---:|
| Inverse PINN total loss | {inv[0]["loss"]:.3e} | {inv[-1]["loss"]:.3e} | {reduction(inv[0]["loss"], inv[-1]["loss"]):.3e} |
| PIDL total loss | {pidl[0]["loss"]:.3e} | {pidl[-1]["loss"]:.3e} | {reduction(pidl[0]["loss"], pidl[-1]["loss"]):.3e} |
| Direct control PINN total loss | {control[0]["loss"]:.3e} | {control[-1]["loss"]:.3e} | {reduction(control[0]["loss"], control[-1]["loss"]):.3e} |
| PMP-informed stationarity loss | {pmp[0]["stationarity_loss"]:.3e} | {pmp[-1]["stationarity_loss"]:.3e} | {reduction(pmp[0]["stationarity_loss"], pmp[-1]["stationarity_loss"]):.3e} |

## 5. Files To Open First

| Category | File |
|---|---|
| Summary | `experiments/training_summary.md` |
| Learning curves | `figures/training_iteration_diagnostics.png` |
| Baseline comparison | `figures/baseline_comparison.png` |
| Baseline metrics | `experiments/baseline_comparison_metrics.csv` |
| Inverse PINN CSV | `experiments/inverse_pinn_training_history.csv` |
| PIDL CSV | `experiments/pidl_training_history.csv` |
| Direct control CSV | `experiments/control_pinn_training_history.csv` |
| PMP-informed CSV | `experiments/pmp_informed_pinn_training_history.csv` |
"""
    path.write_text(text)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run training-iteration experiments for Note 2.")
    parser.add_argument("--profile", choices=sorted(TRAINING_PROFILES), default="teaching", help="Named PINN/PIDL training profile.")
    parser.add_argument("--iters", type=int, default=None, help="Override iteration count for each PINN/PIDL diagnostic.")
    args = parser.parse_args()
    profile = resolve_training_profile(args.profile, args.iters)

    exp_dir = ROOT / "experiments"
    fig_dir = ROOT / "figures"

    inv_args = inverse_args(profile)
    pidl_cfg = pidl_args(profile)
    control_cfg = control_args(profile)
    pmp_cfg = pmp_args(profile)

    inverse_model, beta, gamma, inverse_history = train_inverse_pinn(inv_args)
    pidl_state, pidl_correction, pidl_history = train_pidl(pidl_cfg)
    _, direct_control, control_history = train_control_pinn(control_cfg)
    _, _, pmp_control, pmp_history = train_pmp_pinn(pmp_cfg)
    rollout_control = train_rollout_optimized_control(control_cfg)

    histories = {
        "inverse": inverse_history,
        "pidl": pidl_history,
        "control": control_history,
        "pmp": pmp_history,
    }
    baseline_rows = []
    baseline_rows.extend(evaluate_inverse_baselines(inverse_model, beta, gamma, inv_args))
    baseline_rows.extend(evaluate_pidl_baselines(pidl_state, pidl_correction))
    baseline_rows.extend(evaluate_control_baselines(direct_control, pmp_control, rollout_control, control_cfg))

    write_csv(exp_dir / "inverse_pinn_training_history.csv", inverse_history)
    write_csv(exp_dir / "pidl_training_history.csv", pidl_history)
    write_csv(exp_dir / "control_pinn_training_history.csv", control_history)
    write_csv(exp_dir / "pmp_informed_pinn_training_history.csv", pmp_history)
    write_csv(exp_dir / "baseline_comparison_metrics.csv", baseline_rows)
    write_output_preview(exp_dir / "OUTPUT_PREVIEW.md", histories, baseline_rows, profile)
    write_summary(exp_dir / "training_summary.md", histories, baseline_rows, profile)
    plot_training_diagnostics(fig_dir / "training_iteration_diagnostics.png", histories)
    plot_baseline_comparison(fig_dir / "baseline_comparison.png", baseline_rows)

    print(f"Wrote experiment CSV files to {exp_dir}")
    print(f"Wrote training diagnostic figure to {fig_dir / 'training_iteration_diagnostics.png'}")
    print(f"Wrote baseline comparison figure to {fig_dir / 'baseline_comparison.png'}")


if __name__ == "__main__":
    main()
