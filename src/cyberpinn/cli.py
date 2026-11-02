"""Infer rates from sparse data and compare PINN predictions with ODE fitting."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import torch
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares

from cybercontrol.models import MalwareParams, controlled_sir_rhs

from .configs import InverseConfig
from .inverse import observation_data, train

ROOT = Path(__file__).resolve().parents[2]


def ode_trajectory(beta: float, gamma: float, time: np.ndarray) -> np.ndarray:
    """Integrate the known SIR mechanism with the same initial state as the PINN."""
    params = MalwareParams(beta=beta, gamma=gamma, omega=0.0)
    solution = solve_ivp(
        lambda t, x: controlled_sir_rhs(x, 0.0, 0.0, params),
        (0.0, float(time[-1])),
        [0.95, 0.05, 0.0],
        t_eval=time,
        rtol=1e-8,
        atol=1e-10,
    )
    if not solution.success:
        raise RuntimeError(solution.message)
    return solution.y.T


def fit_ode(time: np.ndarray, infected: np.ndarray):
    """Fit only observed I values; positivity is enforced through log parameters."""
    result = least_squares(
        lambda rates: ode_trajectory(*np.exp(rates), time)[:, 1] - infected,
        np.log([0.7, 0.3]),
        bounds=(np.log([0.01, 0.01]), np.log([3.0, 3.0])),
        max_nfev=100,
    )
    if not result.success:
        raise RuntimeError(f"ODE fitting did not converge: {result.message}")
    return tuple(np.exp(result.x))


def run_example(args: argparse.Namespace) -> pd.DataFrame:
    """Use the same observations for a neural trajectory and a mechanistic baseline."""
    import matplotlib.pyplot as plt

    config = InverseConfig(
        iters=args.iters,
        width=32,
        depth=2,
        n_data=args.observations,
        n_collocation=64,
        noise=args.noise,
        log_every=max(1, args.iters // 20),
        seed=args.seed,
        device=args.device,
    )
    time, truth, indices, measured = observation_data(config.n_data, config.noise, config.seed)
    model, beta, gamma, history = train(config)
    device = next(model.parameters()).device
    with torch.no_grad():
        prediction = model(time.to(device)).cpu().numpy()
    grid, exact = time[:, 0].numpy().astype(float), truth.numpy()
    index = indices.numpy()
    observations = measured[:, 0].numpy()
    fitted_beta, fitted_gamma = fit_ode(grid[index], observations)
    fitted = ode_trajectory(fitted_beta, fitted_gamma, grid)
    replay = ode_trajectory(beta, gamma, grid)
    methods = {
        "PINN trajectory": (prediction, beta, gamma),
        "PINN rates / ODE replay": (replay, beta, gamma),
        "ODE least squares": (fitted, fitted_beta, fitted_gamma),
    }
    heldout = np.ones(len(grid), dtype=bool)
    heldout[index] = False
    rows = []
    for name, (state, estimated_beta, estimated_gamma) in methods.items():
        rows.append(
            {
                "method": name,
                "beta": estimated_beta,
                "gamma": estimated_gamma,
                "observed_I_rmse": np.sqrt(np.mean((state[index, 1] - observations) ** 2)),
                "heldout_I_rmse": np.sqrt(np.mean((state[heldout, 1] - exact[heldout, 1]) ** 2)),
            }
        )
    comparison = pd.DataFrame(rows)
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(out / "comparison.csv", index=False)
    pd.DataFrame({"time": grid[index], "infected_observed": observations}).to_csv(
        out / "observations.csv", index=False
    )
    table = {"time": grid}
    for name, state in (("true", exact), ("pinn", prediction), ("replay", replay), ("ode", fitted)):
        for j, compartment in enumerate(("S", "I", "R")):
            table[f"{compartment}_{name}"] = state[:, j]
    pd.DataFrame(table).to_csv(out / "trajectories.csv", index=False)
    history_table = pd.DataFrame(history)
    history_table.to_csv(out / "training.csv", index=False)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(grid, exact[:, 1], label="Synthetic truth")
    for name, (state, _, _) in methods.items():
        ax.plot(grid, state[:, 1], label=name)
    ax.scatter(grid[index], observations, s=18, label="Observed I")
    ax.set(xlabel="Time", ylabel="Infected fraction")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "fit.png", dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4))
    for column in ("data_loss", "ode_loss", "initial_condition_loss"):
        ax.semilogy(history_table["iteration"], history_table[column], label=column.replace("_", " "))
    ax.set(xlabel="Training step", ylabel="Unweighted mean-square loss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "learning.png", dpi=160)
    plt.close(fig)
    print("Synthetic generating rates: beta=0.8, gamma=0.2")
    print(comparison.to_string(index=False, float_format=lambda value: f"{value:.4g}"))
    print(f"Figures and tables: {out}; held-out times measure interpolation, not extrapolation.")
    return comparison


def _check() -> None:
    config = InverseConfig(
        iters=4, width=8, depth=2, n_data=6, n_collocation=12, log_every=1, device="cpu"
    )
    model, beta, gamma, history = train(config)
    state = model(torch.tensor([[0.0], [10.0], [20.0]]))
    assert torch.isfinite(state).all() and torch.allclose(state.sum(1), torch.ones(3))
    assert beta > 0 and gamma > 0 and np.isfinite(history[-1]["loss"])
    print("PINN gradients, simplex states and positive rates passed.")


def _build_docs() -> None:
    source = ROOT / "docs" / "source"
    if not source.is_dir():
        raise RuntimeError("Building the lecture PDF requires a repository checkout.")
    name = "note2_pinn_pidl_cyber_control"
    subprocess.run(
        ["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", f"{name}.tex"],
        cwd=source,
        check=True,
    )
    (source.parent / f"{name}.pdf").write_bytes((source / f"{name}.pdf").read_bytes())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare inverse PINN and ODE parameter fitting.")
    parser.add_argument(
        "command", nargs="?", choices=["example", "check", "study", "docs"], default="example"
    )
    parser.add_argument("--iters", type=int, default=1500)
    parser.add_argument("--observations", type=int, default=24)
    parser.add_argument("--noise", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", choices=["cpu", "auto", "cuda", "mps"], default="cpu")
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    return parser


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    aliases = {"smoke": "check", "figures": "example", "medium": "study"}
    if argv:
        argv[0] = aliases.get(argv[0], argv[0])
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.iters < 1 or not 2 <= args.observations < 400 or args.noise < 0:
        parser.error("Use positive training steps, 2-399 observations, and nonnegative noise.")
    if args.command == "check":
        _check()
    elif args.command == "docs":
        _build_docs()
    elif args.command == "study":
        from .experiments import run_medium

        run_medium(args.output_dir, args.device)
    else:
        run_example(args)


if __name__ == "__main__":
    main()
