"""
Copyright (c) 2026 Luxing Yang.
Licensed under the MIT License. See LICENSE in the repository root.

Small inverse PINN for node-level SIPRS dynamics on a graph.

The example uses the canonical foundation SIPRS simulator to create synthetic
truth, observes infected probabilities for only a subset of nodes/times, and
learns hidden node states plus beta/gamma.  It is intentionally small: a
time-only MLP is a bridge from aggregate examples to graph-aware models, not a
claim that this architecture scales to large graphs.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, asdict
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shared_setup import ensure_foundation_package

ensure_foundation_package()

from cybercontrol.network_models import NodeSIPRSParams, node_siprs_rhs_numpy, node_siprs_rhs_torch, normalize_adjacency
from cybercontrol.numerics import project_compartments, rk4_integrate
from cybercontrol.torch_utils import MLP, positive, time_derivative, configure_torch


@dataclass
class NodeSIPRSInverseConfig:
    """Configuration for a tiny node-SIPRS inverse PINN smoke experiment."""

    nodes: int = 8
    horizon: float = 6.0
    grid: int = 61
    beta_true: float = 0.82
    gamma_true: float = 0.18
    omega_p: float = 0.03
    omega_r: float = 0.02
    patch: float = 0.08
    clean: float = 0.04
    observed_nodes: int = 4
    observed_times: int = 14
    noise: float = 0.0
    seed: int = 31


def toy_adjacency(nodes: int) -> np.ndarray:
    """Return a row-normalized ring-plus-chord graph."""

    A = np.zeros((nodes, nodes), dtype=np.float64)
    for i in range(nodes):
        A[i, (i - 1) % nodes] = 1.0
        A[i, (i + 1) % nodes] = 1.0
        if nodes > 4:
            A[i, (i + 3) % nodes] = 0.5
    return normalize_adjacency(A)


def generate_truth(cfg: NodeSIPRSInverseConfig):
    """Generate canonical node-SIPRS truth with fixed patch/clean controls."""

    rng = np.random.default_rng(cfg.seed)
    A = toy_adjacency(cfg.nodes)
    x0 = np.zeros((cfg.nodes, 4), dtype=np.float64)
    x0[:, 0] = 1.0 - 0.04
    x0[:, 1] = 0.04
    seeds = rng.choice(cfg.nodes, size=max(1, cfg.nodes // 4), replace=False)
    x0[seeds, 1] += 0.08
    x0[seeds, 0] -= 0.08
    x0 = project_compartments(x0)
    params = NodeSIPRSParams(beta=cfg.beta_true, gamma=cfg.gamma_true, omega_p=cfg.omega_p, omega_r=cfg.omega_r)
    t = np.linspace(0.0, cfg.horizon, cfg.grid)

    def rhs_flat(y, tau):
        x = y.reshape(cfg.nodes, 4)
        return node_siprs_rhs_numpy(x, A, params, patch=cfg.patch, clean=cfg.clean).reshape(-1)

    y = x0.reshape(-1)
    path = [x0.copy()]
    for k in range(cfg.grid - 1):
        y, _ = rk4_integrate(
            rhs_flat,
            y,
            t0=float(t[k]),
            dt=float(t[k + 1] - t[k]),
            substeps=2,
            project=lambda z: project_compartments(z.reshape(cfg.nodes, 4)).reshape(-1),
        )
        path.append(y.reshape(cfg.nodes, 4).copy())
    return t, np.asarray(path), A


class NodeSIPRSStateNet:
    """Time-to-node-compartment network using a shared MLP block."""

    def __init__(self, torch, nodes: int, width: int, depth: int, device: str):
        self.torch = torch
        self.nodes = nodes
        self.net = MLP(1, nodes * 4, width=width, depth=depth).to(device)

    def parameters(self):
        return self.net.parameters()

    def __call__(self, t):
        raw = self.net(t).reshape(t.shape[0], self.nodes, 4)
        return self.torch.softmax(raw, dim=-1)


def train(args):
    """Train the small node-SIPRS inverse PINN and return diagnostics."""

    torch, device, _ = configure_torch(seed=args.seed, device=args.device, threads=1)
    cfg = NodeSIPRSInverseConfig(
        nodes=args.nodes,
        grid=args.grid,
        observed_nodes=args.observed_nodes,
        observed_times=args.observed_times,
        noise=args.noise,
        seed=args.seed,
    )
    t_np, x_np, A_np = generate_truth(cfg)
    rng = np.random.default_rng(cfg.seed)
    node_idx = np.sort(rng.choice(cfg.nodes, size=min(cfg.observed_nodes, cfg.nodes), replace=False))
    data_idx = np.linspace(0, cfg.grid - 1, cfg.observed_times, dtype=int)
    heldout_idx = np.setdiff1d(np.arange(cfg.grid), data_idx)[:: max(1, cfg.grid // 12)]
    observed_I = x_np[np.ix_(data_idx, node_idx, [1])].squeeze(-1)
    if cfg.noise > 0:
        observed_I = np.clip(observed_I + rng.normal(0.0, cfg.noise, observed_I.shape), 0.0, 1.0)

    t_data = torch.tensor(t_np[data_idx, None], dtype=torch.float32, device=device)
    y_data = torch.tensor(observed_I, dtype=torch.float32, device=device)
    t_f = torch.linspace(0.0, cfg.horizon, args.collocation, device=device).reshape(-1, 1)
    t_f.requires_grad_(True)
    x0 = torch.tensor(x_np[0], dtype=torch.float32, device=device)
    A_t = torch.tensor(A_np, dtype=torch.float32, device=device)
    model = NodeSIPRSStateNet(torch, cfg.nodes, args.width, args.depth, device)
    beta_raw = torch.nn.Parameter(torch.tensor(0.0, device=device))
    gamma_raw = torch.nn.Parameter(torch.tensor(-1.0, device=device))
    opt = torch.optim.Adam(list(model.parameters()) + [beta_raw, gamma_raw], lr=args.lr)
    node_idx_t = torch.tensor(node_idx, dtype=torch.long, device=device)
    history = []

    for it in range(args.iters):
        opt.zero_grad()
        beta = positive(beta_raw)
        gamma = positive(gamma_raw)
        pred_data = model(t_data)[:, node_idx_t, 1]
        loss_data = torch.mean((pred_data - y_data) ** 2)
        pred0 = model(torch.zeros(1, 1, device=device))[0]
        loss_ic = torch.mean((pred0 - x0) ** 2)
        x_f = model(t_f)
        flat = x_f.reshape(x_f.shape[0], -1)
        dxdt = time_derivative(flat, t_f).reshape_as(x_f)
        params = NodeSIPRSParams(beta=beta, gamma=gamma, omega_p=cfg.omega_p, omega_r=cfg.omega_r)
        rhs = torch.stack(
            [
                node_siprs_rhs_torch(x_f[k], A_t, params, patch=cfg.patch, clean=cfg.clean)
                for k in range(x_f.shape[0])
            ],
            dim=0,
        )
        loss_residual = torch.mean((dxdt - rhs) ** 2)
        loss_mass = torch.mean((x_f.sum(dim=-1) - 1.0) ** 2)
        loss = loss_data + args.w_ic * loss_ic + args.w_residual * loss_residual + args.w_mass * loss_mass
        loss.backward()
        opt.step()
        if it % args.log_every == 0 or it == args.iters - 1:
            with torch.no_grad():
                held_t = torch.tensor(t_np[heldout_idx, None], dtype=torch.float32, device=device)
                held_pred = model(held_t).cpu().numpy()
                held_true = x_np[heldout_idx]
                held_mse = float(np.mean((held_pred - held_true) ** 2))
                row = {
                    "iteration": it,
                    "loss": float(loss.detach().cpu()),
                    "data_loss": float(loss_data.detach().cpu()),
                    "residual_loss": float(loss_residual.detach().cpu()),
                    "heldout_state_mse": held_mse,
                    "beta": float(beta.detach().cpu()),
                    "gamma": float(gamma.detach().cpu()),
                    "beta_abs_error": abs(float(beta.detach().cpu()) - cfg.beta_true),
                    "gamma_abs_error": abs(float(gamma.detach().cpu()) - cfg.gamma_true),
                    "mass_error": float(torch.max(torch.abs(x_f.sum(dim=-1) - 1.0)).detach().cpu()),
                }
                history.append(row)
                print(
                    f"it={it:05d}, loss={row['loss']:.3e}, heldout={held_mse:.2e}, "
                    f"beta={row['beta']:.3f}, gamma={row['gamma']:.3f}"
                )
    if getattr(args, "return_history", False):
        return model, history, asdict(cfg)
    return model, history


def build_parser():
    parser = argparse.ArgumentParser(description="Small inverse PINN for canonical node-SIPRS graph dynamics.")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--nodes", type=int, default=8)
    parser.add_argument("--grid", type=int, default=61)
    parser.add_argument("--observed-nodes", type=int, default=4)
    parser.add_argument("--observed-times", type=int, default=14)
    parser.add_argument("--collocation", type=int, default=32)
    parser.add_argument("--iters", type=int, default=500)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--noise", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--w-ic", type=float, default=10.0)
    parser.add_argument("--w-residual", type=float, default=1.0)
    parser.add_argument("--w-mass", type=float, default=1.0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--output-csv", type=Path, default=None)
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    if args.device == "auto":
        args.device = None
    if args.smoke:
        args.nodes = 6
        args.grid = 25
        args.observed_nodes = 3
        args.observed_times = 8
        args.collocation = 12
        args.iters = 12
        args.width = 16
        args.log_every = 4
    _, history = train(args)
    if args.output_csv is not None:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(history[0].keys()))
            writer.writeheader()
            writer.writerows(history)
        print(f"wrote {args.output_csv}")
