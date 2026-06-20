"""
Copyright (c) 2026 Luxing Yang.
Licensed under the MIT License. See LICENSE in the repository root.

PIDL example: known malware mechanism plus an unknown correction term.

The synthetic data are generated from a model with an extra nonlinear behavior
change term:
    S' = -beta S I - q S I^2
    I' =  beta S I + q S I^2 - gamma I
    R' =  gamma I
The learner is given the known SIR part and learns a neural correction g_phi(S,I,R)
so that
    x' = f_known(x; beta,gamma) + B g_phi(x)
where B=[-1,+1,0]^T keeps mass conserved.

PIDL keeps the known mechanism explicit and assigns the correction network only
to the residual behavior that the known model cannot explain.
"""
from __future__ import annotations

import argparse
import torch
import torch.nn as nn
from shared_setup import ensure_foundation_package, resolve_torch_device

ensure_foundation_package()
from cybercontrol.models import sir_rhs_torch as known_rhs
from cybercontrol.torch_utils import MLP, SimplexStateNet, configure_torch, positive, rk4_step_torch


def true_rhs(x, beta, gamma, q):
    S, I, R = x[..., 0], x[..., 1], x[..., 2]
    corr = q*S*I*I
    return known_rhs(x, beta, gamma) + torch.stack([-corr, corr, torch.zeros_like(corr)], dim=-1)


def generate(T=20.0, n=400, beta=0.8, gamma=0.2, q=1.2):
    """Generate data from a known SIR model plus a hidden nonlinear correction."""
    t = torch.linspace(0, T, n).view(-1, 1)
    dt = T/(n-1)
    x = torch.zeros(n, 3); x[0] = torch.tensor([0.95, 0.05, 0.0])
    for k in range(n-1):
        rhs = lambda y: true_rhs(y, torch.tensor(beta), torch.tensor(gamma), torch.tensor(q))
        x[k+1] = rk4_step_torch(x[k], dt, rhs, project_simplex=True)
    return t, x


class CorrectionNet(nn.Module):
    def __init__(self, width=64, depth=2):
        super().__init__()
        self.net = MLP(3, 1, width=width, depth=depth)

    def forward(self, x):
        # softplus makes the correction nonnegative.  Remove it if sign is unknown.
        return torch.nn.functional.softplus(self.net(x))


def train(args):
    """Train a PIDL state network and missing-mechanism correction network.

    The known SIR part remains explicit.  The correction network only explains
    residual behavior that the known mechanism cannot capture.
    """
    _, device, _ = resolve_torch_device(
        configure_torch,
        seed=args.seed,
        device=getattr(args, "device", "auto"),
        threads=getattr(args, "threads", 1),
    )
    t_all, x_all = generate()
    idx = torch.linspace(0, len(t_all)-1, args.n_data).long()
    t_data = t_all[idx].to(device)
    I_data = x_all[idx, 1:2].to(device)
    depth = getattr(args, "depth", 2)
    state_net = SimplexStateNet(width=args.width, depth=depth).to(device)
    corr_net = CorrectionNet(args.width, depth=depth).to(device)
    beta_raw = nn.Parameter(torch.tensor(0.0, device=device))
    gamma_raw = nn.Parameter(torch.tensor(-1.0, device=device))
    opt = torch.optim.Adam(list(state_net.parameters()) + list(corr_net.parameters()) + [beta_raw, gamma_raw], lr=args.lr)
    t_f = torch.linspace(0, 20.0, args.n_collocation).view(-1,1).to(device); t_f.requires_grad_(True)
    x0 = torch.tensor([[0.95,0.05,0.0]], device=device)
    B = torch.tensor([[-1.0, 1.0, 0.0]], device=device)
    history = []

    for it in range(args.iters):
        opt.zero_grad()
        beta, gamma = positive(beta_raw), positive(gamma_raw)
        x_data_pred = state_net(t_data)
        loss_data = torch.mean((x_data_pred[:,1:2] - I_data)**2)
        loss_ic = torch.mean((state_net(torch.zeros(1,1,device=device)) - x0)**2)
        x_f = state_net(t_f)
        dxdt = torch.cat([torch.autograd.grad(x_f[:,j].sum(), t_f, create_graph=True)[0] for j in range(3)], dim=1)
        g = corr_net(x_f)              # shape [N,1]
        rhs = known_rhs(x_f, beta, gamma) + g @ B
        loss_res = torch.mean((dxdt - rhs)**2)
        # Mild regularization discourages unneeded correction if known model suffices.
        loss_corr = torch.mean(g**2)
        loss = loss_data + args.w_ic*loss_ic + args.w_res*loss_res + args.w_corr*loss_corr
        loss.backward(); opt.step()
        if it % args.log_every == 0:
            row = {
                "iteration": it,
                "loss": float(loss.detach().item()),
                "beta": float(beta.detach().item()),
                "gamma": float(gamma.detach().item()),
                "data_loss": float(loss_data.detach().item()),
                "residual_loss": float(loss_res.detach().item()),
                "correction_regularizer": float(loss_corr.detach().item()),
                "mean_correction": float(g.mean().detach().item()),
            }
            history.append(row)
            print(
                f"it={it:05d}, loss={row['loss']:.2e}, "
                f"beta={row['beta']:.3f}, gamma={row['gamma']:.3f}, "
                f"mean_g={row['mean_correction']:.3f}"
            )
    if getattr(args, "return_history", False):
        return state_net, corr_net, history
    return state_net, corr_net


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Train a PIDL model with a known SIR mechanism and learned correction.")
    p.add_argument("--smoke", action="store_true", help="Run a tiny execution check.")
    p.add_argument("--iters", type=int, default=5000, help="Number of optimizer iterations.")
    p.add_argument("--n-data", type=int, default=40, help="Number of sparse infected-observation points.")
    p.add_argument("--n-collocation", type=int, default=200, help="Number of collocation points.")
    p.add_argument("--width", type=int, default=64, help="Hidden width for state/correction networks.")
    p.add_argument("--depth", type=int, default=2, help="Hidden-layer depth for state/correction networks.")
    p.add_argument("--lr", type=float, default=1e-3, help="Adam learning rate.")
    p.add_argument("--w-ic", type=float, default=10.0, help="Weight on initial-condition loss.")
    p.add_argument("--w-res", type=float, default=1.0, help="Weight on residual loss.")
    p.add_argument("--w-corr", type=float, default=1e-3, help="Weight on correction regularization.")
    p.add_argument("--log-every", type=int, default=1000, help="Iteration interval for console logs and history rows.")
    p.add_argument("--seed", type=int, default=2, help="Random seed.")
    p.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto", help="Training device.")
    p.add_argument("--threads", type=int, default=1, help="Torch CPU thread count; use 0 to leave unchanged.")
    args = p.parse_args()
    if args.smoke:
        args.iters = 10
        args.n_collocation = 50
        args.n_data = 10
        args.width = 16
        args.depth = 2
        args.log_every = 1
    train(args)
