"""Learn a contact-driven correction to a known SIR mechanism.

Copyright (c) 2026 Luxing Yang. Licensed under the repository MIT License.
The synthetic extra flux is q*S*I**2. The learned flux S*I*softplus(N(x))
preserves mass and vanishes at the susceptible and infected boundaries.
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn

from cybercontrol.experiments import configure_torch
from cybercontrol.models import sir_rhs_torch as known_rhs
from cybercontrol.nn import MLP, SimplexStateNet
from cybercontrol.pinn import positive, time_derivative
from cybercontrol.torch_numerics import rk4_step_torch

LOGGER = logging.getLogger(__name__)


def true_rhs(x, beta, gamma, q):
    susceptible = x[..., 0]
    infected = x[..., 1]
    flux = q * susceptible * infected * infected
    return known_rhs(x, beta, gamma) + torch.stack([-flux, flux, torch.zeros_like(flux)], dim=-1)


def generate(T=20.0, n=400, beta=0.8, gamma=0.2, q=1.2):
    """Generate a trajectory with the known mechanism and an additional flux."""
    if T <= 0 or n < 2 or min(beta, gamma, q) < 0:
        raise ValueError("Use positive time, at least two points and nonnegative rates.")
    t = torch.linspace(0, T, n).view(-1, 1)
    x = torch.zeros(n, 3)
    x[0] = torch.tensor([0.95, 0.05, 0.0])
    beta_t, gamma_t, q_t = torch.tensor(beta), torch.tensor(gamma), torch.tensor(q)
    for k in range(n - 1):
        x[k + 1] = rk4_step_torch(
            x[k],
            T / (n - 1),
            lambda state: true_rhs(state, beta_t, gamma_t, q_t),
            project_simplex=True,
        )
    return t, x


class CorrectionNet(nn.Module):
    """Return an extra transmission flux, rather than an unrestricted source."""

    def __init__(self, width=64, depth=2):
        super().__init__()
        self.net = MLP(3, 1, width=width, depth=depth)

    def forward(self, x):
        rate = torch.nn.functional.softplus(self.net(x))
        return x[..., 0:1] * x[..., 1:2] * rate


def train(args):
    """Fit observed infection, the initial state and the corrected differential equation."""
    if args.iters < 1 or args.n_collocation < 2 or args.log_every < 1:
        raise ValueError(
            "Use positive iterations/log interval and at least two collocation points."
        )
    if not 2 <= args.n_data <= 400:
        raise ValueError("Use between 2 and 400 distinct observation points.")
    _, device, _ = configure_torch(
        seed=args.seed,
        device=getattr(args, "device", "auto"),
        threads=getattr(args, "threads", 1),
    )
    t_all, x_all = generate()
    index = torch.linspace(0, len(t_all) - 1, args.n_data).long()
    t_data = t_all[index].to(device)
    infected_data = x_all[index, 1:2].to(device)
    depth = getattr(args, "depth", 2)
    state_net = SimplexStateNet(width=args.width, depth=depth).to(device)
    corr_net = CorrectionNet(args.width, depth=depth).to(device)
    beta_raw = nn.Parameter(torch.tensor(0.0, device=device))
    gamma_raw = nn.Parameter(torch.tensor(-1.0, device=device))
    trainable = list(state_net.parameters()) + list(corr_net.parameters()) + [beta_raw, gamma_raw]
    optimizer = torch.optim.Adam(trainable, lr=args.lr)
    time = torch.linspace(0, 20.0, args.n_collocation, device=device).view(-1, 1)
    time.requires_grad_(True)
    initial = torch.tensor([[0.95, 0.05, 0.0]], device=device)
    transfer = torch.tensor([[-1.0, 1.0, 0.0]], device=device)
    history = []

    for iteration in range(args.iters):
        optimizer.zero_grad()
        time.grad = None
        beta, gamma = positive(beta_raw), positive(gamma_raw)
        loss_data = torch.mean((state_net(t_data)[:, 1:2] - infected_data) ** 2)
        loss_ic = torch.mean((state_net(torch.zeros(1, 1, device=device)) - initial) ** 2)
        state = state_net(time)
        derivative = time_derivative(state, time)
        flux = corr_net(state)
        rhs = known_rhs(state, beta, gamma) + flux @ transfer
        loss_residual = torch.mean((derivative - rhs) ** 2)
        loss_correction = torch.mean(flux**2)
        loss = (
            loss_data
            + args.w_ic * loss_ic
            + args.w_res * loss_residual
            + args.w_corr * loss_correction
        )
        loss.backward()
        optimizer.step()
        if iteration % args.log_every == 0 or iteration == args.iters - 1:
            row = {
                "iteration": iteration,
                "loss": float(loss.detach().item()),
                "beta": float(positive(beta_raw).detach().item()),
                "gamma": float(positive(gamma_raw).detach().item()),
                "data_loss": float(loss_data.detach().item()),
                "residual_loss": float(loss_residual.detach().item()),
                "correction_regularizer": float(loss_correction.detach().item()),
                "mean_correction": float(flux.mean().detach().item()),
            }
            history.append(row)
            LOGGER.info(
                "step=%d loss=%.3g extra_flux=%.3g", iteration, row["loss"], row["mean_correction"]
            )
    return (
        (state_net, corr_net, history)
        if getattr(args, "return_history", False)
        else (state_net, corr_net)
    )
