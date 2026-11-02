"""Infer SIR states and transmission/recovery rates from sparse infection data.

Copyright (c) 2026 Luxing Yang. Licensed under the repository MIT License.
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn

from cybercontrol.experiments import configure_torch
from cybercontrol.models import sir_rhs_torch as sir_rhs
from cybercontrol.nn import SimplexStateNet
from cybercontrol.pinn import positive, time_derivative
from cybercontrol.torch_numerics import project_compartments_torch, rk4_step_torch

LOGGER = logging.getLogger(__name__)


def generate_data(beta_true=0.8, gamma_true=0.2, T=20.0, n_grid=400, noise=0.0):
    """Generate an SIR trajectory; use observation_data for measurement noise."""
    if n_grid < 2 or T <= 0:
        raise ValueError("Use a positive horizon and at least two time points.")
    t = torch.linspace(0, T, n_grid).view(-1, 1)
    x = torch.zeros(n_grid, 3)
    x[0] = torch.tensor([0.95, 0.05, 0.0])
    beta, gamma = torch.tensor(beta_true), torch.tensor(gamma_true)
    for k in range(n_grid - 1):
        x[k + 1] = rk4_step_torch(
            x[k], T / (n_grid - 1), lambda state: sir_rhs(state, beta, gamma), project_simplex=True
        )
    if noise > 0:
        noisy = x.clone()
        noisy[:, 1] = torch.clamp(noisy[:, 1] + noise * torch.randn_like(noisy[:, 1]), 0.0, 1.0)
        x = project_compartments_torch(noisy)
    return t, x


def observation_data(n_data: int, noise: float = 0.0, seed: int = 1):
    """Return clean truth, observed indices and noisy I measurements separately."""
    if not 2 <= n_data <= 400 or noise < 0:
        raise ValueError("Use 2-400 observations and nonnegative measurement noise.")
    time, truth = generate_data()
    indices = torch.linspace(0, len(time) - 1, n_data).long()
    generator = torch.Generator().manual_seed(seed + 1_000_000)
    perturbation = noise * torch.randn((n_data, 1), generator=generator)
    observed = truth[indices, 1:2] + perturbation
    return time, truth, indices, observed


class StateNet(SimplexStateNet):
    """A simplex-valued trajectory network with scaled physical time in [0,20]."""

    def __init__(self, width=64, depth=4):
        super().__init__(width=width, depth=depth)

    def forward(self, time):
        return super().forward(time / 10.0 - 1.0)


def train(args):
    """Minimize data, initial-condition and ODE losses with positive rate parameters."""
    if args.iters < 1 or args.n_collocation < 2:
        raise ValueError("Use at least one training step and two collocation points.")
    _, device, _ = configure_torch(
        seed=args.seed,
        device=getattr(args, "device", "auto"),
        threads=getattr(args, "threads", 1),
    )
    time, _, indices, observed = observation_data(args.n_data, args.noise, args.seed)
    t_data, infected_data = time[indices].to(device), observed.to(device)
    model = StateNet(width=args.width, depth=args.depth).to(device)
    beta_raw = nn.Parameter(torch.tensor(0.0, device=device))
    gamma_raw = nn.Parameter(torch.tensor(-1.0, device=device))
    optimizer = torch.optim.Adam(list(model.parameters()) + [beta_raw, gamma_raw], lr=args.lr)
    t_f = torch.linspace(0, 20.0, args.n_collocation, device=device).view(-1, 1)
    t_f.requires_grad_(True)
    x0 = torch.tensor([[0.95, 0.05, 0.0]], device=device)
    history = []

    for iteration in range(args.iters):
        optimizer.zero_grad()
        t_f.grad = None
        beta, gamma = positive(beta_raw), positive(gamma_raw)
        loss_data = torch.mean((model(t_data)[:, 1:2] - infected_data) ** 2)
        loss_ic = torch.mean((model(torch.zeros(1, 1, device=device)) - x0) ** 2)
        state = model(t_f)
        derivative = time_derivative(state, t_f)
        loss_ode = torch.mean((derivative - sir_rhs(state, beta, gamma)) ** 2)
        loss = loss_data + args.w_ic * loss_ic + args.w_ode * loss_ode
        loss.backward()
        optimizer.step()
        if iteration % args.log_every == 0 or iteration == args.iters - 1:
            row = {
                "iteration": iteration,
                "loss": float(loss.detach().item()),
                "beta": float(positive(beta_raw).detach().item()),
                "gamma": float(positive(gamma_raw).detach().item()),
                "data_loss": float(loss_data.detach().item()),
                "ode_loss": float(loss_ode.detach().item()),
                "initial_condition_loss": float(loss_ic.detach().item()),
            }
            history.append(row)
            LOGGER.info(
                "step=%d loss=%.3g beta=%.3f gamma=%.3f",
                iteration,
                row["loss"],
                row["beta"],
                row["gamma"],
            )
    result = (model, positive(beta_raw).item(), positive(gamma_raw).item())
    return (*result, history) if getattr(args, "return_history", False) else result
