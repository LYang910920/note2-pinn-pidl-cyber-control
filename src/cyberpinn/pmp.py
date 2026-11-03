"""PMP-informed neural control for the bounded SIR patching problem.

Copyright (c) 2026 Luxing Yang. Licensed under the repository MIT License.
State, costate and control networks approximate the normal PMP system. The
projected control residual includes interior and boundary Hamiltonian minima.
"""

from __future__ import annotations

import logging

import torch

from cybercontrol.experiments import configure_torch
from cybercontrol.models import controlled_sir_rhs_torch as f_state
from cybercontrol.nn import BoundedControlNet, MLP, SimplexStateNet
from cybercontrol.pinn import time_derivative

StateNet = SimplexStateNet
ControlNet = BoundedControlNet
LOGGER = logging.getLogger(__name__)


def hamiltonian(x, u, lam, A, B, beta, gamma):
    """Minimization Hamiltonian: A*I + B*u**2/2 + lambda^T f."""
    f = f_state(x, u, beta, gamma)
    running = A * x[:, 1:2] + 0.5 * B * u * u
    return running + torch.sum(lam * f, dim=1, keepdim=True)


def projected_control_residual(u, hu, umax: float, curvature: float):
    """Zero at the constrained minimizer of the quadratic control Hamiltonian."""
    if curvature <= 0 or umax < 0:
        raise ValueError("Control curvature must be positive and its upper bound nonnegative.")
    return u - torch.clamp(u - hu / curvature, min=0.0, max=umax)


def train(args):
    """Fit state, costate, constrained optimality and endpoint residuals."""
    if args.T <= 0 or args.n_collocation < 2 or args.iters < 1 or args.log_every < 1:
        raise ValueError(
            "Use a positive horizon, iterations/log interval and two collocation points."
        )
    if args.B <= 0 or args.umax < 0:
        raise ValueError("B must be positive and umax nonnegative.")
    _, device, _ = configure_torch(
        seed=args.seed,
        device=getattr(args, "device", "auto"),
        threads=getattr(args, "threads", 1),
    )
    depth = getattr(args, "depth", 2)
    state = StateNet(width=args.width, depth=depth).to(device)
    costate = MLP(1, 3, args.width, depth=depth).to(device)
    control = ControlNet(width=args.width, depth=depth, umax=args.umax).to(device)
    opt = torch.optim.Adam(
        list(state.parameters()) + list(costate.parameters()) + list(control.parameters()),
        lr=args.lr,
    )
    t = torch.linspace(0, args.T, args.n_collocation, device=device).view(-1, 1)
    t.requires_grad_(True)
    x0 = torch.tensor([[0.95, 0.05, 0.0]], device=device)
    lamT = torch.tensor([[0.0, args.AT, 0.0]], device=device)
    history = []

    for it in range(args.iters):
        opt.zero_grad()
        t.grad = None
        x, lam, u = state(t), costate(t), control(t)
        dxdt = time_derivative(x, t)
        dlamdt = time_derivative(lam, t)
        f = f_state(x, u, args.beta, args.gamma)
        H = hamiltonian(x, u, lam, args.A, args.B, args.beta, args.gamma)
        # Independent networks give the displayed Hamiltonian partial derivatives.
        Hx = torch.autograd.grad(H.sum(), x, create_graph=True, retain_graph=True)[0]
        Hu = torch.autograd.grad(H.sum(), u, create_graph=True, retain_graph=True)[0]
        residual_u = projected_control_residual(u, Hu, args.umax, args.B)
        loss_state = torch.mean((dxdt - f) ** 2)
        loss_costate = torch.mean((dlamdt + Hx) ** 2)
        loss_stationarity = torch.mean(residual_u**2)
        loss_ic = torch.mean((state(torch.zeros(1, 1, device=device)) - x0) ** 2)
        loss_terminal = torch.mean((costate(torch.tensor([[args.T]], device=device)) - lamT) ** 2)
        loss = (
            args.w_state * loss_state
            + args.w_costate * loss_costate
            + args.w_stat * loss_stationarity
            + args.w_bc * (loss_ic + loss_terminal)
        )
        loss.backward()
        opt.step()
        if it % args.log_every == 0 or it == args.iters - 1:
            row = {
                "iteration": it,
                "loss": float(loss.detach().item()),
                "state_loss": float(loss_state.detach().item()),
                "costate_loss": float(loss_costate.detach().item()),
                "stationarity_loss": float(loss_stationarity.detach().item()),
                "boundary_loss": float((loss_ic + loss_terminal).detach().item()),
            }
            history.append(row)
            LOGGER.info(
                "step=%d loss=%.3g projected_control=%.3g",
                it,
                row["loss"],
                row["stationarity_loss"],
            )
    return (
        (state, costate, control, history)
        if getattr(args, "return_history", False)
        else (state, costate, control)
    )
