"""
Copyright (c) 2026 Luxing Yang.
Licensed under the MIT License. See LICENSE in the repository root.

Direct neural-control PINN for a malware optimal-control problem.

This script does not use costates.  It trains a state network x_theta(t) and an
open-loop control network u_phi(t) by minimizing:
    objective + state residual + initial condition + mass constraint.

Direct control PINN is easy to implement, but it is a direct optimization
method. For stronger connection to PMP, see ``cyberpinn.pmp``.
"""

from __future__ import annotations

import logging
import torch

from cybercontrol.models import controlled_sir_rhs_torch as rhs
from cybercontrol.torch_utils import (
    BoundedControlNet,
    SimplexStateNet,
    configure_torch,
    time_derivative,
)

StateNet = SimplexStateNet
ControlNet = BoundedControlNet


LOGGER = logging.getLogger(__name__)


def train(args):
    """Train state and control networks by direct optimal-control loss.

    The optimizer needs collocation points, an initial condition, malware
    dynamics, and cost weights.  It produces a state trajectory model and an
    open-loop neural control `u(t)`.
    """
    _, device, _ = configure_torch(
        seed=args.seed,
        device=getattr(args, "device", "auto"),
        threads=getattr(args, "threads", 1),
    )
    depth = getattr(args, "depth", 2)
    state = StateNet(width=args.width, depth=depth).to(device)
    control = ControlNet(width=args.width, depth=depth, umax=args.umax).to(device)
    opt = torch.optim.Adam(list(state.parameters()) + list(control.parameters()), lr=args.lr)
    t = torch.linspace(0, args.T, args.n_collocation).view(-1, 1).to(device)
    t.requires_grad_(True)
    x0 = torch.tensor([[0.95, 0.05, 0.0]], device=device)
    history = []

    for it in range(args.iters):
        opt.zero_grad()
        x = state(t)
        u = control(t)
        dxdt = time_derivative(x, t)
        loss_res = torch.mean((dxdt - rhs(x, u, args.beta, args.gamma)) ** 2)
        loss_ic = torch.mean((state(torch.zeros(1, 1, device=device)) - x0) ** 2)
        running = args.A * x[:, 1:2] + 0.5 * args.B * u * u
        terminal_infected = state(torch.tensor([[args.T]], device=device))[:, 1].mean()
        loss_obj = args.T * torch.mean(running) + args.AT * terminal_infected
        loss = args.w_res * loss_res + args.w_ic * loss_ic + loss_obj
        loss.backward()
        opt.step()
        if it % args.log_every == 0 or it == args.iters - 1:
            row = {
                "iteration": it,
                "loss": float(loss.detach().item()),
                "objective": float(loss_obj.detach().item()),
                "residual_loss": float(loss_res.detach().item()),
                "initial_condition_loss": float(loss_ic.detach().item()),
                "mean_control": float(u.mean().detach().item()),
            }
            history.append(row)
            LOGGER.info(
                f"it={it:05d}, loss={row['loss']:.3e}, "
                f"obj={row['objective']:.3e}, res={row['residual_loss']:.3e}, "
                f"mean_u={row['mean_control']:.3f}"
            )
    if getattr(args, "return_history", False):
        return state, control, history
    return state, control
