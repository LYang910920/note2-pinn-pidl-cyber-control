"""
Copyright (c) 2026 Luxing Yang.
Licensed under the MIT License. See LICENSE in the repository root.

PMP-informed PINN for controlled malware propagation.

Compared with control_pinn_malware.py, this script trains state, costate,
and control networks using residuals of the PMP optimality system:
    state residual          x' - f(x,u) = 0
    costate residual        lambda' + H_x = 0
    stationarity residual   H_u = 0 for interior controls
    boundary residual       x(0)=x0, lambda(T)=terminal gradient

The example is compact but shows how PMP is explicitly connected
to a neural loss.  This is different from ordinary RL, which may use only the
same simulator and reward.
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
    f = f_state(x, u, beta, gamma)
    running = A * x[:, 1:2] + 0.5 * B * u * u
    return running + torch.sum(lam * f, dim=1, keepdim=True)


def train(args):
    """Train state, costate, and control networks against PMP residuals.

    This needs the Hamiltonian ingredients, initial/terminal boundary
    conditions, and collocation points.  It produces neural approximations of
    `x(t)`, `lambda(t)`, and `u(t)`, plus separate residual diagnostics.
    """
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
    t = torch.linspace(0, args.T, args.n_collocation).view(-1, 1).to(device)
    t.requires_grad_(True)
    x0 = torch.tensor([[0.95, 0.05, 0.0]], device=device)
    lamT = torch.tensor([[0.0, args.AT, 0.0]], device=device)
    history = []

    for it in range(args.iters):
        opt.zero_grad()
        x = state(t)
        lam = costate(t)
        u = control(t)
        dxdt = time_derivative(x, t)
        dlamdt = time_derivative(lam, t)
        f = f_state(x, u, args.beta, args.gamma)
        H = hamiltonian(x, u, lam, args.A, args.B, args.beta, args.gamma)
        # H_x and H_u by autodiff.  Compute them on the live graph so the
        # PMP residuals train the state, costate, and control networks.
        Hx = torch.autograd.grad(H.sum(), x, create_graph=True, retain_graph=True)[0]
        Hu = torch.autograd.grad(H.sum(), u, create_graph=True, retain_graph=True)[0]
        # Interior stationarity Hu=0.  With sigmoid control this is approximate;
        # use a projected/KKT residual in research models where controls sit on
        # bounds for a large part of the horizon.
        loss_state = torch.mean((dxdt - f) ** 2)
        loss_costate = torch.mean((dlamdt + Hx) ** 2)
        loss_stationarity = torch.mean(Hu**2)
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
                f"it={it:05d}, loss={row['loss']:.2e}, "
                f"state={row['state_loss']:.2e}, "
                f"costate={row['costate_loss']:.2e}, "
                f"stat={row['stationarity_loss']:.2e}"
            )
    if getattr(args, "return_history", False):
        return state, costate, control, history
    return state, costate, control
