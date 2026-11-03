"""Mathematical properties used by the worked neural-control examples."""

import numpy as np
import torch

from cyberpinn.configs import PIDLConfig, PMPConfig
from cyberpinn.pidl import CorrectionNet, known_rhs, train as train_pidl
from cyberpinn.pmp import hamiltonian, projected_control_residual, train as train_pmp


def test_sir_hamiltonian_partials_match_the_lecture():
    x = torch.tensor([[0.8, 0.15, 0.05], [0.5, 0.3, 0.2]], requires_grad=True)
    u = torch.tensor([[0.2], [0.4]], requires_grad=True)
    lam = torch.tensor([[1.0, 2.0, 0.5], [-0.2, 0.7, 0.3]], requires_grad=True)
    beta, gamma, q, r = 0.8, 0.2, 10.0, 2.0
    H = hamiltonian(x, u, lam, q, r, beta, gamma)
    hx, hu = torch.autograd.grad(H.sum(), (x, u), create_graph=True)
    s, infected = x[:, 0:1], x[:, 1:2]
    ls, li, lr = lam[:, 0:1], lam[:, 1:2], lam[:, 2:3]
    expected_hx = torch.cat(
        [
            beta * infected * (li - ls) + u * (lr - ls),
            q + beta * s * (li - ls) + gamma * (lr - li),
            torch.zeros_like(s),
        ],
        dim=1,
    )
    assert torch.allclose(hx, expected_hx)
    assert torch.allclose(hu, r * u + s * (lr - ls))


def test_projected_condition_accepts_both_bounds_and_an_interior_minimum():
    control = torch.tensor([[0.0], [1.0], [0.4]])
    derivative = torch.tensor([[2.0], [-2.0], [0.0]])
    residual = projected_control_residual(control, derivative, 1.0, 2.0)
    assert torch.equal(residual, torch.zeros_like(residual))
    wrong_sign = projected_control_residual(control[:2], -derivative[:2], 1.0, 2.0)
    assert torch.all(wrong_sign.abs() > 0)


def test_projected_residual_trains_the_live_control_graph():
    u = torch.tensor([[0.3]], requires_grad=True)
    hu = 2.0 * u - 1.0
    loss = projected_control_residual(u, hu, 1.0, 2.0).square().mean()
    loss.backward()
    assert torch.isfinite(u.grad).all()
    assert u.grad.abs().sum() > 0


def test_pidl_flux_preserves_nonnegative_compartment_boundaries():
    states = torch.tensor([[0.0, 0.4, 0.6], [0.7, 0.0, 0.3], [0.8, 0.2, 0.0]])
    correction = CorrectionNet(width=8, depth=1)
    flux = correction(states)
    assert torch.equal(flux[:2], torch.zeros_like(flux[:2]))
    assert flux[2].item() >= 0
    rhs = known_rhs(states, 0.8, 0.2) + flux @ torch.tensor([[-1.0, 1.0, 0.0]])
    assert torch.allclose(rhs.sum(1), torch.zeros(3), atol=1e-7)
    assert rhs[0, 0] >= 0 and rhs[1, 1] >= 0 and rhs[2, 2] >= 0
    flux.sum().backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in correction.parameters())


def test_pmp_and_pidl_examples_train_with_the_stated_conditions():
    pmp = PMPConfig(iters=3, width=8, depth=1, n_collocation=10, log_every=1, device="cpu")
    pidl = PIDLConfig(
        iters=3, width=8, depth=1, n_data=6, n_collocation=10, log_every=1, device="cpu"
    )
    for result in (train_pmp(pmp), train_pidl(pidl)):
        history = result[-1]
        assert history[-1]["iteration"] == 2
        assert all(np.isfinite(row["loss"]) for row in history)
