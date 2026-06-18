# Copyright (c) 2026 Luxing Yang.
# Licensed under the MIT License. See LICENSE in the repository root.

from pathlib import Path
import sys
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from control_pinn_malware import ControlNet, StateNet, rhs
from experiment_profiles import describe_profiles, get_profile
from inverse_pinn_sir_malware import generate_data
from pidl_unknown_mechanism import generate
from pmp_informed_pinn_malware import hamiltonian


class PinnComponentTests(unittest.TestCase):
    def test_inverse_data_generation_shape_and_mass(self):
        t, x = generate_data(n_grid=40)

        self.assertEqual(t.shape, (40, 1))
        self.assertEqual(x.shape, (40, 3))
        self.assertTrue(torch.allclose(x.sum(dim=1), torch.ones(40), atol=1e-5))

    def test_pidl_data_generation_shape_and_mass(self):
        t, x = generate(n=40)

        self.assertEqual(t.shape, (40, 1))
        self.assertEqual(x.shape, (40, 3))
        self.assertTrue(torch.allclose(x.sum(dim=1), torch.ones(40), atol=1e-5))

    def test_control_network_outputs_are_bounded(self):
        t = torch.linspace(0.0, 1.0, 8).view(-1, 1)
        state = StateNet(width=8)
        control = ControlNet(width=8, umax=0.7)
        x = state(t)
        u = control(t)

        self.assertTrue(torch.allclose(x.sum(dim=1), torch.ones(8), atol=1e-6))
        self.assertTrue(torch.all((u >= 0.0) & (u <= 0.7)))
        self.assertEqual(rhs(x, u, beta=0.8, gamma=0.2).shape, (8, 3))

    def test_hamiltonian_stationarity_keeps_live_gradients(self):
        x = torch.tensor([[0.9, 0.1, 0.0]], requires_grad=True)
        u = torch.tensor([[0.2]], requires_grad=True)
        lam = torch.tensor([[1.0, 2.0, 0.5]], requires_grad=True)
        H = hamiltonian(x, u, lam, A=10.0, B=1.0, beta=0.8, gamma=0.2)
        Hu = torch.autograd.grad(H.sum(), u, create_graph=True)[0]
        loss = Hu.pow(2).mean()
        loss.backward()

        self.assertIsNotNone(u.grad)
        self.assertTrue(torch.isfinite(u.grad).all())

    def test_experiment_profiles_are_readable_extension_entries(self):
        profile = get_profile("pmp-informed-pinn")
        rows = describe_profiles()

        self.assertIn("hamiltonian", profile.first_functions_to_edit)
        self.assertIn("stationarity_loss", profile.key_losses)
        self.assertIn(("learning rate", "1e-3"), profile.hyperparameters)
        self.assertTrue(any(row["name"] == "direct-control-pinn" for row in rows))


if __name__ == "__main__":
    unittest.main()
