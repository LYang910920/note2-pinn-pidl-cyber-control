# Copyright (c) 2026 Luxing Yang.
# Licensed under the MIT License. See LICENSE in the repository root.

import unittest

import numpy as np
import torch

from control_pinn_malware import ControlNet, StateNet, rhs
from experiment_profiles import describe_profiles, get_profile
from inverse_pinn_sir_malware import generate_data
from node_sips_inverse_pinn import NodeSIPSInverseConfig, generate_truth, train as train_node_sips
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

    def test_node_sips_truth_generation_shape_and_mass(self):
        cfg = NodeSIPSInverseConfig(nodes=6, communities=2, grid=16)
        t, x, A, community, params = generate_truth(cfg)

        self.assertEqual(t.shape, (16,))
        self.assertEqual(x.shape, (16, 6, 3))
        self.assertEqual(A.shape, (6, 6))
        self.assertEqual(community.shape, (6,))
        self.assertGreater(float(params.resolve(6).susceptibility.max()), float(params.resolve(6).susceptibility.min()))
        self.assertTrue(np.allclose(x.sum(axis=-1), 1.0, atol=1e-8))

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

    def test_node_sips_inverse_pinn_smoke_metrics(self):
        class Args:
            nodes = 5
            communities = 2
            grid = 15
            observed_nodes = 3
            observed_times = 6
            collocation = 8
            iters = 2
            width = 8
            depth = 2
            noise = 0.0
            lr = 1e-3
            w_ic = 10.0
            w_residual = 1.0
            w_mass = 1.0
            w_param_reg = 1e-3
            device = "cpu"
            seed = 12
            heterogeneity_strength = 0.25
            log_every = 1
            return_history = True

        _, history, cfg = train_node_sips(Args())
        iterations = [row["iteration"] for row in history]
        self.assertEqual(cfg["nodes"], 5)
        self.assertGreaterEqual(len(history), 1)
        self.assertEqual(len(iterations), len(set(iterations)))
        self.assertIn("heldout_state_mse", history[-1])
        self.assertIn("heldout_node_state_mse", history[-1])
        self.assertIn("homogeneous_misspec_state_mse", history[-1])
        self.assertIn("susceptibility_rmse", history[-1])
        self.assertGreater(history[-1]["homogeneous_misspec_state_mse"], 0.0)
        self.assertLess(history[-1]["mass_error"], 1e-6)

    def test_experiment_profiles_are_readable_extension_entries(self):
        profile = get_profile("pmp-informed-pinn")
        rows = describe_profiles()

        self.assertIn("hamiltonian", profile.first_functions_to_edit)
        self.assertIn("stationarity_loss", profile.key_losses)
        self.assertIn(("learning rate", "1e-3"), profile.hyperparameters)
        self.assertTrue(any(row["name"] == "direct-control-pinn" for row in rows))


if __name__ == "__main__":
    unittest.main()
