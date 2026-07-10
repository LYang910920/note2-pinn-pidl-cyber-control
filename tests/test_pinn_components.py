# Copyright (c) 2026 Luxing Yang.
# Licensed under the MIT License. See LICENSE in the repository root.

import unittest

import numpy as np
import torch

from cyberpinn.architectures import (
    DenseTimeStateNet,
    FactorizedNodeTimeStateNet,
    build_node_features,
    matched_factorized_width,
)
from cyberpinn.control import ControlNet, StateNet, rhs
from cyberpinn.inverse import generate_data
from cyberpinn.configs import NodeSIPSDataConfig
from cyberpinn.configs import InverseConfig, NodeInverseTrainConfig
from cyberpinn.inverse import train as train_inverse
from cyberpinn.node_inverse import (
    evaluate_factorized_transfer,
    resolve_node_inverse_device,
    train as train_node_sips,
)
from cyberpinn.node_problem import generate_truth, observed_node_indices, rollout_known_params
from cyberpinn.pidl import generate
from cyberpinn.pmp import hamiltonian
from cyberpinn.profiles import describe_profiles, get_profile


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
        cfg = NodeSIPSDataConfig(nodes=6, communities=2, grid=16)
        t, x, A, community, params = generate_truth(cfg)

        self.assertEqual(t.shape, (16,))
        self.assertEqual(x.shape, (16, 6, 3))
        self.assertEqual(A.shape, (6, 6))
        self.assertEqual(community.shape, (6,))
        self.assertGreater(
            float(params.resolve(6).susceptibility.max()),
            float(params.resolve(6).susceptibility.min()),
        )
        self.assertTrue(np.allclose(x.sum(axis=-1), 1.0, atol=1e-8))
        observed_three = observed_node_indices(6, 3, cfg.seed)
        observed_four = observed_node_indices(6, 4, cfg.seed)
        self.assertTrue(set(observed_three).issubset(set(observed_four)))
        resolved = params.resolve(cfg.nodes)
        matched = resolved.matched_mean()
        for field in ("susceptibility", "infectivity", "gamma", "patch_efficacy", "clean_efficacy"):
            self.assertTrue(np.allclose(getattr(matched, field), np.mean(getattr(resolved, field))))
        matched_path = rollout_known_params(cfg, A, x[0], matched)
        self.assertGreater(float(np.mean((matched_path - x) ** 2)), 0.0)
        with self.assertRaises(ValueError):
            observed_node_indices(6, 0, cfg.seed)

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
            architecture = "dense"
            graph_layers = 0
            threads = 1

        _, history, cfg = train_node_sips(Args())
        iterations = [row["iteration"] for row in history]
        self.assertEqual(cfg["nodes"], 5)
        self.assertGreaterEqual(len(history), 1)
        self.assertEqual(len(iterations), len(set(iterations)))
        self.assertIn("heldout_state_mse", history[-1])
        self.assertIn("heldout_node_state_mse", history[-1])
        self.assertIn("joint_node_time_holdout_state_mse", history[-1])
        self.assertIn("homogeneous_misspec_state_mse", history[-1])
        self.assertIn("homogeneous_temporal_holdout_state_mse", history[-1])
        self.assertIn("homogeneous_node_holdout_at_observed_times_mse", history[-1])
        self.assertIn("homogeneous_joint_node_time_holdout_state_mse", history[-1])
        self.assertIn("susceptibility_rmse", history[-1])
        self.assertGreater(history[-1]["homogeneous_misspec_state_mse"], 0.0)
        self.assertLess(history[-1]["mass_error"], 1e-6)
        self.assertIn("effective_transmission_rmse", history[-1])
        self.assertAlmostEqual(history[-1]["infectivity_geometric_mean"], 1.0, places=5)
        self.assertEqual(history[-1]["architecture_activation"], "tanh")
        self.assertIn("time", history[-1]["architecture_input_shape"])
        self.assertLess(
            history[-1]["known_patch_efficacy_min"],
            history[-1]["known_patch_efficacy_max"],
        )
        self.assertLess(
            history[-1]["known_clean_efficacy_min"],
            history[-1]["known_clean_efficacy_max"],
        )

    def test_node_inverse_supports_no_temporal_holdout(self):
        config = NodeInverseTrainConfig(
            nodes=5,
            communities=2,
            grid=8,
            observed_nodes=3,
            observed_times=8,
            collocation=8,
            iters=2,
            width=8,
            depth=2,
            log_every=10,
            device="cpu",
        )

        _, history, split = train_node_sips(config)

        self.assertEqual(history[-1]["iteration"], 1)
        self.assertTrue(np.isnan(history[-1]["temporal_holdout_state_mse"]))
        self.assertEqual(split["heldout_time_indices"], [])
        self.assertLess(history[-1]["mass_error"], 1e-6)

    def test_factorized_node_time_shape_and_budget_matching(self):
        cfg = NodeSIPSDataConfig(
            nodes=6,
            communities=2,
            grid=12,
            observed_nodes=3,
            observed_times=6,
        )
        _, truth, adjacency, community, params = generate_truth(cfg)
        features = build_node_features(
            truth[0],
            adjacency,
            community,
            params.resolve(cfg.nodes).criticality,
            np.arange(cfg.observed_nodes),
        )
        width, dense_count, factorized_count = matched_factorized_width(
            nodes=cfg.nodes,
            dense_width=12,
            dense_depth=2,
            node_feature_dim=features.shape[1],
        )
        dense = DenseTimeStateNet(cfg.nodes, 12, 2)
        factorized = FactorizedNodeTimeStateNet(
            torch.tensor(features),
            torch.tensor(adjacency, dtype=torch.float32),
            width=width,
            depth=2,
        )
        time = torch.linspace(0.0, 1.0, 5).reshape(-1, 1)
        self.assertEqual(dense(time).shape, (5, cfg.nodes, 3))
        self.assertEqual(factorized(time).shape, (5, cfg.nodes, 3))
        self.assertLess(abs(factorized_count - dense_count) / dense_count, 0.25)
        self.assertTrue(torch.allclose(factorized(time).sum(dim=-1), torch.ones(5, 6), atol=1e-6))

        transfer = evaluate_factorized_transfer(factorized, cfg, nodes=7, seed=99)
        self.assertTrue(np.isfinite(transfer["graph_transfer_state_mse"]))
        self.assertLess(transfer["mass_error"], 1e-6)

    def test_aggregate_inverse_logs_the_final_optimizer_step(self):
        config = InverseConfig(
            iters=3,
            width=8,
            depth=2,
            n_data=6,
            n_collocation=10,
            log_every=10,
            device="cpu",
        )

        *_, history = train_inverse(config)

        self.assertEqual(history[-1]["iteration"], 2)

    def test_node_sips_inverse_device_resolution_is_explicit(self):
        self.assertEqual(resolve_node_inverse_device("cpu"), "cpu")
        self.assertEqual(resolve_node_inverse_device("cuda"), "cuda")
        self.assertEqual(resolve_node_inverse_device("mps"), "mps")
        self.assertIn(resolve_node_inverse_device("auto"), {"cpu", "cuda"})

    def test_experiment_profiles_are_readable_extension_entries(self):
        profile = get_profile("pmp-informed-pinn")
        rows = describe_profiles()

        self.assertIn("hamiltonian", profile.first_functions_to_edit)
        self.assertIn("stationarity_loss", profile.key_losses)
        self.assertIn(("learning rate", "1e-3"), profile.hyperparameters)
        self.assertTrue(any(row["name"] == "direct-control-pinn" for row in rows))


if __name__ == "__main__":
    unittest.main()
