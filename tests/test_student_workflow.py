"""Test paired observations, parameter fitting and the introductory experiment."""

import numpy as np
import torch

from cyberpinn.cli import build_parser, fit_ode, ode_trajectory, run_example
from cyberpinn.inverse import observation_data


def test_measurement_noise_does_not_change_truth_or_global_random_state():
    random_state = torch.random.get_rng_state().clone()
    time, truth, indices, clean = observation_data(12, noise=0.0, seed=9)
    other_time, other_truth, other_indices, noisy = observation_data(12, noise=0.02, seed=9)
    assert torch.equal(time, other_time)
    assert torch.equal(truth, other_truth)
    assert torch.equal(indices, other_indices)
    assert not torch.equal(clean, noisy)
    assert torch.equal(torch.random.get_rng_state(), random_state)
    assert torch.equal(noisy, observation_data(12, noise=0.02, seed=9)[3])


def test_ode_fitting_recovers_the_clean_generating_rates():
    time, truth, indices, observed = observation_data(16)
    grid = time[:, 0].numpy().astype(float)
    beta, gamma = fit_ode(grid[indices.numpy()], observed[:, 0].numpy())
    assert np.allclose([beta, gamma], [0.8, 0.2], rtol=0.01)
    replay = ode_trajectory(beta, gamma, grid)
    assert np.allclose(replay.sum(axis=1), 1.0)
    assert np.max(np.abs(replay - truth.numpy())) < 0.002


def test_first_example_runs_with_a_small_training_budget(tmp_path):
    args = build_parser().parse_args(["--iters", "4", "--observations", "8"])
    args.output_dir = tmp_path
    comparison = run_example(args)
    assert len(comparison) == 3
    assert np.isfinite(comparison.select_dtypes(include="number")).all().all()
    assert set(path.name for path in tmp_path.iterdir()) == {
        "observations.csv",
        "trajectories.csv",
        "comparison.csv",
        "training.csv",
        "fit.png",
        "learning.png",
    }
