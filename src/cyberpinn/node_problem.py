"""Synthetic heterogeneous node-SIPS inverse problem and rollout helpers."""

from __future__ import annotations

import numpy as np

from cybercontrol.network_models import (
    NodeSIPSParams,
    community_correlated_node_sips_params,
    contiguous_community_index,
    node_sips_rhs_numpy,
    normalize_adjacency,
)
from cybercontrol.numerics import project_compartments, rk4_integrate

from .configs import NodeSIPSDataConfig


def toy_adjacency(nodes: int) -> np.ndarray:
    """Return a row-normalized ring-plus-chord graph."""

    adjacency = np.zeros((nodes, nodes), dtype=np.float64)
    for node in range(nodes):
        adjacency[node, (node - 1) % nodes] = 1.0
        adjacency[node, (node + 1) % nodes] = 1.0
        if nodes > 4:
            adjacency[node, (node + 3) % nodes] = 0.5
    return normalize_adjacency(adjacency)


def observed_node_indices(nodes: int, count: int, seed: int) -> np.ndarray:
    """Return a deterministic nested observation subset for sparsity studies."""

    if not 1 <= count <= nodes:
        raise ValueError("count must be between one and nodes")
    order = np.random.default_rng(seed).permutation(nodes)
    return np.sort(order[:count])


def rollout_known_params(
    cfg: NodeSIPSDataConfig,
    adjacency: np.ndarray,
    initial: np.ndarray,
    params: NodeSIPSParams,
) -> np.ndarray:
    """Roll out one specified SIPS parameterization on the data grid."""

    time = np.linspace(0.0, cfg.horizon, cfg.grid)

    def rhs_flat(values, _time):
        state = values.reshape(cfg.nodes, 3)
        return node_sips_rhs_numpy(
            state,
            adjacency,
            params,
            patch=cfg.patch,
            clean=cfg.clean,
        ).reshape(-1)

    values = project_compartments(initial).reshape(-1)
    path = [values.reshape(cfg.nodes, 3).copy()]
    for index in range(cfg.grid - 1):
        values, _ = rk4_integrate(
            rhs_flat,
            values,
            t0=float(time[index]),
            dt=float(time[index + 1] - time[index]),
            substeps=2,
            project=lambda x: project_compartments(x.reshape(cfg.nodes, 3)).reshape(-1),
        )
        path.append(values.reshape(cfg.nodes, 3).copy())
    return np.asarray(path)


def generate_truth(cfg: NodeSIPSDataConfig):
    """Generate heterogeneous SIPS truth under fixed patch and clean controls."""

    cfg.validate()
    rng = np.random.default_rng(cfg.seed)
    adjacency = toy_adjacency(cfg.nodes)
    community = contiguous_community_index(cfg.nodes, cfg.communities)
    initial = np.zeros((cfg.nodes, 3), dtype=np.float64)
    initial[:, 0] = 0.96
    initial[:, 1] = 0.04
    infected_seeds = rng.choice(cfg.nodes, size=max(1, cfg.nodes // 4), replace=False)
    initial[infected_seeds, 1] += 0.08
    initial[infected_seeds, 0] -= 0.08
    initial = project_compartments(initial)
    params = community_correlated_node_sips_params(
        community,
        strength=cfg.heterogeneity_strength,
        beta=cfg.beta_true,
        gamma=cfg.gamma_true,
        omega=cfg.omega,
    )
    path = rollout_known_params(cfg, adjacency, initial, params)
    return np.linspace(0.0, cfg.horizon, cfg.grid), path, adjacency, community, params


def truth_gauge(params, community: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return community rates under the unit-geometric-mean infectivity gauge."""

    resolved = params.resolve(len(community))
    count = int(community.max()) + 1
    susceptibility = np.asarray(
        [resolved.susceptibility[community == group].mean() for group in range(count)]
    )
    infectivity = np.asarray(
        [resolved.infectivity[community == group].mean() for group in range(count)]
    )
    gamma = np.asarray([resolved.gamma[community == group].mean() for group in range(count)])
    geometric_mean = float(np.exp(np.mean(np.log(infectivity))))
    return susceptibility * geometric_mean, infectivity / geometric_mean, gamma


def effective_transmission(
    beta: float,
    susceptibility: np.ndarray,
    infectivity: np.ndarray,
    adjacency: np.ndarray,
) -> np.ndarray:
    """Return the identifiable edgewise transmission matrix."""

    return beta * susceptibility[:, None] * adjacency * infectivity[None, :]
