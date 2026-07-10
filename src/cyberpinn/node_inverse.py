"""Inverse PINN for heterogeneous node-level SIPS graph dynamics.

The dense baseline maps time to a fixed ``N x 3`` state. The factorized path
combines a time encoder with shared node features and can be evaluated on a
different node count. Both paths enforce ``[S,I,P]`` mass with a softmax.

Susceptibility and infectivity have a global multiplicative gauge. Training
therefore normalizes the geometric mean of infectivity to one and reports the
effective edge-transmission matrix instead of treating two unconstrained RMSEs
as independently identifiable physical quantities.
"""

from __future__ import annotations

from dataclasses import asdict, replace
import logging

import numpy as np
import torch

from cybercontrol.network_models import (
    NodeSIPSParams,
    node_sips_rhs_torch,
)
from cybercontrol.nn import parameter_count
from cybercontrol.pinn import time_derivative
from cybercontrol.torch_utils import configure_torch

from .architectures import (
    ARCHITECTURE_REGISTRY,
    CommunityRateHead,
    FactorizedNodeTimeStateNet,
    build_node_features,
    matched_factorized_width,
)
from .configs import NodeSIPSDataConfig
from .node_problem import (
    effective_transmission,
    generate_truth,
    observed_node_indices,
    rollout_known_params,
    truth_gauge,
)

LOGGER = logging.getLogger(__name__)


def _make_model(args, initial, adjacency, community, criticality, observed_nodes, device):
    node_features = build_node_features(
        initial,
        adjacency,
        community,
        criticality,
        observed_nodes,
    )
    architecture = getattr(args, "architecture", "dense")
    if architecture == "dense":
        model = ARCHITECTURE_REGISTRY.build(
            "dense",
            nodes=args.nodes,
            width=args.width,
            depth=args.depth,
        ).to(device)
        resolved_width = int(args.width)
        matched_target = parameter_count(model)
    elif architecture == "factorized":
        resolved_width, matched_target, _ = matched_factorized_width(
            nodes=args.nodes,
            dense_width=args.width,
            dense_depth=args.depth,
            node_feature_dim=node_features.shape[1],
            graph_layers=getattr(args, "graph_layers", 0),
        )
        model = ARCHITECTURE_REGISTRY.build(
            "factorized",
            nodes=args.nodes,
            width=resolved_width,
            depth=args.depth,
            node_features=torch.as_tensor(node_features, device=device),
            adjacency=torch.as_tensor(adjacency, dtype=torch.float32, device=device),
            graph_layers=getattr(args, "graph_layers", 0),
        ).to(device)
    else:
        raise ValueError("architecture must be dense or factorized")
    architecture_summary = ARCHITECTURE_REGISTRY.describe(
        architecture,
        model,
        configuration={
            "width": resolved_width,
            "depth": int(args.depth),
            "graph_layers": int(getattr(args, "graph_layers", 0)),
            "fourier_features": 0,
        },
    )
    return model, node_features, resolved_width, matched_target, architecture_summary


def train(args):
    """Train a dense or factorized inverse PINN and return diagnostics."""

    requested_device = resolve_node_inverse_device(getattr(args, "device", "auto"))
    torch_module, device, _ = configure_torch(
        seed=args.seed,
        device=requested_device,
        threads=getattr(args, "threads", 1),
    )
    cfg = NodeSIPSDataConfig(
        nodes=args.nodes,
        communities=args.communities,
        grid=args.grid,
        observed_nodes=args.observed_nodes,
        observed_times=args.observed_times,
        noise=args.noise,
        seed=args.seed,
        heterogeneity_strength=args.heterogeneity_strength,
    )
    time_np, truth, adjacency_np, community_np, truth_params = generate_truth(cfg)
    rng = np.random.default_rng(cfg.seed)
    observed_nodes = observed_node_indices(cfg.nodes, cfg.observed_nodes, cfg.seed)
    unobserved_nodes = np.setdiff1d(np.arange(cfg.nodes), observed_nodes)
    data_index = np.linspace(0, cfg.grid - 1, cfg.observed_times, dtype=int)
    heldout_time = np.setdiff1d(np.arange(cfg.grid), data_index)[:: max(1, cfg.grid // 12)]
    observed_time_after_initial = data_index[data_index != 0]
    truth_resolved = truth_params.resolve(cfg.nodes)
    homogeneous_path = rollout_known_params(
        cfg,
        adjacency_np,
        truth[0],
        truth_resolved.matched_mean(),
    )
    homogeneous_misspecification_mse = float(np.mean((homogeneous_path - truth) ** 2))
    homogeneous_temporal_mse = (
        float(np.mean((homogeneous_path[heldout_time] - truth[heldout_time]) ** 2))
        if len(heldout_time)
        else float("nan")
    )
    homogeneous_node_mse = (
        float(
            np.mean(
                (
                    homogeneous_path[np.ix_(observed_time_after_initial, unobserved_nodes)]
                    - truth[np.ix_(observed_time_after_initial, unobserved_nodes)]
                )
                ** 2
            )
        )
        if len(unobserved_nodes) and len(observed_time_after_initial)
        else float("nan")
    )
    homogeneous_joint_mse = (
        float(
            np.mean(
                (
                    homogeneous_path[np.ix_(heldout_time, unobserved_nodes)]
                    - truth[np.ix_(heldout_time, unobserved_nodes)]
                )
                ** 2
            )
        )
        if len(unobserved_nodes) and len(heldout_time)
        else float("nan")
    )
    observed_infected = truth[np.ix_(data_index, observed_nodes, [1])].squeeze(-1)
    if cfg.noise > 0:
        full_noise = rng.normal(0.0, cfg.noise, (len(data_index), cfg.nodes))
        observed_infected = np.clip(
            observed_infected + full_noise[:, observed_nodes],
            0.0,
            1.0,
        )

    model, node_features, resolved_width, matched_target, architecture_summary = _make_model(
        args,
        truth[0],
        adjacency_np,
        community_np,
        truth_resolved.criticality,
        observed_nodes,
        device,
    )
    rate_head = CommunityRateHead(cfg.communities).to(device)
    optimizer = torch_module.optim.Adam(
        list(model.parameters()) + list(rate_head.parameters()),
        lr=args.lr,
    )
    data_time = torch.tensor(time_np[data_index, None], dtype=torch.float32, device=device)
    data_target = torch.tensor(observed_infected, dtype=torch.float32, device=device)
    collocation_time = torch.linspace(0.0, cfg.horizon, args.collocation, device=device).reshape(
        -1, 1
    )
    collocation_time.requires_grad_(True)
    initial = torch.tensor(truth[0], dtype=torch.float32, device=device)
    adjacency = torch.tensor(adjacency_np, dtype=torch.float32, device=device)
    community = torch.tensor(community_np, dtype=torch.long, device=device)
    observed_nodes_tensor = torch.tensor(observed_nodes, dtype=torch.long, device=device)
    truth_sus, truth_inf, truth_gamma = truth_gauge(truth_params, community_np)
    truth_effective = effective_transmission(
        cfg.beta_true,
        truth_sus[community_np],
        truth_inf[community_np],
        adjacency_np,
    )
    history: list[dict[str, float | int | str]] = []

    for iteration in range(args.iters):
        optimizer.zero_grad()
        susceptibility_group, infectivity_group, gamma_group = rate_head()
        susceptibility = susceptibility_group[community]
        infectivity = infectivity_group[community]
        gamma = gamma_group[community]
        data_prediction = model(data_time)[:, observed_nodes_tensor, 1]
        data_loss = (data_prediction - data_target).square().mean()
        initial_loss = (model(torch.zeros(1, 1, device=device))[0] - initial).square().mean()
        collocation_state = model(collocation_time)
        derivative = time_derivative(collocation_state, collocation_time)
        params = NodeSIPSParams(
            beta=cfg.beta_true,
            susceptibility=susceptibility,
            infectivity=infectivity,
            gamma=gamma,
            omega=cfg.omega,
            patch_efficacy=truth_resolved.patch_efficacy,
            clean_efficacy=truth_resolved.clean_efficacy,
        )
        rhs = torch.stack(
            [
                node_sips_rhs_torch(
                    collocation_state[index],
                    adjacency,
                    params,
                    patch=cfg.patch,
                    clean=cfg.clean,
                )
                for index in range(collocation_state.shape[0])
            ]
        )
        residual_loss = (derivative - rhs).square().mean()
        parameter_regularizer = (susceptibility_group - 1.0).square().mean()
        loss = (
            data_loss
            + args.w_ic * initial_loss
            + args.w_residual * residual_loss
            + args.w_param_reg * parameter_regularizer
        )
        loss.backward()
        optimizer.step()

        if iteration % args.log_every == 0 or iteration == args.iters - 1:
            with torch.no_grad():
                susceptibility_group, infectivity_group, gamma_group = rate_head()
                all_time_tensor = torch.tensor(time_np[:, None], dtype=torch.float32, device=device)
                all_prediction = model(all_time_tensor).cpu().numpy()
                temporal_mse = (
                    float(np.mean((all_prediction[heldout_time] - truth[heldout_time]) ** 2))
                    if len(heldout_time)
                    else float("nan")
                )
                node_holdout_mse = (
                    float(
                        np.mean(
                            (
                                all_prediction[
                                    np.ix_(observed_time_after_initial, unobserved_nodes)
                                ]
                                - truth[np.ix_(observed_time_after_initial, unobserved_nodes)]
                            )
                            ** 2
                        )
                    )
                    if len(unobserved_nodes) and len(observed_time_after_initial)
                    else float("nan")
                )
                joint_holdout_mse = (
                    float(
                        np.mean(
                            (
                                all_prediction[np.ix_(heldout_time, unobserved_nodes)]
                                - truth[np.ix_(heldout_time, unobserved_nodes)]
                            )
                            ** 2
                        )
                    )
                    if len(unobserved_nodes) and len(heldout_time)
                    else float("nan")
                )
                learned_sus = susceptibility_group.cpu().numpy()
                learned_inf = infectivity_group.cpu().numpy()
                learned_gamma = gamma_group.cpu().numpy()
                learned_effective = effective_transmission(
                    cfg.beta_true,
                    learned_sus[community_np],
                    learned_inf[community_np],
                    adjacency_np,
                )
                effective_rmse = float(np.sqrt(np.mean((learned_effective - truth_effective) ** 2)))
                history.append(
                    {
                        "iteration": iteration,
                        "optimizer_loss_before_step": float(loss.detach().cpu()),
                        "data_loss_before_step": float(data_loss.detach().cpu()),
                        "residual_loss_before_step": float(residual_loss.detach().cpu()),
                        "heldout_state_mse": temporal_mse,
                        "temporal_holdout_state_mse": temporal_mse,
                        "heldout_node_state_mse": node_holdout_mse,
                        "unobserved_node_state_mse": node_holdout_mse,
                        "node_holdout_at_observed_times_mse": node_holdout_mse,
                        "joint_node_time_holdout_state_mse": joint_holdout_mse,
                        "homogeneous_misspec_state_mse": homogeneous_misspecification_mse,
                        "homogeneous_known_rate_misspecification_mse": homogeneous_misspecification_mse,
                        "homogeneous_temporal_holdout_state_mse": homogeneous_temporal_mse,
                        "homogeneous_node_holdout_at_observed_times_mse": homogeneous_node_mse,
                        "homogeneous_joint_node_time_holdout_state_mse": homogeneous_joint_mse,
                        "homogeneous_baseline_rule": "node-wise arithmetic mean of resolved truth",
                        "effective_transmission_rmse": effective_rmse,
                        "susceptibility_rmse": float(
                            np.sqrt(np.mean((learned_sus - truth_sus) ** 2))
                        ),
                        "infectivity_rmse": float(np.sqrt(np.mean((learned_inf - truth_inf) ** 2))),
                        "gamma_rmse": float(np.sqrt(np.mean((learned_gamma - truth_gamma) ** 2))),
                        "infectivity_geometric_mean": float(np.exp(np.mean(np.log(learned_inf)))),
                        "mass_error": float(np.max(np.abs(all_prediction.sum(axis=-1) - 1.0))),
                        "architecture": getattr(args, "architecture", "dense"),
                        "resolved_width": resolved_width,
                        "state_parameters": parameter_count(model),
                        "total_parameters": parameter_count(model) + parameter_count(rate_head),
                        "dense_parameter_target": matched_target,
                        "observation_fraction": float(len(observed_nodes) / cfg.nodes),
                        "noise_std": float(cfg.noise),
                        "identifiability_note": "infectivity geometric mean fixed to one",
                        "known_patch_efficacy_min": float(truth_resolved.patch_efficacy.min()),
                        "known_patch_efficacy_max": float(truth_resolved.patch_efficacy.max()),
                        "known_clean_efficacy_min": float(truth_resolved.clean_efficacy.min()),
                        "known_clean_efficacy_max": float(truth_resolved.clean_efficacy.max()),
                        "architecture_activation": str(architecture_summary["activation"]),
                        "architecture_normalization": str(architecture_summary["normalization"]),
                        "architecture_encoder": str(architecture_summary["encoder"]),
                        "architecture_pooling": str(architecture_summary["pooling"]),
                        "architecture_decoder": str(architecture_summary["decoder"]),
                        "architecture_input_shape": str(architecture_summary["input_shape"]),
                        "architecture_output_shape": str(architecture_summary["output_shape"]),
                        "architecture_depth": int(args.depth),
                        "architecture_graph_layers": int(getattr(args, "graph_layers", 0)),
                        "architecture_fourier_features": 0,
                    }
                )
            LOGGER.info(
                "iteration=%05d loss=%.3e temporal_mse=%.2e effective_K_rmse=%.3e",
                iteration,
                history[-1]["optimizer_loss_before_step"],
                history[-1]["temporal_holdout_state_mse"],
                history[-1]["effective_transmission_rmse"],
            )
    if getattr(args, "return_history", False):
        return (
            model,
            history,
            {
                **asdict(cfg),
                "node_feature_dim": node_features.shape[1],
                "observed_node_indices": observed_nodes.tolist(),
                "unobserved_node_indices": unobserved_nodes.tolist(),
                "observed_time_indices": data_index.tolist(),
                "heldout_time_indices": heldout_time.tolist(),
            },
        )
    return model, history


def evaluate_factorized_transfer(
    model: FactorizedNodeTimeStateNet,
    cfg: NodeSIPSDataConfig,
    *,
    nodes: int,
    seed: int,
) -> dict[str, float | int | str]:
    """Evaluate the shared state decoder on an unseen graph size and seed."""

    test_cfg = replace(
        cfg,
        nodes=nodes,
        observed_nodes=min(cfg.observed_nodes, nodes),
        seed=seed,
    )
    time, truth, adjacency, community, params = generate_truth(test_cfg)
    observed_nodes = np.arange(test_cfg.observed_nodes)
    features = build_node_features(
        truth[0],
        adjacency,
        community,
        params.resolve(nodes).criticality,
        observed_nodes,
    )
    device = next(model.parameters()).device
    with torch.no_grad():
        prediction = (
            model.predict_graph(
                torch.tensor(time[:, None], dtype=torch.float32, device=device),
                torch.tensor(features, dtype=torch.float32, device=device),
                torch.tensor(adjacency, dtype=torch.float32, device=device),
            )
            .cpu()
            .numpy()
        )
    return {
        "architecture": "factorized",
        "nodes": nodes,
        "seed": seed,
        "graph_transfer_state_mse": float(np.mean((prediction - truth) ** 2)),
        "mass_error": float(np.max(np.abs(prediction.sum(axis=-1) - 1.0))),
    }


def resolve_node_inverse_device(requested: str) -> str:
    """Use CUDA when available; keep residual-heavy auto runs off unstable MPS."""

    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        LOGGER.warning("auto device resolved to CPU for the higher-order node residual workload")
    return "cpu"
