"""Bounded, reproducible experiment profiles for physics-informed methods."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import time
from typing import Any

from cybercontrol.experiments import run_provenance
from cybercontrol.io import write_csv, write_json

from .configs import (
    ControlConfig,
    InverseConfig,
    NodeInverseTrainConfig,
    NodeSIPSDataConfig,
    PIDLConfig,
    PMPConfig,
)
from .control import train as train_control
from .evaluation import evaluation_mask_rows, training_result_parameter_count
from .inverse import train as train_inverse
from .node_inverse import evaluate_factorized_transfer, train as train_node_inverse
from .pidl import train as train_pidl
from .pmp import train as train_pmp

ROOT = Path(__file__).resolve().parents[2]
ExperimentRow = dict[str, Any]


@dataclass(frozen=True)
class MediumProfile:
    """Work limits and data regimes for the documented medium run."""

    seeds: tuple[int, ...] = (31, 43, 59)
    aggregate_iterations: int = 150
    node_iterations: int = 300
    noise_levels: tuple[float, ...] = (0.0, 0.02)
    observed_node_counts: tuple[int, ...] = (4, 3)
    architectures: tuple[str, ...] = ("dense", "factorized")
    transfer_nodes: int = 10


def _aggregate_configs(
    seed: int, device: str, profile: MediumProfile
) -> tuple[tuple[str, object], ...]:
    common = {
        "iters": profile.aggregate_iterations,
        "n_collocation": 40,
        "width": 24,
        "log_every": 20,
        "seed": seed,
        "device": device,
    }
    return (
        ("inverse_pinn", InverseConfig(n_data=16, depth=3, **common)),
        ("pidl", PIDLConfig(n_data=16, depth=2, **common)),
        ("direct_control", ControlConfig(depth=2, **common)),
        ("pmp_informed", PMPConfig(depth=2, **common)),
    )


def _train_aggregate(method: str, config: object) -> tuple[Any, ...]:
    if method == "inverse_pinn":
        return train_inverse(config)
    if method == "pidl":
        return train_pidl(config)
    if method == "direct_control":
        return train_control(config)
    return train_pmp(config)


def _run_aggregate(seed: int, device: str, profile: MediumProfile) -> list[ExperimentRow]:
    rows: list[ExperimentRow] = []
    for method, config in _aggregate_configs(seed, device, profile):
        started = time.perf_counter()
        result = _train_aggregate(method, config)
        history = result[-1]
        model = result[0]
        row = dict(history[-1])
        row.update(
            {
                "method": method,
                "seed": seed,
                "profile": "medium",
                "training_runtime_seconds": time.perf_counter() - started,
                "network_parameters": training_result_parameter_count(
                    result,
                    auxiliary_parameters=(2 if method in {"inverse_pinn", "pidl"} else 0),
                ),
                "resolved_device": str(next(model.parameters()).device),
            }
        )
        rows.append(row)
    return rows


def _truth_data_config(truth_config: dict[str, Any]) -> NodeSIPSDataConfig:
    keys = asdict(NodeSIPSDataConfig()).keys()
    return NodeSIPSDataConfig(**{key: truth_config[key] for key in keys})


def _run_node_studies(
    seed: int,
    device: str,
    profile: MediumProfile,
) -> tuple[list[ExperimentRow], list[ExperimentRow]]:
    rows: list[ExperimentRow] = []
    mask_rows: list[ExperimentRow] = []
    for noise in profile.noise_levels:
        for observed_nodes in profile.observed_node_counts:
            for architecture in profile.architectures:
                config = NodeInverseTrainConfig(
                    nodes=8,
                    communities=2,
                    grid=41,
                    observed_nodes=observed_nodes,
                    observed_times=10,
                    collocation=24,
                    iters=profile.node_iterations,
                    width=24,
                    depth=2,
                    noise=noise,
                    seed=seed,
                    device=device,
                    log_every=40,
                    architecture=architecture,
                )
                started = time.perf_counter()
                model, history, truth_config = train_node_inverse(config)
                row = dict(history[-1])
                row.update(
                    {
                        "method": "node_sips_inverse",
                        "seed": seed,
                        "profile": "medium",
                        "training_runtime_seconds": time.perf_counter() - started,
                        "resolved_device": str(next(model.parameters()).device),
                    }
                )
                rows.append(row)
                if architecture == "dense":
                    mask_rows.extend(
                        evaluation_mask_rows(
                            nodes=config.nodes,
                            grid=config.grid,
                            horizon=float(truth_config["horizon"]),
                            observed_node_indices=truth_config["observed_node_indices"],
                            observed_time_indices=truth_config["observed_time_indices"],
                            seed=seed,
                            noise_std=noise,
                            observed_node_count=observed_nodes,
                        )
                    )
                if architecture == "factorized":
                    transfer = evaluate_factorized_transfer(
                        model,
                        _truth_data_config(truth_config),
                        nodes=profile.transfer_nodes,
                        seed=1_000 + seed,
                    )
                    transfer.update(
                        {
                            "method": "node_sips_inverse_transfer",
                            "training_seed": seed,
                            "noise_std": noise,
                            "observed_nodes": observed_nodes,
                        }
                    )
                    rows.append(transfer)
    return rows, mask_rows


def run_medium(
    output_dir: Path, device: str, profile: MediumProfile | None = None
) -> list[ExperimentRow]:
    """Run the documented three-seed inverse and control study."""

    profile = profile or MediumProfile()
    output_dir.mkdir(parents=True, exist_ok=True)
    total_started = time.perf_counter()
    rows: list[ExperimentRow] = []
    mask_rows: list[ExperimentRow] = []
    for seed in profile.seeds:
        rows.extend(_run_aggregate(seed, device, profile))
        node_rows, node_masks = _run_node_studies(seed, device, profile)
        rows.extend(node_rows)
        mask_rows.extend(node_masks)

    write_csv(output_dir / "medium_metrics.csv", rows)
    write_csv(output_dir / "evaluation_masks.csv", mask_rows)
    write_json(
        output_dir / "medium_manifest.json",
        {
            "seeds": list(profile.seeds),
            "device": device,
            "resolved_devices": sorted(
                {str(row["resolved_device"]) for row in rows if "resolved_device" in row}
            ),
            "total_runtime_seconds": time.perf_counter() - total_started,
            **run_provenance(ROOT),
            "aggregate_iterations": profile.aggregate_iterations,
            "node_iterations": profile.node_iterations,
            "node_architectures": list(profile.architectures),
            "noise_levels": list(profile.noise_levels),
            "observation_counts": list(profile.observed_node_counts),
            "factorial_noise_by_observation_count": True,
            "unseen_nodes": profile.transfer_nodes,
            "evaluation_masks": "evaluation_masks.csv",
        },
    )
    return rows
