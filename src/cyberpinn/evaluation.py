"""Evaluation records shared by the Note 2 command runner.

Training functions retain their compact tuple APIs. These helpers count every
trainable network in such a tuple and construct explicit node/time holdout
records without coupling either concern to CLI orchestration.
"""

from __future__ import annotations

from typing import Any


def training_result_parameter_count(
    result: tuple[Any, ...],
    *,
    auxiliary_parameters: int = 0,
) -> int:
    """Count unique trainable parameters in all returned network modules.

    ``auxiliary_parameters`` records optimized scalar tensors that a legacy
    return tuple exposes only as fitted floats. The inverse-SIR runner uses two
    such scalars for beta and gamma.
    """

    if auxiliary_parameters < 0:
        raise ValueError("auxiliary_parameters must be nonnegative")
    seen: set[int] = set()
    total = int(auxiliary_parameters)
    for item in result:
        parameters = getattr(item, "parameters", None)
        if parameters is None:
            continue
        for parameter in parameters():
            identity = id(parameter)
            if parameter.requires_grad and identity not in seen:
                seen.add(identity)
                total += int(parameter.numel())
    return total


def evaluation_mask_rows(
    *,
    nodes: int,
    grid: int,
    horizon: float,
    observed_node_indices: list[int],
    observed_time_indices: list[int],
    seed: int,
    noise_std: float,
    observed_node_count: int,
) -> list[dict[str, float | int | str]]:
    """Return explicit node/time split records for one inverse experiment."""

    observed_nodes = set(observed_node_indices)
    observed_times = set(observed_time_indices)
    rows: list[dict[str, float | int | str]] = []
    for node in range(nodes):
        rows.append(
            {
                "seed": seed,
                "noise_std": noise_std,
                "observed_node_count": observed_node_count,
                "axis": "node",
                "index": node,
                "coordinate": node,
                "split": (
                    "trajectory_observed"
                    if node in observed_nodes
                    else "trajectory_held_out_after_initial_condition"
                ),
            }
        )
    for index in range(grid):
        rows.append(
            {
                "seed": seed,
                "noise_std": noise_std,
                "observed_node_count": observed_node_count,
                "axis": "time",
                "index": index,
                "coordinate": index * horizon / (grid - 1),
                "split": "observed" if index in observed_times else "held_out",
            }
        )
    return rows
