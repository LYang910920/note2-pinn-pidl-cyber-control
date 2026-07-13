"""Typed configurations for PINN/PIDL runners."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class InverseConfig:
    iters: int = 5_000
    width: int = 64
    depth: int = 4
    n_data: int = 30
    n_collocation: int = 200
    noise: float = 0.0
    lr: float = 1e-3
    w_ic: float = 10.0
    w_ode: float = 1.0
    log_every: int = 1_000
    seed: int = 1
    device: str = "auto"
    threads: int = 1
    return_history: bool = True


@dataclass
class PIDLConfig:
    iters: int = 5_000
    n_data: int = 40
    n_collocation: int = 200
    width: int = 64
    depth: int = 2
    lr: float = 1e-3
    w_ic: float = 10.0
    w_res: float = 1.0
    w_corr: float = 1e-3
    log_every: int = 1_000
    seed: int = 2
    device: str = "auto"
    threads: int = 1
    return_history: bool = True


@dataclass
class ControlConfig:
    iters: int = 5_000
    T: float = 20.0
    n_collocation: int = 200
    width: int = 64
    depth: int = 2
    lr: float = 1e-3
    beta: float = 0.8
    gamma: float = 0.2
    umax: float = 1.0
    A: float = 10.0
    B: float = 1.0
    AT: float = 10.0
    w_res: float = 10.0
    w_ic: float = 10.0
    log_every: int = 1_000
    seed: int = 3
    device: str = "auto"
    threads: int = 1
    return_history: bool = True


@dataclass
class PMPConfig:
    iters: int = 5_000
    T: float = 20.0
    n_collocation: int = 200
    width: int = 64
    depth: int = 2
    lr: float = 1e-3
    beta: float = 0.8
    gamma: float = 0.2
    umax: float = 1.0
    A: float = 10.0
    B: float = 1.0
    AT: float = 10.0
    w_state: float = 10.0
    w_costate: float = 1.0
    w_stat: float = 1.0
    w_bc: float = 10.0
    log_every: int = 1_000
    seed: int = 4
    device: str = "auto"
    threads: int = 1
    return_history: bool = True


@dataclass
class NodeInverseTrainConfig:
    nodes: int = 8
    communities: int = 2
    grid: int = 61
    observed_nodes: int = 4
    observed_times: int = 14
    collocation: int = 32
    iters: int = 500
    width: int = 32
    depth: int = 2
    noise: float = 0.0
    lr: float = 1e-3
    w_ic: float = 10.0
    w_residual: float = 1.0
    w_param_reg: float = 1e-3
    device: str = "auto"
    threads: int = 1
    seed: int = 31
    heterogeneity_strength: float = 0.35
    log_every: int = 100
    architecture: str = "dense"
    graph_layers: int = 0
    return_history: bool = True

    def validate(self) -> None:
        if not 1 <= self.communities <= self.nodes:
            raise ValueError("communities must be between one and nodes")
        if not 1 <= self.observed_nodes <= self.nodes:
            raise ValueError("observed_nodes must be between one and nodes")
        if self.architecture not in {"dense", "factorized"}:
            raise ValueError("architecture must be dense or factorized")
        if min(self.grid, self.observed_times, self.collocation, self.iters) <= 1:
            raise ValueError("grid, observed_times, collocation, and iters must exceed one")


@dataclass(frozen=True)
class NodeSIPSDataConfig:
    """Synthetic heterogeneous node-SIPS inverse-problem configuration."""

    nodes: int = 8
    communities: int = 2
    horizon: float = 6.0
    grid: int = 61
    beta_true: float = 0.82
    gamma_true: float = 0.18
    heterogeneity_strength: float = 0.35
    omega: float = 0.03
    patch: float = 0.08
    clean: float = 0.04
    observed_nodes: int = 4
    observed_times: int = 14
    noise: float = 0.0
    seed: int = 31

    def validate(self) -> None:
        if not 1 <= self.communities <= self.nodes:
            raise ValueError("communities must be between one and nodes")
        if not 1 <= self.observed_nodes <= self.nodes:
            raise ValueError("observed_nodes must be between one and nodes")
        if not 2 <= self.observed_times <= self.grid:
            raise ValueError("observed_times must be between two and grid")
