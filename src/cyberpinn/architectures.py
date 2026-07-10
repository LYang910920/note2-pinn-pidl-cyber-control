"""Dense and scalable node-time architectures for graph PINNs."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from cybercontrol.nn import (
    ArchitectureDescriptor,
    ArchitectureRegistry,
    FactorizedNodeTimeNet,
    MLP,
    parameter_count,
)
from cybercontrol.pinn import positive


class CommunityRateHead(nn.Module):
    """Positive community rates with a unit-geometric-mean infectivity gauge."""

    def __init__(self, communities: int):
        super().__init__()
        self.susceptibility_raw = nn.Parameter(torch.zeros(communities))
        self.infectivity_raw = nn.Parameter(torch.zeros(communities))
        self.gamma_raw = nn.Parameter(torch.full((communities,), -1.6))

    def forward(self):
        susceptibility = positive(self.susceptibility_raw)
        infectivity_positive = positive(self.infectivity_raw)
        geometric_mean = torch.exp(torch.mean(torch.log(infectivity_positive)))
        infectivity = infectivity_positive / geometric_mean
        return susceptibility, infectivity, positive(self.gamma_raw)


class DenseTimeStateNet(nn.Module):
    """Small fixed-graph baseline mapping time to all node compartments."""

    def __init__(self, nodes: int, width: int = 32, depth: int = 2):
        super().__init__()
        self.nodes = int(nodes)
        self.net = MLP(1, nodes * 3, width=width, depth=depth, activation="tanh")

    def forward(self, time):
        logits = self.net(time).reshape(time.shape[0], self.nodes, 3)
        return torch.softmax(logits, dim=-1)


class FactorizedNodeTimeStateNet(nn.Module):
    """Shared node-time decoder without node-ID output weights."""

    def __init__(
        self,
        node_features: torch.Tensor,
        adjacency: torch.Tensor,
        *,
        width: int = 16,
        depth: int = 2,
        graph_layers: int = 0,
    ):
        super().__init__()
        self.register_buffer("node_features", node_features)
        self.register_buffer("adjacency", adjacency)
        self.net = FactorizedNodeTimeNet(
            node_features.shape[1],
            width=width,
            depth=depth,
            fourier_features=0,
            graph_layers=graph_layers,
        )

    def forward(self, time):
        return self.net(time, self.node_features, self.adjacency)

    def predict_graph(self, time, node_features, adjacency):
        """Evaluate the shared decoder on a graph with a different node count."""

        return self.net(time, node_features, adjacency)


def _build_dense_state(
    *,
    nodes: int,
    width: int,
    depth: int,
    node_features=None,
    adjacency=None,
    graph_layers: int = 0,
):
    del node_features, adjacency, graph_layers
    return DenseTimeStateNet(nodes, width, depth)


def _build_factorized_state(
    *,
    nodes: int,
    width: int,
    depth: int,
    node_features,
    adjacency,
    graph_layers: int = 0,
):
    del nodes
    return FactorizedNodeTimeStateNet(
        node_features,
        adjacency,
        width=width,
        depth=depth,
        graph_layers=graph_layers,
    )


ARCHITECTURE_REGISTRY = ArchitectureRegistry()
ARCHITECTURE_REGISTRY.register(
    ArchitectureDescriptor(
        name="dense",
        input_shape="time [batch, 1]",
        output_shape="state [batch, nodes, 3]",
        activation="tanh",
        normalization="softmax over [S,I,P]",
        encoder="fixed-graph time MLP",
        decoder="node-specific 3-state output layer",
    ),
    _build_dense_state,
)
ARCHITECTURE_REGISTRY.register(
    ArchitectureDescriptor(
        name="factorized",
        input_shape="time [batch,1], node features [nodes,f], adjacency [nodes,nodes]",
        output_shape="state [batch, nodes, 3]",
        activation="tanh",
        normalization="softmax over [S,I,P] and normalized adjacency",
        encoder="shared time and node encoders with optional graph layers",
        decoder="shared node-time 3-state decoder",
        pooling="none; node outputs remain explicit",
    ),
    _build_factorized_state,
)


def build_node_features(
    initial_state: np.ndarray,
    adjacency: np.ndarray,
    community: np.ndarray,
    criticality: np.ndarray,
    observed_nodes: np.ndarray,
) -> np.ndarray:
    """Return node features with no node-ID embedding."""

    degree = np.asarray(adjacency > 0.0, dtype=np.float64).sum(axis=1)
    degree = degree / max(float(degree.max()), 1.0)
    community_scale = community / max(float(np.max(community)), 1.0)
    criticality = criticality / max(float(np.mean(criticality)), 1e-12)
    observation_mask = np.zeros(len(community), dtype=np.float64)
    observation_mask[np.asarray(observed_nodes, dtype=int)] = 1.0
    return np.column_stack(
        [initial_state, degree, community_scale, criticality, observation_mask]
    ).astype(np.float32)


def matched_factorized_width(
    *,
    nodes: int,
    dense_width: int,
    dense_depth: int,
    node_feature_dim: int,
    graph_layers: int = 0,
) -> tuple[int, int, int]:
    """Choose the closest factorized width by trainable parameter count."""

    dense = DenseTimeStateNet(nodes, dense_width, dense_depth)
    target = parameter_count(dense)
    features = torch.zeros(nodes, node_feature_dim)
    adjacency = torch.eye(nodes)
    candidates = []
    for width in range(4, 65):
        model = FactorizedNodeTimeStateNet(
            features,
            adjacency,
            width=width,
            depth=dense_depth,
            graph_layers=graph_layers,
        )
        count = parameter_count(model)
        candidates.append((abs(count - target), width, count))
    _, width, count = min(candidates)
    return width, target, count
