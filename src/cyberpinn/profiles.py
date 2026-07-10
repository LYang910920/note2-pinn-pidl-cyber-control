# Copyright (c) 2026 Luxing Yang.
# Licensed under the MIT License. See LICENSE in the repository root.

"""Named experiment profiles for adapting Note 2 code.

The profiles map each method to its model ingredients, losses, editable
functions, and longer-run settings so experiments can be printed, tested,
copied, or imported by paper-level scripts.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class NeuralControlProfile:
    """Readable contract for a PINN/PIDL/PMP-informed experiment."""

    name: str
    script: str
    method: str
    question: str
    state_level: str
    key_losses: tuple[str, ...]
    first_functions_to_edit: tuple[str, ...]
    paper_extension: str
    quick_args: tuple[str, ...]
    hyperparameters: tuple[tuple[str, str], ...]

    def quick_command(self) -> str:
        """Return a shell command for a short run of this profile."""
        args = " ".join(self.quick_args)
        return f"python {self.script} {args}".strip()


PROFILES: dict[str, NeuralControlProfile] = {
    "inverse-pinn-sparse-observation": NeuralControlProfile(
        name="inverse-pinn-sparse-observation",
        script="-m cyberpinn",
        method="inverse PINN",
        question="Can sparse infected observations identify hidden states and rates?",
        state_level="aggregate S/I/R compartments",
        key_losses=("data_loss", "initial_condition_loss", "ode_loss"),
        first_functions_to_edit=("generate_data", "sir_rhs", "StateNet"),
        paper_extension="Add partial observations, noise models, time-varying rates, or graph-level hidden states.",
        quick_args=("smoke",),
        hyperparameters=(
            ("medium iters", "150 per seed"),
            ("width/depth", "24/3"),
            ("n_data", "16"),
            ("n_collocation", "40"),
            ("learning rate", "1e-3"),
            ("loss weights", "w_ic=10.0, w_ode=1.0"),
            ("seeds", "31, 43, 59"),
            ("source", "cyberpinn.inverse and cyberpinn.cli"),
        ),
    ),
    "pidl-missing-mechanism": NeuralControlProfile(
        name="pidl-missing-mechanism",
        script="-m cyberpinn",
        method="PIDL",
        question="Can a neural correction learn the unknown part of the dynamics?",
        state_level="aggregate S/I/R compartments with learned correction",
        key_losses=("data_loss", "residual_loss", "correction_regularizer"),
        first_functions_to_edit=("known_rhs", "CorrectionNet", "train"),
        paper_extension="Move the correction term to node features, degree classes, or unmodeled attacker adaptation.",
        quick_args=("smoke",),
        hyperparameters=(
            ("medium iters", "150 per seed"),
            ("width/depth", "24/2"),
            ("n_data", "16"),
            ("n_collocation", "40"),
            ("learning rate", "1e-3"),
            ("loss weights", "w_ic=10.0, w_res=1.0, w_corr=1e-3"),
            ("seeds", "31, 43, 59"),
            ("source", "cyberpinn.pidl and cyberpinn.cli"),
        ),
    ),
    "direct-control-pinn": NeuralControlProfile(
        name="direct-control-pinn",
        script="-m cyberpinn",
        method="direct neural optimal control",
        question="Can state and control networks minimize the objective while satisfying the ODE?",
        state_level="aggregate S/I/R compartments plus open-loop u(t)",
        key_losses=("objective", "residual_loss", "initial_condition_loss"),
        first_functions_to_edit=("rhs", "StateNet", "ControlNet", "train"),
        paper_extension="Add multiple controls, budget states, path constraints, or scenario-dependent objectives.",
        quick_args=("smoke",),
        hyperparameters=(
            ("medium iters", "150 per seed"),
            ("width/depth", "24/2"),
            ("n_collocation", "40"),
            ("learning rate", "1e-3"),
            ("dynamics", "T=20.0, beta=0.8, gamma=0.2"),
            ("control/objective", "umax=1.0, A=10.0, B=1.0, AT=10.0"),
            ("loss weights", "w_res=10.0, w_ic=10.0"),
            ("seeds", "31, 43, 59"),
            ("source", "cyberpinn.control and cyberpinn.cli"),
        ),
    ),
    "pmp-informed-pinn": NeuralControlProfile(
        name="pmp-informed-pinn",
        script="-m cyberpinn",
        method="PMP-informed PINN",
        question="Can neural state/costate/control satisfy the optimality system?",
        state_level="aggregate S/I/R compartments, lambda(t), and u(t)",
        key_losses=("state_loss", "costate_loss", "stationarity_loss", "boundary_loss"),
        first_functions_to_edit=("f_state", "hamiltonian", "train"),
        paper_extension="Replace the Hamiltonian with paper-specific dynamics, constraints, and terminal costs.",
        quick_args=("smoke",),
        hyperparameters=(
            ("medium iters", "150 per seed"),
            ("width/depth", "24/2"),
            ("n_collocation", "40"),
            ("learning rate", "1e-3"),
            ("dynamics", "T=20.0, beta=0.8, gamma=0.2"),
            ("control/objective", "umax=1.0, A=10.0, B=1.0, AT=10.0"),
            ("loss weights", "w_state=10.0, w_costate=1.0, w_stat=1.0, w_bc=10.0"),
            ("seeds", "31, 43, 59"),
            ("source", "cyberpinn.pmp and cyberpinn.cli"),
        ),
    ),
    "node-sips-inverse-pinn": NeuralControlProfile(
        name="node-sips-inverse-pinn",
        script="-m cyberpinn",
        method="node-level inverse PINN",
        question="Can sparse node/time infected observations recover hidden graph SIPS states and community-specific rates?",
        state_level="node-level SIPS compartments on a small graph",
        key_losses=("data_loss", "initial_condition_loss", "residual_loss", "heldout_state_mse"),
        first_functions_to_edit=(
            "node_problem.generate_truth",
            "node_problem.toy_adjacency",
            "architectures.FactorizedNodeTimeStateNet",
            "node_inverse.train",
        ),
        paper_extension="Add node features, graph encoders, multiple graph seeds, noise/sparsity ablations, and held-out graph sizes.",
        quick_args=("smoke",),
        hyperparameters=(
            ("medium command", "python -m cyberpinn medium --device auto"),
            ("default nodes/grid", "8 nodes, 2 communities, 61 time points"),
            (
                "rate model",
                "positive community-specific susceptibility, infectivity, and gamma; base beta fixed",
            ),
            ("observations", "4 nodes and 14 time points, infected compartment only"),
            ("collocation", "32 points by default"),
            ("medium iters", "300 per architecture and data regime"),
            ("learning rate", "1e-3"),
            ("loss weights", "w_ic=10.0, w_residual=1.0, w_mass=1.0, w_param_reg=1e-3"),
            ("seeds", "31, 43, 59"),
            ("architectures", "dense and factorized at matched parameter budget"),
        ),
    ),
}


def get_profile(name: str) -> NeuralControlProfile:
    """Return one profile with a helpful error if the name is unknown."""
    try:
        return PROFILES[name]
    except KeyError as exc:
        available = ", ".join(sorted(PROFILES))
        raise KeyError(f"unknown profile {name!r}; available: {available}") from exc


def describe_profiles() -> list[dict[str, str]]:
    """Small table-friendly summary used by docs and tests."""
    return [
        {
            "name": profile.name,
            "method": profile.method,
            "state_level": profile.state_level,
            "key_losses": ", ".join(profile.key_losses),
            "first_functions_to_edit": ", ".join(profile.first_functions_to_edit),
            "hyperparameters": "; ".join(
                f"{key}={value}" for key, value in profile.hyperparameters
            ),
            "quick_command": profile.quick_command(),
        }
        for profile in PROFILES.values()
    ]
