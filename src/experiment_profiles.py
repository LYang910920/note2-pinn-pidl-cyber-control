# Copyright (c) 2026 Luxing Yang.
# Licensed under the MIT License. See LICENSE in the repository root.

"""Named experiment profiles for adapting Note 2 code.

The four tutorial scripts are small, but students still need a
clear map from method to model ingredients.  This module provides that map in
code form so it can be printed, tested, copied, or imported by longer paper
experiments.
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
        script="src/inverse_pinn_sir_malware.py",
        method="inverse PINN",
        question="Can sparse infected observations identify hidden states and rates?",
        state_level="aggregate S/I/R compartments",
        key_losses=("data_loss", "initial_condition_loss", "ode_loss"),
        first_functions_to_edit=("generate_data", "sir_rhs", "StateNet"),
        paper_extension="Add partial observations, noise models, time-varying rates, or graph-level hidden states.",
        quick_args=("--smoke",),
        hyperparameters=(
            ("long-run iters", "600 in scripts/run_training_iterations.py; CLI default is 5000"),
            ("width/depth", "24/2 in diagnostics; CLI default 64/4"),
            ("n_data", "16 in diagnostics; CLI default 30"),
            ("n_collocation", "70 in diagnostics; CLI default 200"),
            ("learning rate", "1e-3"),
            ("loss weights", "w_ic=10.0, w_ode=1.0"),
            ("seed", "21 in diagnostics; CLI default 1"),
            ("GPU profile", "width/depth=128/5, n_data=40, n_collocation=400, iters=2000"),
        ),
    ),
    "pidl-missing-mechanism": NeuralControlProfile(
        name="pidl-missing-mechanism",
        script="src/pidl_unknown_mechanism.py",
        method="PIDL",
        question="Can a neural correction learn the unknown part of the dynamics?",
        state_level="aggregate S/I/R compartments with learned correction",
        key_losses=("data_loss", "residual_loss", "correction_regularizer"),
        first_functions_to_edit=("known_rhs", "CorrectionNet", "train"),
        paper_extension="Move the correction term to node features, degree classes, or unmodeled attacker adaptation.",
        quick_args=("--smoke",),
        hyperparameters=(
            ("long-run iters", "600 in scripts/run_training_iterations.py; CLI default is 5000"),
            ("width", "24 in diagnostics; CLI default 64"),
            ("n_data", "18 in diagnostics; CLI default 40"),
            ("n_collocation", "70 in diagnostics; CLI default 200"),
            ("learning rate", "1e-3"),
            ("loss weights", "w_ic=10.0, w_res=1.0, w_corr=1e-3"),
            ("seed", "22 in diagnostics; CLI default 2"),
            ("GPU profile", "width/depth=128/4, n_data=50, n_collocation=400, iters=2000"),
        ),
    ),
    "direct-control-pinn": NeuralControlProfile(
        name="direct-control-pinn",
        script="src/control_pinn_malware.py",
        method="direct neural optimal control",
        question="Can state and control networks minimize the objective while satisfying the ODE?",
        state_level="aggregate S/I/R compartments plus open-loop u(t)",
        key_losses=("objective", "residual_loss", "initial_condition_loss"),
        first_functions_to_edit=("rhs", "StateNet", "ControlNet", "train"),
        paper_extension="Add multiple controls, budget states, path constraints, or scenario-dependent objectives.",
        quick_args=("--smoke",),
        hyperparameters=(
            ("long-run iters", "600 in scripts/run_training_iterations.py; CLI default is 5000"),
            ("width", "24 in diagnostics; CLI default 64"),
            ("n_collocation", "70 in diagnostics; CLI default 200"),
            ("learning rate", "1e-3"),
            ("dynamics", "T=20.0, beta=0.8, gamma=0.2"),
            ("control/objective", "umax=1.0, A=10.0, B=1.0, AT=10.0"),
            ("loss weights", "w_res=10.0, w_ic=10.0"),
            ("seed", "23 in diagnostics; CLI default 3"),
            ("GPU profile", "width/depth=128/4, n_collocation=400, iters=2000"),
        ),
    ),
    "pmp-informed-pinn": NeuralControlProfile(
        name="pmp-informed-pinn",
        script="src/pmp_informed_pinn_malware.py",
        method="PMP-informed PINN",
        question="Can neural state/costate/control satisfy the optimality system?",
        state_level="aggregate S/I/R compartments, lambda(t), and u(t)",
        key_losses=("state_loss", "costate_loss", "stationarity_loss", "boundary_loss"),
        first_functions_to_edit=("f_state", "hamiltonian", "train"),
        paper_extension="Replace the Hamiltonian with paper-specific dynamics, constraints, and terminal costs.",
        quick_args=("--smoke",),
        hyperparameters=(
            ("long-run iters", "600 in scripts/run_training_iterations.py; CLI default is 5000"),
            ("width", "24 in diagnostics; CLI default 64"),
            ("n_collocation", "70 in diagnostics; CLI default 200"),
            ("learning rate", "1e-3"),
            ("dynamics", "T=20.0, beta=0.8, gamma=0.2"),
            ("control/objective", "umax=1.0, A=10.0, B=1.0, AT=10.0"),
            ("loss weights", "w_state=10.0, w_costate=1.0, w_stat=1.0, w_bc=10.0"),
            ("seed", "24 in diagnostics; CLI default 4"),
            ("GPU profile", "width/depth=128/4, n_collocation=400, iters=2000"),
        ),
    ),
    "node-siprs-inverse-pinn": NeuralControlProfile(
        name="node-siprs-inverse-pinn",
        script="src/node_siprs_inverse_pinn.py",
        method="node-level inverse PINN",
        question="Can sparse node/time infected observations recover hidden graph SIPRS states and rates?",
        state_level="node-level SIPRS compartments on a small graph",
        key_losses=("data_loss", "initial_condition_loss", "residual_loss", "heldout_state_mse"),
        first_functions_to_edit=("generate_truth", "toy_adjacency", "NodeSIPRSStateNet", "train"),
        paper_extension="Add node features, graph encoders, multiple graph seeds, noise/sparsity ablations, and held-out graph sizes.",
        quick_args=("--smoke", "--device", "cpu"),
        hyperparameters=(
            ("smoke command", "python src/node_siprs_inverse_pinn.py --smoke --device cpu"),
            ("default nodes/grid", "8 nodes, 61 time points"),
            ("observations", "4 nodes and 14 time points, infected compartment only"),
            ("collocation", "32 points by default"),
            ("learning rate", "1e-3"),
            ("loss weights", "w_ic=10.0, w_residual=1.0, w_mass=1.0"),
            ("seed", "31"),
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
            "hyperparameters": "; ".join(f"{key}={value}" for key, value in profile.hyperparameters),
            "quick_command": profile.quick_command(),
        }
        for profile in PROFILES.values()
    ]


if __name__ == "__main__":
    for row in describe_profiles():
        print(
            f"{row['name']}: method={row['method']}, state={row['state_level']}, "
            f"losses={row['key_losses']}, edit={row['first_functions_to_edit']}, "
            f"hyperparameters={row['hyperparameters']}, run={row['quick_command']}"
        )
