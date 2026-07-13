"""Single public command-line interface for Note 2."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
import subprocess
import sys
import time

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


def _source_root() -> Path:
    """Return the source checkout required for figures and LaTeX commands."""

    if (
        not (ROOT / "scripts" / "generate_figures.py").exists()
        or not (ROOT / "docs" / "source").exists()
    ):
        raise RuntimeError(
            "This command needs a source checkout containing scripts/ and docs/source/. "
            "The installed cyberpinn package can still be imported normally."
        )
    return ROOT


def _smoke() -> dict[str, float | int]:
    inverse = InverseConfig(
        iters=4,
        width=8,
        depth=2,
        n_data=6,
        n_collocation=12,
        log_every=1,
        device="cpu",
    )
    _, _, _, inverse_history = train_inverse(inverse)
    pidl = PIDLConfig(
        iters=4,
        width=8,
        depth=2,
        n_data=6,
        n_collocation=12,
        log_every=1,
        device="cpu",
    )
    _, _, pidl_history = train_pidl(pidl)
    control = ControlConfig(
        iters=4,
        width=8,
        depth=2,
        n_collocation=12,
        log_every=1,
        device="cpu",
    )
    _, _, control_history = train_control(control)
    pmp = PMPConfig(
        iters=4,
        width=8,
        depth=2,
        n_collocation=12,
        log_every=1,
        device="cpu",
    )
    _, _, _, pmp_history = train_pmp(pmp)
    node = NodeInverseTrainConfig(
        nodes=6,
        communities=2,
        grid=17,
        observed_nodes=3,
        observed_times=6,
        collocation=8,
        iters=4,
        width=8,
        depth=2,
        log_every=1,
        device="cpu",
    )
    _, node_history, _ = train_node_inverse(node)
    return {
        "inverse_rows": len(inverse_history),
        "pidl_rows": len(pidl_history),
        "control_rows": len(control_history),
        "pmp_rows": len(pmp_history),
        "node_inverse_rows": len(node_history),
        "node_inverse_mass_error": float(node_history[-1]["mass_error"]),
    }


def _medium(output_dir: Path, device: str) -> list[dict[str, float | int | str]]:
    """Run paired dense/factorized studies plus aggregate method diagnostics."""

    output_dir.mkdir(parents=True, exist_ok=True)
    total_started = time.perf_counter()
    rows: list[dict[str, float | int | str]] = []
    mask_rows: list[dict[str, float | int | str]] = []
    seeds = (31, 43, 59)
    for seed in seeds:
        aggregate_configs = (
            (
                "inverse_pinn",
                InverseConfig(
                    iters=150,
                    n_data=16,
                    n_collocation=40,
                    width=24,
                    depth=3,
                    log_every=20,
                    seed=seed,
                    device=device,
                ),
            ),
            (
                "pidl",
                PIDLConfig(
                    iters=150,
                    n_data=16,
                    n_collocation=40,
                    width=24,
                    depth=2,
                    log_every=20,
                    seed=seed,
                    device=device,
                ),
            ),
            (
                "direct_control",
                ControlConfig(
                    iters=150,
                    n_collocation=40,
                    width=24,
                    depth=2,
                    log_every=20,
                    seed=seed,
                    device=device,
                ),
            ),
            (
                "pmp_informed",
                PMPConfig(
                    iters=150,
                    n_collocation=40,
                    width=24,
                    depth=2,
                    log_every=20,
                    seed=seed,
                    device=device,
                ),
            ),
        )
        for method, config in aggregate_configs:
            started = time.perf_counter()
            if method == "inverse_pinn":
                result = train_inverse(config)
            elif method == "pidl":
                result = train_pidl(config)
            elif method == "direct_control":
                result = train_control(config)
            else:
                result = train_pmp(config)
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

        for noise in (0.0, 0.02):
            for observed_nodes in (4, 3):
                for architecture in ("dense", "factorized"):
                    config = NodeInverseTrainConfig(
                        nodes=8,
                        communities=2,
                        grid=41,
                        observed_nodes=observed_nodes,
                        observed_times=10,
                        collocation=24,
                        iters=300,
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
                            NodeSIPSDataConfig(
                                **{key: truth_config[key] for key in asdict(NodeSIPSDataConfig())}
                            ),
                            nodes=10,
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

    write_csv(output_dir / "medium_metrics.csv", rows)
    write_csv(output_dir / "evaluation_masks.csv", mask_rows)
    write_json(
        output_dir / "medium_manifest.json",
        {
            "seeds": list(seeds),
            "device": device,
            "resolved_devices": sorted(
                {str(row["resolved_device"]) for row in rows if "resolved_device" in row}
            ),
            "total_runtime_seconds": time.perf_counter() - total_started,
            **run_provenance(ROOT),
            "aggregate_iterations": 150,
            "node_iterations": 300,
            "node_architectures": ["dense", "factorized"],
            "noise_levels": [0.0, 0.02],
            "observation_counts": [4, 3],
            "factorial_noise_by_observation_count": True,
            "unseen_nodes": 10,
            "evaluation_masks": "evaluation_masks.csv",
        },
    )
    return rows


def _run_script(path: str) -> None:
    root = _source_root()
    subprocess.run([sys.executable, path], cwd=root, check=True)


def _build_docs() -> None:
    root = _source_root()
    source = root / "docs" / "source"
    subprocess.run(
        ["latexmk", "-pdf", "-interaction=nonstopmode", "note2_pinn_pidl_cyber_control.tex"],
        cwd=source,
        check=True,
    )
    built = source / "note2_pinn_pidl_cyber_control.pdf"
    if built.exists():
        (root / "docs" / built.name).write_bytes(built.read_bytes())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cyberpinn", description="Physics-informed cyber-control experiments."
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("smoke", help="Fast execution and residual-shape checks.")
    medium = sub.add_parser("medium", help="Three-seed architecture and data-regime study.")
    medium.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    medium.add_argument("--output-dir", type=Path, default=Path("artifacts/medium"))
    sub.add_parser("figures", help="Regenerate curated guide figures.")
    sub.add_parser("docs", help="Build the current tutorial PDF.")
    sub.add_parser("all", help="Run smoke, figures, and docs.")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "smoke":
        print(
            write_json(
                Path("artifacts/smoke_summary.json"),
                {"command": "smoke", **run_provenance(ROOT), "metrics": _smoke()},
            )
        )
    elif args.command == "medium":
        rows = _medium(args.output_dir, args.device)
        print(f"wrote {len(rows)} rows to {args.output_dir}")
    elif args.command == "figures":
        _run_script("scripts/generate_figures.py")
    elif args.command == "docs":
        _build_docs()
    elif args.command == "all":
        print(
            write_json(
                Path("artifacts/smoke_summary.json"),
                {"command": "smoke", **run_provenance(ROOT), "metrics": _smoke()},
            )
        )
        _run_script("scripts/generate_figures.py")
        _build_docs()


if __name__ == "__main__":
    main()
