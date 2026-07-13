"""Single public command-line interface for Note 2."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

from cybercontrol.experiments import write_run_manifest

from .configs import (
    ControlConfig,
    InverseConfig,
    NodeInverseTrainConfig,
    PIDLConfig,
    PMPConfig,
)
from .control import train as train_control
from .experiments import run_medium
from .inverse import train as train_inverse
from .node_inverse import train as train_node_inverse
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
            write_run_manifest(
                Path("artifacts/smoke_summary.json"),
                command="smoke",
                metrics=_smoke(),
                repository_root=ROOT,
            )
        )
    elif args.command == "medium":
        rows = run_medium(args.output_dir, args.device)
        print(f"wrote {len(rows)} rows to {args.output_dir}")
    elif args.command == "figures":
        _run_script("scripts/generate_figures.py")
    elif args.command == "docs":
        _build_docs()
    elif args.command == "all":
        print(
            write_run_manifest(
                Path("artifacts/smoke_summary.json"),
                command="smoke",
                metrics=_smoke(),
                repository_root=ROOT,
            )
        )
        _run_script("scripts/generate_figures.py")
        _build_docs()


if __name__ == "__main__":
    main()
