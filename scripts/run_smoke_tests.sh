#!/usr/bin/env bash
# Copyright (c) 2026 Luxing Yang.
# Licensed under the MIT License. See LICENSE in the repository root.

set -euo pipefail

python src/inverse_pinn_sir_malware.py --smoke
python src/pidl_unknown_mechanism.py --smoke
python src/control_pinn_malware.py --smoke
python src/pmp_informed_pinn_malware.py --smoke
python src/experiment_profiles.py
python -m unittest discover -s tests
