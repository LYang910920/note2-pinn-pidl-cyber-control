#!/usr/bin/env bash
# Copyright (c) 2026 Luxing Yang.
# Licensed under the MIT License. See LICENSE in the repository root.

set -euo pipefail

PYTHON_BIN="${PYTHON:-python}"

"${PYTHON_BIN}" src/inverse_pinn_sir_malware.py --smoke
"${PYTHON_BIN}" src/pidl_unknown_mechanism.py --smoke
"${PYTHON_BIN}" src/control_pinn_malware.py --smoke
"${PYTHON_BIN}" src/pmp_informed_pinn_malware.py --smoke
"${PYTHON_BIN}" src/node_siprs_inverse_pinn.py --smoke --device cpu
"${PYTHON_BIN}" src/experiment_profiles.py
"${PYTHON_BIN}" -m unittest discover -s tests
