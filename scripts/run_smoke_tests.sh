#!/usr/bin/env bash
set -euo pipefail

python src/inverse_pinn_sir_malware.py --smoke
python src/pidl_unknown_mechanism.py --smoke
python src/control_pinn_malware.py --smoke
python src/pmp_informed_pinn_malware.py --smoke
python -m unittest discover -s tests
