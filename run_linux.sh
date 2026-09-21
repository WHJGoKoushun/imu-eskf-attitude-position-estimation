#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
# Python 3.12 是已验证版本；Linux 本身尚未完成实测。
version_check='import sys
print("Python", sys.version.split()[0])
if not (3, 12) <= sys.version_info[:2] < (4, 0):
    sys.exit("Locked dependencies require Python >=3.12,<4. Python 3.12 is recommended.")
if sys.version_info[:2] != (3, 12):
    print("WARNING: only Python 3.12 has been validated; trying this version with all checks enabled.")'
if [[ ! -x .venv/bin/python ]]; then
  if command -v python3.12 >/dev/null 2>&1; then
    bootstrap_python=python3.12
  elif command -v python3 >/dev/null 2>&1; then
    bootstrap_python=python3
  else
    printf '%s\n' 'Python >=3.12,<4 with venv support is required. Python 3.12 is recommended.' >&2
    exit 1
  fi
  if ! "$bootstrap_python" -c "$version_check"; then
    printf '%s\n' 'This Python cannot install the locked dependencies. Install Python 3.12.' >&2
    exit 1
  fi
  "$bootstrap_python" -m venv .venv
fi
if ! .venv/bin/python -c "$version_check"; then
  printf '%s\n' 'Preserve the old .venv by renaming it, then recreate it with Python 3.12.' >&2
  exit 1
fi
.venv/bin/python -m pip install --disable-pip-version-check -r requirements.txt
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
.venv/bin/python validate.py --output results/validation
.venv/bin/python validate_pipeline.py --output results/validation/pipeline_checks.json
.venv/bin/python run.py --comparisons
.venv/bin/python build_report.py
printf '%s\n' 'Complete: REPORT.pdf and results/figures/'
