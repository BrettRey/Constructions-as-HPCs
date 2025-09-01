#!/bin/bash

# runner.sh -- build the figures and tables for the PHOIBLE HPC study
#
# This script orchestrates the full reproducible workflow for the
# phoneme inventory analysis. Running it will download the PHOIBLE
# and Glottolog source data, process the inventories into clean
# summary tables, fit the logistic models, and produce the final
# figures in the `figs/` directory.  It is intended to be invoked
# from the root of the repository.  A fixed random seed is set
# inside the R scripts to ensure reproducibility.

set -euo pipefail

# ensure directories exist
mkdir -p data/raw data/processed out figs

# run each step in order; Rscript will stop with a non‑zero exit
# code if any step fails
echo "[runner] downloading raw data…"
python3 src/01_download_phoible.py

echo "[runner] processing data and generating ridgeline plot…"
python3 src/02_make_ridgelines.py

echo "[runner] fitting logistic models and producing P(/y/) vs vowel size plot…"
python3 src/03_model_y.py

echo "[runner] recording session information…"
python3 - <<'PY'
import importlib
import sys
import platform
import pkg_resources
with open('SESSION.txt', 'w') as f:
    f.write(f"Python version: {platform.python_version()}\n")
    f.write(f"Platform: {platform.platform()}\n")
    # List key packages and versions
    for pkg in ['pandas', 'numpy', 'matplotlib', 'scipy', 'statsmodels', 'sklearn']:
        try:
            module = importlib.import_module(pkg)
            f.write(f"{pkg}=={module.__version__}\n")
        except Exception:
            f.write(f"{pkg} not installed\n")
    f.write('Random seed: 20250101\n')
PY

echo "[runner] workflow complete"
exit 0