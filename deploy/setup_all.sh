#!/usr/bin/env bash
# One-shot environment setup (macOS, Apple Silicon).
#
#   1. CPU whole-brain sim (upstream Brian2 model)   -> .venv (python 3.11, numpy<2)
#   2. Metal GPU whole-brain sim                     -> metal_sim/metal_sim binary
#   3. Embodied playground (flygym + MuJoCo)         -> .venv-body (python 3.12)
#
# Prereqs: uv (https://docs.astral.sh/uv/), Xcode command line tools
# (clang + Metal framework). The connectome parquet must already exist at
# data/2025_Connectivity_783.parquet (distributed with the upstream project).
set -euo pipefail
cd "$(dirname "$0")/.."

echo "== [1/3] CPU brain sim (.venv, python 3.11, numpy<2) =="
if [ ! -x .venv/bin/python ]; then
    uv venv .venv --python 3.11
    uv pip install --python .venv/bin/python "numpy<2" brian2 pyarrow
else
    echo "  .venv exists, skip"
fi

echo "== [2/3] Metal GPU whole-brain simulator =="
if [ ! -x metal_sim/metal_sim ]; then
    (cd metal_sim && ./build.sh)
else
    echo "  metal_sim binary exists, skip (rebuild: cd metal_sim && ./build.sh)"
fi
if [ ! -f metal_sim/data/meta.json ]; then
    echo "  exporting connectome -> metal_sim/data/ (~116 MB, one-off)"
    .venv/bin/python metal_sim/export_connectivity.py
fi

echo "== [3/3] Embodied playground (.venv-body, python 3.12) =="
if [ ! -x .venv-body/bin/python ]; then
    uv venv .venv-body --python 3.12
    uv pip install --python .venv-body/bin/python "flygym==2.1.0" pyarrow
else
    echo "  .venv-body exists, skip"
fi

cat <<'EOF'

Done.
  - Brain sim (CPU):   .venv/bin/python main.py --help
  - Brain sim (Metal): deploy/run_brain_sim.sh
  - Playgrounds:       deploy/run_playground.sh            (interactive window)
                       deploy/run_headless_verify.sh 40    (headless smoke test)
EOF
