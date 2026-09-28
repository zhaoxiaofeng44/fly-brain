#!/usr/bin/env bash
# Metal GPU whole-brain simulator: validate, benchmark, or open the viewer.
# Usage: deploy/run_brain_sim.sh [validate|viewer|repl]
set -euo pipefail
cd "$(dirname "$0")/.."

mode="${1:-validate}"
[ -x metal_sim/metal_sim ] || { echo "run deploy/setup_all.sh first" >&2; exit 1; }

case "$mode" in
    validate) .venv/bin/python metal_sim/validate_metal.py ;;
    viewer)   .venv/bin/python metal_sim/brain_viewer.py ;;
    repl)     exec metal_sim/metal_sim metal_sim/data ;;
    *) echo "usage: $0 [validate|viewer|repl]" >&2; exit 1 ;;
esac
