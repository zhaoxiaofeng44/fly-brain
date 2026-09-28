#!/usr/bin/env bash
# Launch an embodied playground with the real GPU brain in the loop.
# Usage: deploy/run_playground.sh [script]   (default: flight playground)
set -euo pipefail
cd "$(dirname "$0")/.."

script="${1:-embodied/interactive_playground_flight.py}"
[ -f "$script" ] || { echo "no such script: $script" >&2; exit 1; }

# uv-built Pythons live outside the system framework; mjpython's app bundle
# needs libpython pointed at the uv prefix, or it dies at startup on macOS.
PYREAL="$(cd "$(dirname "$(readlink ".venv-body/bin/python" 2>/dev/null) \
    || echo .venv-body/bin/python")" && pwd)"
export DYLD_FALLBACK_LIBRARY_PATH="${DYLD_FALLBACK_LIBRARY_PATH:-$PYREAL/../lib}"

exec .venv-body/bin/mjpython "$script"
