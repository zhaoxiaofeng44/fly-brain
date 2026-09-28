#!/usr/bin/env bash
# Headless smoke test: full brain-gated behavioural cycle, no window.
# Usage: deploy/run_headless_verify.sh [sim-seconds]   (default: 40)
set -euo pipefail
cd "$(dirname "$0")/.."

limit="${1:-40}"
HEADLESS=1 "SIM_LIMIT=$limit" FLYDBG=1 PYTHONUNBUFFERED=1 \
    .venv-body/bin/python embodied/interactive_playground_flight.py
