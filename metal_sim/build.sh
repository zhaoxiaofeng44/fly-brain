#!/bin/bash
set -e
cd "$(dirname "$0")"
xcrun -sdk macosx clang++ -O3 -std=c++17 -framework Metal -framework Foundation \
    metal_sim.mm -o metal_sim
echo "built metal_sim"
