#!/usr/bin/env bash
set -euo pipefail

# Long-running RL sweeps stress test

WORKSPACE=${WORKSPACE:-"$(pwd)"}
METHOD=${METHOD:-"eprotein"}
ITER=${ITER:-50}
STEPS=${STEPS:-0}

echo "[stress_sweeps] WS=$WORKSPACE METHOD=$METHOD ITER=$ITER"
dspy-agent rl sweep --workspace "$WORKSPACE" --iterations "$ITER" --method "$METHOD" ${STEPS:+--trainer-steps "$STEPS"} --persist "$WORKSPACE/.dspy/rl/best.json" || true
echo "[stress_sweeps] Done. Inspect .dspy/rl/sweep_state.json and .dspy/rl/best.json"

