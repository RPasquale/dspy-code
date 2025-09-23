#!/usr/bin/env bash
set -euo pipefail

# DSPy Agent stress/load test harness
# - Exercises dataset bootstrap, teleprompt training, RL sweeps, and codectx
# - Designed to run in developer/lab environments; tune ITERATIONS for longer runs

ITERATIONS=${ITERATIONS:-3}
WORKSPACE=${WORKSPACE:-"$(pwd)"}
LOGS=${LOGS:-"$WORKSPACE/logs"}
MODEL=${MODEL:-"qwen3:1.7b"}
OLLAMA=${OLLAMA:-1}

export DSPY_WORKSPACE="$WORKSPACE"
mkdir -p "$LOGS"

echo "== Stress test start: ITERATIONS=$ITERATIONS WS=$WORKSPACE =="

for i in $(seq 1 "$ITERATIONS"); do
  echo "-- Iteration $i/$ITERATIONS --"

  echo "[A] Dataset bootstrap"
  dspy-agent dataset --workspace "$WORKSPACE" --logs "$LOGS" --out "$WORKSPACE/.dspy_data" --split --dedup || true

  echo "[B] Code context snapshot (no-LM)"
  dspy-agent codectx --path "$WORKSPACE" --workspace "$WORKSPACE" --use-lm false || true

  if [ "$OLLAMA" != "0" ]; then
    echo "[C] Teleprompt suite (small shots)"
    dspy-agent teleprompt_suite --modules codectx,task --methods bootstrap --dataset-dir "$WORKSPACE/.dspy_data/splits" --shots 2 --save-best-dir "$WORKSPACE/.dspy_prompts" --ollama --model "$MODEL" || true
  fi

  echo "[D] Native RL sweep (eprotein, short)"
  dspy-agent rl sweep --workspace "$WORKSPACE" --iterations 2 --method eprotein --persist "$WORKSPACE/.dspy/rl/best.json" --no-update-config || true

  echo "[E] Inspect sweep state"
  test -f "$WORKSPACE/.dspy/rl/sweep_state.json" && tail -n +1 "$WORKSPACE/.dspy/rl/sweep_state.json" | head -n 50 || echo "(no sweep state)"

done

echo "== Stress test completed =="

