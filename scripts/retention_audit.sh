#!/usr/bin/env bash
set -euo pipefail

# Retention audit: snapshot key metrics from dashboard APIs and append JSONL record

API=${API:-"http://localhost:8080"}
OUT_DIR=${OUT_DIR:-".dspy_reports"}
mkdir -p "$OUT_DIR"
TS=$(date +%s)

get() { curl -fsS "$API$1" || echo '{}'; }

SIGN=$(get /api/signatures)
HIST=$(get /api/rl/sweep/history)
TP=$(get /api/teleprompt/experiments)
RED=$(get /api/reddb/summary)
LOGS=$(get /api/logs)

COUNT_SIG=$(echo "$SIGN" | jq '.signatures | length' 2>/dev/null || echo 0)
COUNT_EXP=$(echo "$HIST" | jq '.experiments | length' 2>/dev/null || echo 0)
COUNT_TP=$(echo "$TP" | jq '.experiments | length' 2>/dev/null || echo 0)
COUNT_LOG=$(echo "$LOGS" | jq '.logs | length' 2>/dev/null || echo 0)

COUNT_RA=$(echo "$RED" | jq '.recent_actions' 2>/dev/null || echo 0)
COUNT_RT=$(echo "$RED" | jq '.recent_training' 2>/dev/null || echo 0)

REC=$(jq -n --arg ts "$TS" --argjson sig "$COUNT_SIG" --argjson exp "$COUNT_EXP" --argjson tp "$COUNT_TP" --argjson logs "$COUNT_LOG" --argjson ra "$COUNT_RA" --argjson rt "$COUNT_RT" '{ts: ($ts|tonumber), signatures: $sig, sweep_experiments: $exp, teleprompt_experiments: $tp, logs: $logs, recent_actions: $ra, recent_training: $rt}')
echo "$REC" >> "$OUT_DIR/retention_audit.jsonl"
echo "[retention_audit] $REC"
