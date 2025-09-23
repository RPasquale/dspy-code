DSPy Agent Production Guide
===========================

This document outlines how to deploy, scale, and validate the DSPy Agent for production. It assumes:
- Python 3.10+
- Node 18+
- Optional: Docker + Docker Compose
- Optional: Ollama (local LLM) or an OpenAI-compatible endpoint
- Optional: Kafka + RedDB for streaming telemetry + durable storage

1) Core Services and Environment
- LLM provider:
  - Ollama: OLLAMA_URL=http://localhost:11434 and ensure model (qwen3:1.7b) is pulled.
  - OpenAI-compatible: set OPENAI_API_KEY and OPENAI_API_BASE.
- RedDB storage:
  - Set REDDB_URL to a persistent RedDB instance; otherwise the agent falls back to in-memory storage.
- Kafka (optional but recommended for streaming metrics):
  - Ensure KAFKA_BOOTSTRAP_SERVERS is accessible.

2) Frontend + Dashboard
- Build the React dashboard:
  - cd frontend/react-dashboard && npm install && npm run build
- Start the dashboard server:
  - python -m enhanced_dashboard_server
- Open http://localhost:8080/dashboard
  - Use Sweeps tab to run native RL sweeps (eprotein/ecarbs)
  - Use Advanced Learning / Signatures to monitor teleprompt training

3) Lightweight Stack (Docker)
- Initialize:
  - dspy-agent lightweight_init --workspace /path/to/ws
- Start:
  - dspy-agent lightweight_up
- Logs:
  - docker compose -f docker/lightweight/docker-compose.yml logs -f dspy-agent
- Stop:
  - dspy-agent lightweight_down

4) Performance Profiles
- CLI supports --profile fast|balanced|maxquality for codectx/edit/start/chat.
- fast → minimal latency; balanced → default; maxquality → highest reasoning fidelity.

5) Training + Learning
- Teleprompt training:
  - Build datasets: dspy-agent dataset --workspace WS --logs WS/logs --out WS/.dspy_data --split
  - Run suite: dspy-agent teleprompt_suite --modules all --methods bootstrap --dataset-dir WS/.dspy_data/splits --shots 8
- RL sweeps:
  - Native EProtein/ECarbs: dspy-agent rl sweep --workspace WS --method eprotein --iterations 20
  - Persisted best at WS/.dspy/rl/best.json and sweep state at WS/.dspy/rl/sweep_state.json

6) Auto-Training (lightweight CLI)
- Start interactive agent (auto training enabled via UI): dspy-agent code
- Environment toggles:
  - DSPY_AUTO_SWEEP_METHOD=eprotein
  - DSPY_AUTO_SWEEP_ITERS=6
  - DSPY_AUTO_TRAIN_INTERVAL_SEC=1800

7) Data Retention and Tracking (RedDB)
- Persist signature metrics, actions, training metrics via EnhancedDataManager.
- Set REDDB_URL and REDDB_NAMESPACE (defaults to dspy).
- Backup strategy: snapshot RedDB data volume daily; export .dspy_reports for audit/history.

8) Informesh + Embeddings
- Follow docs/infermesh_integration.md to enable Infermesh embedding worker + Kafka topics.
- Build embedding index: dspy-agent emb-index --workspace WS --model <embedding-model>.
- Use /api/infermesh/stream to monitor the worker.

9) Load + Soak Testing
- Quick smoke: ./scripts/smoke_test.sh
- Stress cycles: ITERATIONS=5 ./scripts/stress_test.sh
- RL sweeps soak: run dashboard RL sweep with 50+ iterations, then inspect /api/rl/sweep/state Pareto and /api/rl/sweep/history.

10) Observability
- Dashboard pages:
  - Overview: top metrics
  - Advanced Learning: signature performance, training progress
  - Sweeps: RL sweep controls, state, Pareto, history
  - Bus: streaming + DLQ metrics
- Logs: ./logs and Docker logs; .dspy_reports for training artifacts.

11) Hardening Checklist
- [ ] REDDB_URL configured and reachable; regular snapshots
- [ ] LLM endpoints healthy (Ollama /api/tags or OpenAI base)
- [ ] Kafka reachable and topics created (if streaming used)
- [ ] Frontend build served by enhanced_dashboard_server
- [ ] Profiles tuned: codectx/edit use --profile balanced (default), escalate to maxquality for critical actions
- [ ] RL sweeps produce best.json and updated .dspy_rl.json
- [ ] Teleprompt suite run against realistic datasets
- [ ] Guardrails enabled; approval mode set appropriately

12) Troubleshooting
- Dashboard unreachable → ensure npm run build completed; check server logs.
- LM timeouts → reduce beam, enable speculative with a small draft model, check Ollama model availability.
- RL sweep state missing → run at least one sweep; confirm write perms to .dspy/rl.
- RedDB fallback → check REDDB_URL env and network connectivity.

