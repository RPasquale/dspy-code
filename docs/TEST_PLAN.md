DSPy Agent Test Plan (CLI + Frontend + Training)

This plan verifies that the agent is usable end-to-end:

1) Prereqs
- Python 3.10+
- Optional: uv (recommended), Docker (for lightweight stack)
- Local LLM (Ollama) if you want LM-powered steps

2) Quick Smoke Test
- Run: `./scripts/smoke_test.sh`
- What it checks:
  - CLI help pages
  - codectx without LM (snapshot-only)
  - dataset bootstrapping into `.dspy_data/splits`
  - teleprompt suite (if LM configured)
  - native RL sweep with eprotein (2 iters), persistence to `.dspy/rl/sweep_state.json`
  - dashboard HTTP endpoints respond

3) Manual CLI Walkthrough
- Context: `dspy-agent ctx --workspace . --logs ./logs`
- Plan: `dspy-agent plan "fix failing test"`
- Edit: `dspy-agent edit "implement X" --speculative --draft-model qwen2:0.5b --beam 3`
- Sweep: `dspy-agent rl sweep --method ecarbs --iterations 10`
- Teleprompt: `dspy-agent teleprompt_suite --modules all --methods bootstrap --dataset-dir .dspy_data/splits --shots 4`

4) Frontend/Dashboard
- Start: `python -m enhanced_dashboard_server` (or via `dspy-agent code --open-dashboard`)
- Open http://localhost:8080/dashboard
- Verify: overview, signatures, optimization history, teleprompt experiment list
- Trigger a teleprompt run: POST `/api/teleprompt/run`

5) Lightweight Stack
- Prepare: `dspy-agent lightweight_init --workspace .`
- Build/Up: `dspy-agent lightweight_up`
- Open dashboard, run agent commands in container logs

6) Persistent Memory & Learning
- Verify `.dspy_session_memory.json` is created and grows with tool runs
- Verify `.dspy/rl/sweep_state.json` is updated after sweeps
- Verify `.dspy_reports/*` and `.dspy_prompts/*` get updated by training

7) Troubleshooting
- If LM missing: most steps still work without LM; LM steps are skipped
- If Docker missing: lightweight commands show manual hints

