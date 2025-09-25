from dspy_agent.rl.fallback_rl import get_fallback_trainer, train_fallback_rl, select_fallback_action, get_fallback_stats

trainer = get_fallback_trainer()
state = {"pass_rate": 0.5}
action = select_fallback_action(state, ["run_tests", "patch", "lint"])
print('action:', action)
for i in range(10):
    # Simulate improving reward when choosing 'patch'
    r = 1.0 if action == 'patch' else 0.2
    out = train_fallback_rl(state, action, r, state, done=(i%3==2))
print('stats:', get_fallback_stats())
print('OK')
