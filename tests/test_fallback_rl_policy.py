from dspy_agent.rl.fallback_rl import SimplePolicy, RLConfig


def test_policy_prefers_coding_actions():
    cfg = RLConfig(epsilon=1.0, epsilon_decay=1.0)
    policy = SimplePolicy(cfg)
    actions = [
        'shell_ls',
        'shell_pwd',
        'patch',
        'run_tests',
        'shell_cd',
    ]
    # Force exploration branch multiple times; should never pick shell_ls/pwd/cd
    for _ in range(20):
        choice = policy.select_action({}, actions)
        assert choice in {'patch', 'run_tests'}, choice
