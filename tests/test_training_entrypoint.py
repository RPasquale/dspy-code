import sys
from types import SimpleNamespace


def test_training_entrypoint_persists_results(monkeypatch, tmp_path):
    # Monkeypatch detect_toolchain and ToolchainExecutor to avoid running shell
    import dspy_agent.training.entrypoint as ep
    import dspy_agent.rl.rlkit as rl
    called = {'recorded': False}

    def fake_detect_toolchain(ws):
        return SimpleNamespace(workspace=tmp_path, timeout_sec=10, shell_timeout=10, test_cmd=None, lint_cmd=None, build_cmd=None, shell_defaults={})
    class FakeExec:
        def __init__(self, cfg): pass
        def __call__(self, tool, args):
            return rl.AgentResult(metrics={'pass_rate': 1.0, 'blast_radius': 0.0}, info={'doc_id': 'abc123'})

    def fake_bandit_trainer(make_env, cfg, analytics_cb=None):
        # Emulate a single step; call analytics_cb with verifier_scores and doc_id
        if analytics_cb:
            info = {'verifier_scores': {'custom_x': 0.9}, 'doc_id': 'abc123'}
            analytics_cb(0, info, 1.0)
        return rl.EpisodeStats(rewards=[1.0], infos=[{'verifier_scores': {'custom_x': 0.9}}])

    def fake_record(signature_name, scores, reward, **kw):
        called['recorded'] = True
        return 'A1'

    monkeypatch.setattr(rl, 'detect_toolchain', fake_detect_toolchain)
    monkeypatch.setattr(rl, 'ToolchainExecutor', FakeExec)
    monkeypatch.setattr(rl, 'bandit_trainer', fake_bandit_trainer)
    monkeypatch.setattr(rl, 'record_verifier_scores', fake_record)

    # Simulate CLI argv
    argv = [
        'prog',
        '--workspace', str(tmp_path),
        '--signature', 'CodeContextSig',
        '--verifiers', 'dspy_agent.verifiers.custom',
        '--steps', '1',
        '--env', 'development'
    ]
    old_argv = sys.argv
    try:
        sys.argv = argv
        ep.main()
    finally:
        sys.argv = old_argv
    assert called['recorded'], 'expected record_verifier_scores to be called'

