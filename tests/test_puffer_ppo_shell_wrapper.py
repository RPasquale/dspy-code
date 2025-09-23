import types


def test_puffer_shell_delegates(monkeypatch):
    # Patch rlkit.run_puffer_ppo so we don't require pufferlib
    called = types.SimpleNamespace(args=None, kwargs=None)
    def fake_run(**kwargs):
        called.kwargs = kwargs
        return "ok"
    import dspy_agent.rl.rlkit as rlkit
    monkeypatch.setattr(rlkit, 'run_puffer_ppo', fake_run, raising=True)

    from dspy_agent.rl import puffer_ppo_shell

    def make_env():
        return object()

    out = puffer_ppo_shell.run_puffer_ppo(make_env, n_envs=3, total_steps=123)
    assert out == "ok"
    assert called.kwargs == {"make_env": make_env, "n_envs": 3, "total_steps": 123}

