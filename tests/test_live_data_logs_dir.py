from dspy_agent.server.live_data import WorkspaceAnalyzer


def _clear_env(monkeypatch):
    monkeypatch.delenv("DSPY_LOGS", raising=False)
    monkeypatch.delenv("LOGS_DIR", raising=False)


def test_logs_dir_env_override(tmp_path, monkeypatch):
    _clear_env(monkeypatch)
    env_dir = tmp_path / "custom_logs"
    monkeypatch.setenv("DSPY_LOGS", str(env_dir))

    analyzer = WorkspaceAnalyzer(str(tmp_path))

    assert analyzer.logs_dir == env_dir.resolve()
    assert analyzer.logs_dir.exists()


def test_logs_dir_defaults_to_workspace(tmp_path, monkeypatch):
    _clear_env(monkeypatch)
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    analyzer = WorkspaceAnalyzer(str(workspace))

    assert analyzer.logs_dir == (workspace / "logs").resolve()
    assert analyzer.logs_dir.exists()


def test_logs_dir_falls_back_to_cwd(tmp_path, monkeypatch):
    _clear_env(monkeypatch)
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    blocking_file = workspace / "logs"
    blocking_file.write_text("conflict")

    fallback_root = tmp_path / "fallback"
    fallback_root.mkdir()
    monkeypatch.chdir(fallback_root)

    analyzer = WorkspaceAnalyzer(str(workspace))

    assert analyzer.logs_dir == (fallback_root / "logs").resolve()
    assert analyzer.logs_dir.exists()

def test_ingest_request_schema_alias():
    from dspy_agent.server.fastapi_backend import IngestRequest

    payload = {"foo": "bar"}
    req = IngestRequest(schema=payload)

    assert req.schema == payload
    assert req.schema_payload == payload
    assert req.dict()["schema"] == payload
