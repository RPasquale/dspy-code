from dspy_agent.dbkit import RedDBStorage


def test_reddb_storage_health_in_memory():
    st = RedDBStorage(url=None, namespace="testns")
    hc = st.health_check()
    assert isinstance(hc, dict)
    assert hc.get("backend") == "reddb"
    assert hc.get("namespace") == "testns"
    assert hc.get("mode") == "memory"
    assert hc.get("ok") is True
