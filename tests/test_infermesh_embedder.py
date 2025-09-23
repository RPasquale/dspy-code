import json
from types import SimpleNamespace

from dspy_agent.embedding.infermesh import InferMeshEmbedder


class _Resp:
    def __init__(self, payload: dict, code: int = 200):
        self._data = json.dumps(payload).encode('utf-8')
        self._code = code
    def read(self):
        return self._data
    def getcode(self):
        return self._code
    def __enter__(self):
        return self
    def __exit__(self, *args):
        return False


def test_infermesh_embedder_basic(monkeypatch):
    # Mock urllib.request.urlopen
    import urllib.request as _req

    def _fake_urlopen(req, timeout=0):  # noqa: ARG001
        # Accept any request and return 2 small vectors
        return _Resp({'vectors': [[1.0, 0.0], [0.0, 1.0]]}, 200)

    monkeypatch.setattr(_req, 'urlopen', _fake_urlopen)

    emb = InferMeshEmbedder('http://infermesh:9000', 'm-test')
    out = emb.embed(['a', 'b'])
    assert isinstance(out, list)
    assert out and len(out[0]) == 2

