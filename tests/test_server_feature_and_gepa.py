import io
import json
import time
from types import SimpleNamespace

import pytest


class DummyDM:
    def __init__(self, actions=None, metrics=None):
        self._actions = actions or []
        self._metrics = metrics

    def get_recent_actions(self, limit=100):
        return list(self._actions)[:limit]

    def get_signature_metrics(self, name: str):
        return self._metrics


class DummyHandler:
    def __init__(self):
        self.headers = {}
        self._code = None
        self._headers = []
        self.wfile = io.BytesIO()
        self.data_manager = None
        self._reddb = {}

    def send_response(self, code, msg=None):
        self._code = code
    def send_header(self, k, v):
        self._headers.append((k, v))
    def end_headers(self):
        pass

    def _reddb_get(self, key: str):
        return self._reddb.get(key)


def _call(handler, method):
    # Call EnhancedDashboardHandler.method(self)
    import enhanced_dashboard_server as eds
    fn = getattr(eds.EnhancedDashboardHandler, method)
    fn(handler)
    data = handler.wfile.getvalue().decode('utf-8')
    if data.startswith('data: '):
        data = data[len('data: '):]
    try:
        return handler._code or 200, json.loads(data)
    except Exception:
        return handler._code or 200, data


def test_feature_analysis_endpoint_with_mocked_kv():
    import enhanced_dashboard_server as eds
    h = DummyHandler()
    # Two actions with doc_ids belonging to the same signature
    now = time.time()
    a1 = SimpleNamespace(timestamp=now-100, reward=1.0, environment=eds.Environment.DEVELOPMENT,
                         parameters={'signature_name': 'CodeContextSig', 'doc_id': 'd1'}, result={})
    a2 = SimpleNamespace(timestamp=now-50, reward=0.5, environment=eds.Environment.DEVELOPMENT,
                         parameters={'signature_name': 'CodeContextSig', 'doc_id': 'd2'}, result={})
    h.data_manager = DummyDM(actions=[a1, a2])
    # KV returns unit vectors for embvec:doc_id
    h._reddb['embvec:d1'] = {'unit': [1.0, 0.0]}
    h._reddb['embvec:d2'] = {'unit': [0.0, 1.0]}
    h.path = '/api/signature/feature-analysis?name=CodeContextSig&timeframe=24h'
    code, payload = _call(h, 'serve_signature_feature_analysis')
    assert code == 200
    assert isinstance(payload, dict)
    assert payload.get('n_dims') == 2
    assert isinstance(payload.get('direction'), list)
    assert len(payload['direction']) == 2


def test_gepa_analysis_endpoint_with_mocked_actions():
    import enhanced_dashboard_server as eds
    from dspy_agent.db.data_models import SignatureMetrics
    h = DummyHandler()
    now = time.time()
    # metrics with last optimization timestamp
    m = SignatureMetrics(signature_name='TaskAgentSig', performance_score=90.0, success_rate=95.0,
                         avg_response_time=1.5, memory_usage='100MB', iterations=10,
                         last_updated='now', signature_type='execution', active=True,
                         optimization_history=[{'timestamp': now - 10, 'type': 'gepa', 'performance_gain': 5.0, 'accuracy_improvement': 2.0, 'response_time_reduction': 0.1}])
    # actions around that timestamp
    pre = SimpleNamespace(timestamp=now-50, reward=0.2, environment=eds.Environment.DEVELOPMENT,
                          parameters={'signature_name': 'TaskAgentSig'}, result={})
    post = SimpleNamespace(timestamp=now-5, reward=0.9, environment=eds.Environment.DEVELOPMENT,
                           parameters={'signature_name': 'TaskAgentSig'}, result={})
    h.data_manager = DummyDM(actions=[pre, post], metrics=m)
    h.path = '/api/signature/gepa-analysis?name=TaskAgentSig&window=60'
    code, payload = _call(h, 'serve_signature_gepa_analysis')
    assert code == 200
    assert 'pre' in payload and 'post' in payload and 'delta' in payload
    assert payload['pre']['count'] >= 1 and payload['post']['count'] >= 1
    # Expect avg reward increased post optimization in this mocked setup
    assert payload['delta']['reward'] > 0


def test_signature_graph_with_filters_min_reward_and_verifier():
    import enhanced_dashboard_server as eds
    h = DummyHandler()
    now = time.time()
    a1 = SimpleNamespace(
        timestamp=now-100,
        reward=0.9,
        environment=eds.Environment.DEVELOPMENT,
        parameters={'signature_name': 'SigA', 'verifier_scores': {'VerifierX': 0.8}},
        result={}
    )
    a2 = SimpleNamespace(
        timestamp=now-80,
        reward=0.2,
        environment=eds.Environment.DEVELOPMENT,
        parameters={'signature_name': 'SigB', 'verifier_scores': {'VerifierY': 0.5}},
        result={}
    )
    h.data_manager = DummyDM(actions=[a1, a2])
    # Filter to min_reward=0.5 and verifier=VerifierX should include only SigA->VerifierX
    h.path = '/api/signature/graph?timeframe=24h&min_reward=0.5&verifier=VerifierX'
    code, payload = _call(h, 'serve_signature_graph')
    assert code == 200
    assert any(n['id'] == 'SigA' for n in payload['nodes'])
    assert any(n['id'] == 'VerifierX' for n in payload['nodes'])
    assert any(e['source'] == 'SigA' and e['target'] == 'VerifierX' for e in payload['edges'])
    # Changing filter to verifier=VerifierY returns 0 edges due to reward threshold
    h.path = '/api/signature/graph?timeframe=24h&min_reward=0.5&verifier=VerifierY'
    code2, payload2 = _call(h, 'serve_signature_graph')
    assert code2 == 200
    assert len(payload2['edges']) == 0


def test_signature_graph_export_headers():
    import enhanced_dashboard_server as eds
    h = DummyHandler()
    now = time.time()
    a = SimpleNamespace(
        timestamp=now-10,
        reward=0.9,
        environment=eds.Environment.DEVELOPMENT,
        parameters={'signature_name': 'SigA', 'verifier_scores': {'VerifierX': 0.8}},
        result={}
    )
    h.data_manager = DummyDM(actions=[a])
    h.path = '/api/signature/graph?timeframe=24h&download=1'
    code, _ = _call(h, 'serve_signature_graph')
    assert code == 200
    # Content-Disposition header should be present
    headers = dict(h._headers)
    cd = headers.get('Content-Disposition')
    assert cd and 'attachment' in cd and 'signature-graph.json' in cd


def test_feature_analysis_limit_variations():
    import enhanced_dashboard_server as eds
    h = DummyHandler()
    now = time.time()
    # 6 actions to satisfy min sample size (>=5)
    acts = []
    for i in range(6):
        acts.append(SimpleNamespace(timestamp=now-10*(i+1), reward=1.0 - 0.1*i, environment=eds.Environment.DEVELOPMENT,
                                    parameters={'signature_name': 'CodeContextSig', 'doc_id': f'd{i}'}, result={}))
        # Orthogonal-ish vectors
        vec = [0.0]*6; vec[i] = 1.0
        h._reddb[f'embvec:d{i}'] = {'unit': vec}
    h.data_manager = DummyDM(actions=acts)
    # limit=2 should fail with insufficient data (<5)
    h.path = '/api/signature/feature-analysis?name=CodeContextSig&timeframe=24h&limit=2'
    code, payload = _call(h, 'serve_signature_feature_analysis')
    assert 'insufficient' in json.dumps(payload).lower()
    # limit=6 should succeed
    h.path = '/api/signature/feature-analysis?name=CodeContextSig&timeframe=24h&limit=6'
    code2, payload2 = _call(h, 'serve_signature_feature_analysis')
    assert code2 == 200
    assert payload2.get('n_dims') == 6
