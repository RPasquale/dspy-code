import types

from dspy_agent.rl.rlkit import make_default_analytics_cb


def test_make_default_analytics_cb_static_signature():
    cb = make_default_analytics_cb('MySig', env='production', action_type='VERIFICATION')
    info = {'doc_id': 'abc123', 'execution_time': 0.42, 'verifier_scores': {'V': 0.9}}
    meta = cb(0, info, 1.0)
    assert meta is not None
    assert meta['signature_name'] == 'MySig'
    assert meta['doc_id'] == 'abc123'
    assert meta['environment'] == 'production'
    assert meta['action_type'] == 'VERIFICATION'
    assert isinstance(meta['execution_time'], float)


def test_make_default_analytics_cb_from_info_signature():
    cb = make_default_analytics_cb(None, env='development', action_type='VERIFICATION')
    info = {'signature_name': 'OtherSig', 'doc_id': 'xyz', 'verifier_scores': {'A': 0.1}}
    meta = cb(1, info, 0.5)
    assert meta is not None
    assert meta['signature_name'] == 'OtherSig'
    assert meta['doc_id'] == 'xyz'

