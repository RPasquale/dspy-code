from datetime import datetime

from dspy_agent.db.enhanced_storage import EnhancedDataManager
from dspy_agent.dbkit import RedDBStorage
from dspy_agent.db.data_models import (
    ActionRecord,
    ActionType,
    Environment,
    RedDBDataManager,
    SignatureMetrics,
    VerifierMetrics,
)


def test_record_action_signature_middleware(monkeypatch):
    captured = {}

    def stub_record_action(self, action):
        captured['action'] = action

    # Monkeypatch base class method to capture post-middleware action
    monkeypatch.setattr('dspy_agent.db.data_models.RedDBDataManager.record_action', stub_record_action, raising=False)

    dm = EnhancedDataManager()
    a = ActionRecord(
        action_id='A1',
        timestamp=0.0,
        action_type=ActionType.VERIFICATION,
        state_before={},
        state_after={},
        parameters={'signature_name': 'MySig'},
        result={},
        reward=1.0,
        confidence=0.9,
        execution_time=0.1,
        environment=Environment.DEVELOPMENT,
    )
    dm.record_action(a)

    post = captured.get('action')
    assert post is not None
    # Middleware should inject signature_name everywhere
    assert post.parameters.get('signature_name') == 'MySig'
    assert post.result.get('signature_name') == 'MySig'
    assert post.state_before.get('signature_name') == 'MySig'
    assert post.state_after.get('signature_name') == 'MySig'


def test_store_signature_metrics_updates_registry():
    dm = RedDBDataManager(namespace='test-signature-registry')
    dm.storage = RedDBStorage(url=None, namespace=dm.namespace)
    sig = SignatureMetrics(
        signature_name='RegistrySig',
        performance_score=0.91,
        success_rate=0.95,
        avg_response_time=1.2,
        memory_usage='100MB',
        iterations=12,
        last_updated=datetime.now().isoformat(),
        signature_type='analysis',
        active=True,
    )

    dm.store_signature_metrics(sig)

    names = dm.storage.get(dm._key('registries', 'signatures')) or []
    assert 'RegistrySig' in names
    all_metrics = dm.get_all_signature_metrics()
    assert any(m.signature_name == 'RegistrySig' for m in all_metrics)


def test_store_verifier_metrics_updates_registry():
    dm = RedDBDataManager(namespace='test-verifier-registry')
    dm.storage = RedDBStorage(url=None, namespace=dm.namespace)
    verifier = VerifierMetrics(
        verifier_name='RegistryVerifier',
        accuracy=0.97,
        status='active',
        checks_performed=42,
        issues_found=2,
        last_run=datetime.now().isoformat(),
        avg_execution_time=0.5,
    )

    dm.store_verifier_metrics(verifier)

    names = dm.storage.get(dm._key('registries', 'verifiers')) or []
    assert 'RegistryVerifier' in names
    all_metrics = dm.get_all_verifier_metrics()
    assert any(v.verifier_name == 'RegistryVerifier' for v in all_metrics)
