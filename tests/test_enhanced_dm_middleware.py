from dspy_agent.db.enhanced_storage import EnhancedDataManager
from dspy_agent.db.data_models import ActionRecord, ActionType, Environment


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

