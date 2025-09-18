from dspy_agent.streaming.feature_store import FeatureStore


def test_feature_store_snapshot_basic():
    store = FeatureStore(window=4, feature_names=['a', 'b'])
    store.update({'features': [1.0, 2.0], 'feature_names': ['a', 'b']})
    store.update({'features': [3.0, 4.0], 'feature_names': ['a', 'b']})
    snap = store.snapshot()
    assert snap is not None
    assert snap.count == 2
    assert snap.feature_names == ['a', 'b']
    assert snap.means == [2.0, 3.0]
    store.reset()
    assert store.snapshot() is None


def test_feature_store_window_rolls():
    store = FeatureStore(window=2)
    for i in range(3):
        store.update({'features': [float(i)]})
    snap = store.snapshot()
    assert snap is not None
    # Only last two should remain
    assert snap.count == 2
    assert snap.means[0] == 1.5
