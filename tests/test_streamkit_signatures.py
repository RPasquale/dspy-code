import time

from dspy_agent.streaming.streamkit import (
    LocalBus,
    SignatureStream,
    SignatureVerifierConfig,
    SignatureStreamManager,
)


def test_signature_stream_manager_publishes_and_rewards():
    bus = LocalBus()
    cfg = SignatureStream(
        name='test_signature',
        sources=['source.topic'],
        sink='signatures.test_signature',
        verifiers=[
            SignatureVerifierConfig(name='applied', module='dspy_agent.verifiers.signatures:verify_patch_applied'),
        ],
    )
    sink_queue = bus.subscribe('signatures.test_signature')
    reward_queue = bus.subscribe('agent.learning')
    manager = SignatureStreamManager(bus, cfg)
    manager.start()
    try:
        bus.publish('source.topic', {'applied': True})
        timeout = time.time() + 2.0
        event = None
        while time.time() < timeout:
            try:
                event = sink_queue.get(timeout=0.1)
                break
            except Exception:
                continue
        assert event is not None
        assert event['signature'] == 'test_signature'
        assert event['verifier_scores']['applied'] == 1.0
        assert event['reward'] >= 0.9
        # Ensure learning event emitted
        learn = None
        timeout = time.time() + 2.0
        while time.time() < timeout:
            try:
                candidate = reward_queue.get(timeout=0.1)
                if candidate.get('signature') == 'test_signature':
                    learn = candidate
                    break
            except Exception:
                continue
        assert learn is not None
        assert learn['verifier_scores']['applied'] == 1.0
    finally:
        manager.stop()
        manager.join(timeout=1.0)
