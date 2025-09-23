from types import SimpleNamespace

from dspy_agent.verifiers.custom import get_verifiers


def test_custom_verifiers_return_scores():
    verifiers = get_verifiers()
    assert verifiers, 'expected at least one custom verifier'
    result = SimpleNamespace(metrics={
        'pass_rate': 0.8,
        'baseline_pass_rate': 0.5,
        'blast_radius': 10.0,
        'quality_policy': 0.9,
        'lint_issues': 2,
    })
    scores = {getattr(v, 'kind', f'v{i}'): float(v(result)) for i, v in enumerate(verifiers)}
    # pass_rate_improve should be positive
    assert scores.get('pass_rate_improve', 0) > 0
    # low_blast should be negative (inverted)
    assert scores.get('low_blast', 0) < 0
    # quality_policy positive in [0,1]
    assert 0.0 <= scores.get('quality_policy', 0) <= 1.0
    # lint_penalty negative when issues > 0
    assert scores.get('lint_penalty', 0) < 0

