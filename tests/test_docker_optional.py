import os
import pytest


@pytest.mark.docker
def test_docker_marker_only_runs_when_enabled():
    if not os.environ.get('DOCKER_TESTS'):
        pytest.skip('Docker tests disabled (set DOCKER_TESTS=1 to enable)')
    # Placeholder: In real CI with docker, we could exec into kafka and validate configs.
    assert True

