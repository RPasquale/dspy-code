# DSPy Agent Testing Guide

This guide covers how to test the DSPy Agent comprehensively, add new tests, and ensure production-quality reliability.

## 🧪 Test Categories

### 1. **Unit Tests** (`tests/`)
- **Purpose**: Test individual components in isolation
- **Coverage**: Core modules, utilities, data structures
- **Run**: `uv run pytest tests/`

### 2. **Integration Tests** (`scripts/test_*.py`)
- **Purpose**: Test component interactions and end-to-end workflows
- **Coverage**: Database, CLI, Docker stack, full agent workflows
- **Run**: `uv run python scripts/run_all_tests.py`

### 3. **RL Tests** (`tests/test_rl_tooling.py`)
- **Purpose**: Test reinforcement learning components and bandit training
- **Coverage**: RLToolEnv, reward aggregation, PufferLib integration
- **Run**: `uv run pytest tests/test_rl_tooling.py`

### 4. **Simple Tests** (`scripts/test_agent_simple.py`)
- **Purpose**: Quick validation of core functionality
- **Coverage**: Basic imports, configuration, essential features
- **Run**: `uv run python scripts/test_agent_simple.py`

## 🚀 Quick Test Commands

### Run All Tests
```bash
# Comprehensive test suite (recommended)
./scripts/run_all_tests.py

# Simple functionality test (fast)
./scripts/test_agent_simple.py

# Unit tests only
uv run pytest tests/

# RL tests specifically
uv run pytest tests/test_rl_tooling.py
```

### Test in Different Environments
```bash
# Development environment
./scripts/deploy-dev.sh

# Test environment validation
./scripts/deploy-test.sh

# Production deployment test
./scripts/deploy-prod.sh
```

## 📝 Adding New Tests

### 1. **Unit Tests** (for individual components)

Create a new test file in `tests/`:

```python
# tests/test_new_feature.py
import unittest
from dspy_agent.new_feature import NewFeature

class TestNewFeature(unittest.TestCase):
    def test_basic_functionality(self):
        feature = NewFeature()
        result = feature.process("test input")
        self.assertEqual(result, "expected output")
    
    def test_edge_cases(self):
        feature = NewFeature()
        with self.assertRaises(ValueError):
            feature.process(None)
```

### 2. **Integration Tests** (for workflows)

Add to existing integration scripts or create new ones:

```python
# scripts/test_new_integration.py
def test_new_workflow():
    """Test the new workflow end-to-end"""
    # Setup
    from dspy_agent.workflow import NewWorkflow
    
    # Execute
    workflow = NewWorkflow()
    result = workflow.run()
    
    # Verify
    assert result.success
    assert result.metrics["accuracy"] > 0.8
    
    return True

def main():
    success = test_new_workflow()
    print("✅ New workflow test passed" if success else "❌ New workflow test failed")
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())
```

### 3. **RL Tests** (for reinforcement learning)

Add to `tests/test_rl_tooling.py`:

```python
def test_new_rl_component():
    """Test new RL component behavior"""
    from dspy_agent.rl.new_component import NewRLComponent
    
    component = NewRLComponent()
    
    # Test reward calculation
    result = component.calculate_reward(metrics={"accuracy": 0.9})
    assert result > 0.8
    
    # Test action selection
    action = component.select_action(observations=[0.9, 0.1])
    assert action in component.valid_actions
```

## 🔧 Test Infrastructure

### Test Configuration

The test infrastructure uses:

- **pytest**: For unit tests with fixtures and parametrization
- **unittest**: For integration tests and compatibility
- **Custom test runners**: For comprehensive validation

### Test Data

- **Fixtures**: Use `tests/fixtures/` for test data
- **Temporary files**: Use `tempfile` for isolated test environments
- **Mock data**: Create realistic but controlled test scenarios

### Test Environment

```bash
# Ensure clean test environment
export DSPY_TEST_MODE=true
export DSPY_LOG_LEVEL=DEBUG

# Run tests with coverage
uv run pytest tests/ --cov=dspy_agent --cov-report=html
```

## 🎯 RL Testing Deep Dive

### Testing Bandit Training

The RL tests verify that the bandit trainer learns to prefer high-reward actions:

```python
def test_bandit_trainer_prefers_high_reward_action():
    """Verify bandit trainer learns to prefer 'patch' over 'run_tests'"""
    make_env = _build_env_factory()
    cfg = TrainerConfig(
        steps=200, 
        policy="epsilon-greedy", 
        policy_kwargs={"epsilon": 0.2, "seed": 7}, 
        n_envs=1
    )
    
    stats = bandit_trainer(make_env, cfg)
    
    # Verify learning occurred
    tools = [info.get("tool") for info in stats.infos if info.get("tool")]
    counts = Counter(tools)
    
    assert counts["patch"] > counts.get("run_tests", 0)
    assert counts["patch"] >= int(cfg.steps * 0.6)
```

### Testing PufferLib Integration

Tests automatically skip if PufferLib isn't installed:

```python
@pytest.mark.skipif(
    importlib.util.find_spec("pufferlib") is None, 
    reason="pufferlib not installed"
)
def test_bandit_trainer_puffer_vectorizes_and_learns():
    """Test vectorized training with PufferLib"""
    # Test multi-environment training
    # Verify reward aggregation across environments
    # Check action selection consistency
```

### Testing Reward Aggregation

Verify that rewards are calculated correctly:

```python
def test_rl_tool_env_reward_and_info():
    """Test reward calculation and observation updates"""
    env = make_env()
    obs, info = env.reset()
    
    # Take high-reward action
    patch_idx = env.action_names.index("patch")
    obs_after, reward, terminated, truncated, step_info = env.step(patch_idx)
    
    # Verify reward calculation
    expected_reward = 0.92 - 0.5 * 0.05  # pass_rate - penalty * blast_radius
    assert reward == pytest.approx(expected_reward, rel=1e-3)
    
    # Verify observation updates
    assert obs_after[0] == pytest.approx(0.92, rel=1e-3)  # pass_rate
    assert obs_after[1] == pytest.approx(0.05, rel=1e-3)  # blast_radius
```

## 🚨 Test Reliability

### Ensuring Production Quality

1. **No Mocking in Critical Paths**: RL tests use real implementations
2. **Deterministic Results**: Use fixed seeds for reproducible tests
3. **Comprehensive Coverage**: Test happy path, edge cases, and error conditions
4. **Performance Validation**: Ensure tests complete in reasonable time

### Test Dependencies

```python
# Handle optional dependencies gracefully
try:
    import pufferlib
    PUFFERLIB_AVAILABLE = True
except ImportError:
    PUFFERLIB_AVAILABLE = False

@pytest.mark.skipif(not PUFFERLIB_AVAILABLE, reason="pufferlib not installed")
def test_puffer_integration():
    # Test only runs if dependency is available
    pass
```

### Test Isolation

```python
# Use temporary directories for test isolation
import tempfile
from pathlib import Path

def test_with_clean_environment():
    with tempfile.TemporaryDirectory() as tmpdir:
        test_path = Path(tmpdir)
        # Run test in isolated environment
        result = run_test(test_path)
        assert result.success
```

## 📊 Test Metrics

### Coverage Targets

- **Unit Tests**: >90% line coverage
- **Integration Tests**: Cover all major workflows
- **RL Tests**: Verify learning behavior and reward calculations
- **End-to-End Tests**: Validate complete user journeys

### Performance Benchmarks

- **Unit Tests**: <1 second per test
- **Integration Tests**: <30 seconds total
- **RL Tests**: <60 seconds for training validation
- **Full Test Suite**: <5 minutes

## 🔍 Debugging Tests

### Common Issues

1. **Import Errors**: Ensure virtual environment is activated
2. **Permission Errors**: Check file permissions and Docker access
3. **Timeout Issues**: Increase timeout for slow operations
4. **Flaky Tests**: Use fixed seeds and deterministic data

### Debug Commands

```bash
# Run specific test with verbose output
uv run pytest tests/test_rl_tooling.py::test_bandit_trainer_prefers_high_reward_action -v

# Run tests with debugging
uv run pytest tests/ --pdb

# Check test coverage
uv run pytest tests/ --cov=dspy_agent --cov-report=term-missing
```

## 🎯 Best Practices

### Writing Reliable Tests

1. **Test Behavior, Not Implementation**: Focus on what the code does, not how
2. **Use Realistic Data**: Create test data that mirrors production scenarios
3. **Test Edge Cases**: Include boundary conditions and error cases
4. **Keep Tests Fast**: Optimize for speed while maintaining coverage
5. **Document Test Purpose**: Clear docstrings explaining what each test validates

### Test Organization

```
tests/
├── test_core/           # Core functionality tests
├── test_rl/            # RL-specific tests
├── test_integration/   # Integration tests
├── fixtures/           # Test data and fixtures
└── conftest.py         # Pytest configuration
```

### Continuous Integration

```yaml
# .github/workflows/tests.yml
name: tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - name: Install dependencies
        run: |
          pip install uv
          uv sync
      - name: Run tests
        run: |
          uv run python scripts/run_all_tests.py
          uv run pytest tests/ --cov=dspy_agent
```

## 🚀 Getting Started

### For New Contributors

1. **Run Existing Tests**: `./scripts/test_agent_simple.py`
2. **Understand Test Structure**: Read `tests/test_rl_tooling.py`
3. **Add Your Tests**: Follow the patterns in existing test files
4. **Validate Changes**: Run full test suite before submitting

### For Production Deployment

1. **Run Full Test Suite**: `./scripts/run_all_tests.py`
2. **Validate RL Components**: `uv run pytest tests/test_rl_tooling.py`
3. **Test Deployment**: `./scripts/deploy-test.sh`
4. **Monitor Test Results**: Check coverage and performance metrics

## 📚 Additional Resources

- **Test Examples**: See `tests/test_rl_tooling.py` for RL testing patterns
- **Integration Examples**: See `scripts/test_*.py` for workflow testing
- **Configuration**: Check `pyproject.toml` for test dependencies
- **CI/CD**: See `.github/workflows/tests.yml` for automated testing

---

**Remember**: Good tests are the foundation of reliable software. Take time to write comprehensive, maintainable tests that give you confidence in your code changes.
