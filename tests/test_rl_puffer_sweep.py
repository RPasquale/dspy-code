import importlib
import importlib.util
import sys
import types

import pytest

from dspy_agent.rl import puffer_sweep


def _install_fake_carbs(monkeypatch):
    fake = types.ModuleType("carbs")

    class _Space:
        def __init__(self, min, max, is_integer=False):
            self.min = min
            self.max = max
            self.is_integer = is_integer

    class Param:
        def __init__(self, name, space, search_center=None):
            self.name = name
            self.space = space
            self.search_center = search_center

    class CARBSParams:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class _Suggestion:
        def __init__(self, names, counter):
            # Give different values for different parameters
            values = {}
            for i, name in enumerate(names):
                if "pass_rate" in name:
                    values[name] = 0.55  # Fixed value to match test expectation
                elif "blast_radius" in name:
                    values[name] = 0.01 + 0.01 * counter  # Should be in range [0.01, 0.2]
                elif "steps" in name:
                    values[name] = 50 + 10 * counter  # Should be >= 50
                else:
                    values[name] = 0.5 + 0.05 * counter
            self.suggestion = values
            self.metadata = {"call": counter}

    class CARBS:
        def __init__(self, params, flat_params):
            self.params = params
            self.flat_params = flat_params
            self._calls = 0
            self.observations = []

        def suggest(self):
            self._calls += 1
            names = [param.name for param in self.flat_params]
            return _Suggestion(names, self._calls)

        def observe(self, observation):
            self.observations.append(observation)

    class ObservationInParam:
        def __init__(self, input, output, cost, is_failure):
            self.input = input
            self.output = output
            self.cost = cost
            self.is_failure = is_failure

    fake.Param = Param
    fake.LinearSpace = _Space
    fake.LogSpace = _Space
    fake.LogitSpace = _Space
    fake.CARBSParams = CARBSParams
    fake.CARBS = CARBS
    fake.ObservationInParam = ObservationInParam

    monkeypatch.setitem(sys.modules, "carbs", fake)
    return fake


def _reload_puffer_sweep():
    return importlib.reload(puffer_sweep)


def test_carbs_strategy_round_trip(monkeypatch):
    _install_fake_carbs(monkeypatch)
    module = _reload_puffer_sweep()
    Carbs = module.Carbs

    sweep_config = {
        "method": "carbs",
        "metric": "reward",
        "goal": "maximize",
        "weights": {
            "pass_rate": {"distribution": "uniform", "min": 0.6, "max": 1.0, "mean": 0.8},
            "blast_radius": {"distribution": "logit_normal", "min": 0.01, "max": 0.2, "mean": 0.05},
        },
        "trainer": {
            "steps": {"distribution": "int_uniform", "min": 50, "max": 150, "mean": 100},
        },
    }

    strategy = Carbs(sweep_config)
    suggestion, info = strategy.suggest()

    print(f"Suggestion: {suggestion}")
    print(f"Info: {info}")
    print(f"Raw params: {info.get('raw_params', {})}")

    assert suggestion["weights"]["pass_rate"] != suggestion["weights"]["blast_radius"]
    assert suggestion["trainer"]["steps"] >= 50
    assert info["method"] == "carbs"
    assert "raw_params" in info
    assert "weights.pass_rate" in info["raw_params"]
    assert info["raw_params"]["weights.pass_rate"] == pytest.approx(0.55)

    strategy.observe(suggestion, score=0.9, cost=120.0, is_failure=False)
    assert len(strategy.carbs.observations) == 1
    observation = strategy.carbs.observations[0]
    assert observation.output == pytest.approx(0.9)
    assert observation.cost == pytest.approx(120.0)


def test_get_strategy_resolution():
    module = _reload_puffer_sweep()
    assert module.get_strategy("random") is module.RandomStrategy
    assert module.get_strategy("pareto") is module.ParetoGenetic
    assert module.get_strategy("carbs").__name__ == "Carbs"
    assert module.get_strategy("protein").__name__ == "Protein"


@pytest.mark.skipif(importlib.util.find_spec("pyro") is not None, reason="pyro installed; runtime error guard not triggered")
def test_protein_requires_pyro(monkeypatch):
    module = _reload_puffer_sweep()
    with pytest.raises(RuntimeError, match="Pyro / torch not available"):
        module.Protein({"metric": "reward"})


def test_carbs_requires_dependency(monkeypatch):
    monkeypatch.delitem(sys.modules, "carbs", raising=False)
    module = _reload_puffer_sweep()
    with pytest.raises(RuntimeError, match="CARBS not available"):
        module.Carbs({"metric": "reward"})
