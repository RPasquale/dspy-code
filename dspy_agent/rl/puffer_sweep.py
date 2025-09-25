"""Wrappers around PufferLib's hyperparameter sweep utilities.

The original implementation lives at:
https://github.com/PufferAI/PufferLib/blob/3.0/pufferlib/sweep.py

We vendor a lightly adapted copy so the agent can schedule sweeps without
introducing a mandatory dependency on the full `pufferlib` package at runtime.

Only the parts that integrate with DSPy are kept. Optional dependencies such as
Pyro are imported lazily so that light-weight environments can still run the
code when advanced strategies are unused.
"""

from __future__ import annotations

import math
import random
from copy import deepcopy
from typing import Dict, Iterable, Iterator, List, Mapping, MutableMapping, Optional, Sequence, Tuple

# Avoid importing numpy at module import time to prevent native crashes in
# restricted environments. Strategies that require numpy import it lazily.
_HAS_NUMPY = False
def _ensure_numpy():
    global _HAS_NUMPY
    if _HAS_NUMPY:
        return
    import importlib
    try:
        importlib.import_module("numpy")
        _HAS_NUMPY = True
    except Exception as exc:  # pragma: no cover - optional
        raise RuntimeError("numpy is required for this operation") from exc

# PufferLib is optional - we'll import it only when needed
pufferlib = None  # type: ignore[assignment]


def _ensure_pufferlib():
    """Ensure PufferLib is available, import it if needed."""
    global pufferlib
    if pufferlib is None:
        try:
            import pufferlib  # type: ignore
        except ImportError:
            raise ImportError("PufferLib not available. Install with: pip install pufferlib>=3.0.0")


def _unroll_nested_dict(mapping: Mapping[str, object], prefix: Tuple[str, ...] = ()) -> Iterator[Tuple[str, object]]:
    try:
        _ensure_pufferlib()
        for key, value in pufferlib.unroll_nested_dict(mapping):  # type: ignore[attr-defined]
            # Normalize to dot-separated keys
            yield str(key).replace('/', '.'), value
        return
    except ImportError:
        pass  # Fall back to manual implementation
    for key, value in mapping.items():
        key_str = str(key)
        path = prefix + (key_str,)
        if isinstance(value, Mapping):
            yield from _unroll_nested_dict(value, path)
        else:
            yield ".".join(path), value

# Pyro/torch are optional; import lazily only when required by Protein strategy
_HAS_PYRO = False
torch = None  # type: ignore
pyro = None  # type: ignore
_gp = None  # type: ignore

def _ensure_pyro_optional_imports() -> None:
    global _HAS_PYRO, torch, pyro, _gp
    if _HAS_PYRO:
        return
    try:  # pragma: no cover - environment dependent
        import torch as _torch  # type: ignore
        import pyro as _pyro  # type: ignore
        from pyro.contrib import gp as _gp_mod  # type: ignore
        torch = _torch
        pyro = _pyro
        _gp = _gp_mod
        _HAS_PYRO = True
    except Exception:
        _HAS_PYRO = False


class Space:
    def __init__(self, min: float, max: float, scale: float, mean: float, *, is_integer: bool = False) -> None:
        self.min = min
        self.max = max
        self.scale = scale
        self.mean = mean
        self.is_integer = is_integer
        self.norm_min = self.normalize(min)
        self.norm_max = self.normalize(max)
        self.norm_mean = self.normalize(mean)

    def normalize(self, value: float) -> float:
        raise NotImplementedError

    def unnormalize(self, value: float) -> float:
        raise NotImplementedError


class Linear(Space):
    def __init__(self, min: float, max: float, scale: float, mean: float, *, is_integer: bool = False) -> None:
        if scale == "auto":
            scale = 0.5
        super().__init__(min, max, scale, mean, is_integer=is_integer)

    def normalize(self, value: float) -> float:
        zero_one = (value - self.min) / (self.max - self.min)
        return 2 * zero_one - 1

    def unnormalize(self, value: float) -> float:
        zero_one = (value + 1) / 2
        value = zero_one * (self.max - self.min) + self.min
        if self.is_integer:
            value = round(value)
        return value


class Pow2(Space):
    def __init__(self, min: float, max: float, scale: float, mean: float, *, is_integer: bool = False) -> None:
        if scale == "auto":
            scale = 0.5
        super().__init__(min, max, scale, mean, is_integer=is_integer)

    def normalize(self, value: float) -> float:
        zero_one = (math.log(value, 2) - math.log(self.min, 2)) / (math.log(self.max, 2) - math.log(self.min, 2))
        return 2 * zero_one - 1

    def unnormalize(self, value: float) -> float:
        zero_one = (value + 1) / 2
        log_spaced = zero_one * (math.log(self.max, 2) - math.log(self.min, 2)) + math.log(self.min, 2)
        rounded = round(log_spaced)
        return 2 ** rounded


class Log(Space):
    base: int = 10

    def __init__(self, min: float, max: float, scale: float, mean: float, *, is_integer: bool = False) -> None:
        if scale == "time":
            scale = 1 / (math.log(max, 2) - math.log(min, 2))
        elif scale == "auto":
            scale = 0.5
        super().__init__(min, max, scale, mean, is_integer=is_integer)

    def normalize(self, value: float) -> float:
        zero_one = (math.log(value, self.base) - math.log(self.min, self.base)) / (math.log(self.max, self.base) - math.log(self.min, self.base))
        return 2 * zero_one - 1

    def unnormalize(self, value: float) -> float:
        zero_one = (value + 1) / 2
        log_spaced = zero_one * (math.log(self.max, self.base) - math.log(self.min, self.base)) + math.log(self.min, self.base)
        value = self.base ** log_spaced
        if self.is_integer:
            value = round(value)
        return value


class Logit(Space):
    base: int = 10

    def __init__(self, min: float, max: float, scale: float, mean: float, *, is_integer: bool = False) -> None:
        if scale == "auto":
            scale = 0.5
        super().__init__(min, max, scale, mean, is_integer=is_integer)

    def normalize(self, value: float) -> float:
        zero_one = (math.log(1 - value, self.base) - math.log(1 - self.min, self.base)) / (math.log(1 - self.max, self.base) - math.log(1 - self.min, self.base))
        return 2 * zero_one - 1

    def unnormalize(self, value: float) -> float:
        zero_one = (value + 1) / 2
        log_spaced = zero_one * (math.log(1 - self.max, self.base) - math.log(1 - self.min, self.base)) + math.log(1 - self.min, self.base)
        return 1 - self.base ** log_spaced


def _params_from_puffer_sweep(sweep_config: Mapping[str, Mapping[str, object]]) -> Dict[str, object]:
    param_spaces: Dict[str, object] = {}
    for name, param in sweep_config.items():
        if name in {"method", "metric", "goal", "downsample", "name", "max_score"}:
            continue
        if not isinstance(param, Mapping):
            continue
        if any(isinstance(param[k], Mapping) for k in param):
            param_spaces[name] = _params_from_puffer_sweep(param)  # nested
            continue
        distribution = param.get("distribution")
        if distribution is None:
            raise ValueError(f"Hyperparameter '{name}' missing distribution")
        kwargs = dict(
            min=float(param.get("min")),
            max=float(param.get("max")),
            scale=param.get("scale", 0.5),
            mean=float(param.get("mean")),
        )
        is_int = bool(param.get("is_integer", False))
        if distribution == "uniform":
            space = Linear(**kwargs, is_integer=is_int)
        elif distribution == "int_uniform":
            space = Linear(**kwargs, is_integer=True)
        elif distribution == "uniform_pow2":
            space = Pow2(**kwargs, is_integer=True)
        elif distribution == "log_normal":
            space = Log(**kwargs, is_integer=is_int)
        elif distribution == "logit_normal":
            space = Logit(**kwargs, is_integer=is_int)
        else:
            raise ValueError(f"Invalid distribution for {name}: {distribution}")
        param_spaces[name] = space
    return param_spaces


class Hyperparameters:
    def __init__(self, config: Mapping[str, object], *, verbose: bool = False) -> None:
        self._config = config
        spaces = _params_from_puffer_sweep(config)
        self.spaces = spaces
        self.flat_spaces = dict(_unroll_nested_dict(spaces))
        self.num = len(self.flat_spaces)
        metric = config.get("metric")
        if not isinstance(metric, str):
            raise ValueError("sweep config must provide 'metric'")
        self.metric = metric
        goal = config.get("goal", "maximize")
        if goal not in {"maximize", "minimize"}:
            raise ValueError("goal must be 'maximize' or 'minimize'")
        self.optimize_direction = 1 if goal == "maximize" else -1
        # Store lists to avoid requiring numpy for basic usage (Carbs strategy)
        self.search_centers: List[float] = [space.norm_mean for space in self.flat_spaces.values()]
        self.min_bounds: List[float] = [space.norm_min for space in self.flat_spaces.values()]
        self.max_bounds: List[float] = [space.norm_max for space in self.flat_spaces.values()]
        self.search_scales: List[float] = [float(space.scale) for space in self.flat_spaces.values()]
        if verbose:
            self._print_extrema()

    def _print_extrema(self) -> None:
        print("Min random sample:")
        for name, space in self.flat_spaces.items():
            print(f"\t{name}: {space.unnormalize(max(space.norm_mean - space.scale, space.norm_min))}")
        print("Max random sample:")
        for name, space in self.flat_spaces.items():
            print(f"\t{name}: {space.unnormalize(min(space.norm_mean + space.scale, space.norm_max))}")

    def sample(self, n: int, mu: Optional[Sequence[Sequence[float]]] = None, scale: float = 1.0):  # type: ignore[override]
        # Lightweight fallback; used by strategies that explicitly require numpy.
        _ensure_numpy()
        import numpy as np  # type: ignore
        mu_arr = np.array(mu) if mu is not None else np.array(self.search_centers)[None, :]
        if len(mu_arr.shape) == 1:
            mu_arr = mu_arr[None, :]
        n_input, n_dim = mu_arr.shape
        scale_arr = scale * np.array(self.search_scales)
        mu_idxs = np.random.randint(0, n_input, n)
        samples = scale_arr * (2 * np.random.rand(n, n_dim) - 1) + mu_arr[mu_idxs]
        return np.clip(samples, np.array(self.min_bounds), np.array(self.max_bounds))

    def from_dict(self, params: Mapping[str, object]):  # type: ignore[override]
        flat_params = dict(_unroll_nested_dict(params))
        values: List[float] = []
        for key, space in self.flat_spaces.items():
            if key not in flat_params:
                raise KeyError(f"Missing hyperparameter {key}")
            normed = space.normalize(float(flat_params[key]))
            values.append(normed)
        return values

    def to_dict(self, sample: Sequence[float], fill: Optional[MutableMapping[str, object]] = None) -> MutableMapping[str, object]:
        params = deepcopy(fill) if fill is not None else deepcopy(self.spaces)
        self._fill(params, self.spaces, sample)
        return params

    def _fill(self, params: MutableMapping[str, object], spaces: Mapping[str, object], flat_sample: Sequence[float], idx: int = 0) -> int:
        for name, space in spaces.items():
            if isinstance(space, Mapping):
                idx = self._fill(params[name], space, flat_sample, idx)
            else:
                params[name] = space.unnormalize(flat_sample[idx])
                idx += 1
        return idx


def pareto_points(observations: Sequence[Mapping[str, float]], eps: float = 1e-6) -> Tuple[List[Mapping[str, float]], List[int]]:
    # Pure-Python Pareto front: maximize score, minimize cost
    pareto: List[Mapping[str, float]] = []
    idxs: List[int] = []
    for i, obs_i in enumerate(observations):
        s_i = float(obs_i.get("output", 0.0))
        c_i = float(obs_i.get("cost", float("inf")))
        dominated = False
        for j, obs_j in enumerate(observations):
            if i == j:
                continue
            s_j = float(obs_j.get("output", 0.0))
            c_j = float(obs_j.get("cost", float("inf")))
            better_score = (s_j + eps) > s_i
            lower_cost = (c_j - eps) < c_i
            if better_score and lower_cost:
                dominated = True
                break
        if not dominated:
            pareto.append(obs_i)
            idxs.append(i)
    return pareto, idxs


class RandomStrategy:
    def __init__(self, sweep_config: Mapping[str, object], *, global_search_scale: float = 1.0, random_suggestions: int = 1024) -> None:
        self.hyperparameters = Hyperparameters(sweep_config)
        self.global_search_scale = global_search_scale
        self.random_suggestions = random_suggestions
        self.success_observations: List[Dict[str, object]] = []
        self.suggestion = None  # numpy ndarray when available

    def suggest(self, fill: Optional[MutableMapping[str, object]] = None) -> Tuple[MutableMapping[str, object], Dict[str, object]]:
        suggestions = self.hyperparameters.sample(self.random_suggestions, scale=self.global_search_scale)
        import numpy as np  # noqa: F401
        self.suggestion = random.choice(suggestions)
        return self.hyperparameters.to_dict(self.suggestion, fill), {}

    def observe(self, hypers: Mapping[str, object], score: float, cost: float, *, is_failure: bool = False) -> None:
        params = self.hyperparameters.from_dict(hypers)
        self.success_observations.append({
            "input": params,
            "output": float(score),
            "cost": float(cost),
            "is_failure": is_failure,
        })

    # Lightweight persistence API
    def state(self) -> Dict[str, object]:  # pragma: no cover - simple JSON state
        return {
            "suggestion": (self.suggestion.tolist() if getattr(self.suggestion, "tolist", None) else None),
            "observations": [
                {"input": list(map(float, obs.get("input", []))), "output": float(obs.get("output", 0.0)), "cost": float(obs.get("cost", 0.0)), "is_failure": bool(obs.get("is_failure", False))}
                for obs in self.success_observations
            ],
        }

    def load_state(self, data: Mapping[str, object]) -> None:  # pragma: no cover - simple JSON state
        try:
            _ensure_numpy()
            import numpy as np  # type: ignore
            self.suggestion = np.array(data.get("suggestion") or []) if data.get("suggestion") is not None else None
            self.success_observations = list(data.get("observations") or [])
        except Exception:
            pass


class ParetoGenetic:
    def __init__(
        self,
        sweep_config: Mapping[str, object],
        *,
        global_search_scale: float = 1.0,
        suggestions_per_pareto: int = 1,
        bias_cost: bool = True,
        log_bias: bool = False,
    ) -> None:
        self.hyperparameters = Hyperparameters(sweep_config)
        self.global_search_scale = global_search_scale
        self.suggestions_per_pareto = suggestions_per_pareto
        self.bias_cost = bias_cost
        self.log_bias = log_bias
        self.success_observations: List[Dict[str, object]] = []
        self.suggestion: Optional[np.ndarray] = None

    def suggest(self, fill: Optional[MutableMapping[str, object]] = None) -> Tuple[MutableMapping[str, object], Dict[str, object]]:
        if not self.success_observations:
            suggestion = self.hyperparameters.search_centers
            return self.hyperparameters.to_dict(suggestion, fill), {}
        candidates, _ = pareto_points(self.success_observations)
        _ensure_numpy()
        import numpy as np  # type: ignore
        pareto_costs = np.array([obs["cost"] for obs in candidates])
        if self.bias_cost:
            if self.log_bias:
                cost_dists = np.abs(np.log(pareto_costs[:, None]) - np.log(pareto_costs[None, :]))
            else:
                cost_dists = np.abs(pareto_costs[:, None] - pareto_costs[None, :])
            cost_dists += (np.max(pareto_costs) + 1) * np.eye(len(pareto_costs))
            idx = int(np.argmax(np.min(cost_dists, axis=1)))
            search_centers = candidates[idx]["input"]
        else:
            search_centers = np.stack([obs["input"] for obs in candidates])
        suggestions = self.hyperparameters.sample(len(candidates) * self.suggestions_per_pareto, mu=search_centers)
        suggestion = suggestions[np.random.randint(0, len(suggestions))]
        self.suggestion = suggestion
        return self.hyperparameters.to_dict(suggestion, fill), {}

    def observe(self, hypers: Mapping[str, object], score: float, cost: float, *, is_failure: bool = False) -> None:
        params = self.hyperparameters.from_dict(hypers)
        self.success_observations.append({
            "input": params,
            "output": float(score),
            "cost": float(cost),
            "is_failure": is_failure,
        })

    # Lightweight persistence API
    def state(self) -> Dict[str, object]:  # pragma: no cover - simple JSON state
        return {
            "observations": [
                {"input": list(map(float, obs.get("input", []))), "output": float(obs.get("output", 0.0)), "cost": float(obs.get("cost", 0.0)), "is_failure": bool(obs.get("is_failure", False))}
                for obs in self.success_observations
            ],
            "suggestion": (self.suggestion.tolist() if isinstance(self.suggestion, np.ndarray) else None),
        }

    def load_state(self, data: Mapping[str, object]) -> None:  # pragma: no cover - simple JSON state
        try:
            self.success_observations = list(data.get("observations") or [])
            self.suggestion = np.array(data.get("suggestion") or []) if data.get("suggestion") is not None else None
        except Exception:
            pass


def _require_pyro() -> None:
    _ensure_pyro_optional_imports()
    if not _HAS_PYRO:
        raise RuntimeError("Pyro / torch not available. Install 'pyro-ppl' to use Protein sweeps.")


def create_gp(x_dim: int, scale_length: float = 1.0):  # pragma: no cover - thin wrapper over Pyro objects
    _require_pyro()
    assert torch is not None and pyro is not None and _gp is not None
    X = scale_length * torch.ones((1, x_dim))
    y = torch.zeros((1,))
    matern_kernel = _gp.kernels.Matern32(input_dim=x_dim, lengthscale=X)
    linear_kernel = _gp.kernels.Polynomial(x_dim, degree=1)
    kernel = _gp.kernels.Sum(linear_kernel, matern_kernel)
    model = _gp.models.GPRegression(X, y, kernel=kernel, jitter=1.0e-4)
    model.noise = pyro.nn.PyroSample(pyro.distributions.LogNormal(math.log(1e-2), 0.5))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    return model, optimizer


class Protein:
    def __init__(
        self,
        sweep_config: Mapping[str, object],
        *,
        max_suggestion_cost: float = 3600,
        resample_frequency: int = 0,
        num_random_samples: int = 50,
        global_search_scale: float = 1.0,
        random_suggestions: int = 1024,
        suggestions_per_pareto: int = 256,
        seed_with_search_center: bool = True,
        expansion_rate: float = 0.25,
    ) -> None:
        _require_pyro()
        self.hyperparameters = Hyperparameters(sweep_config)
        self.num_random_samples = num_random_samples
        self.global_search_scale = global_search_scale
        self.random_suggestions = random_suggestions
        self.suggestions_per_pareto = suggestions_per_pareto
        self.seed_with_search_center = seed_with_search_center
        self.resample_frequency = resample_frequency
        self.max_suggestion_cost = max_suggestion_cost
        self.expansion_rate = expansion_rate
        self.success_observations: List[Dict[str, object]] = []
        self.failure_observations: List[Dict[str, object]] = []
        self.suggestion_idx = 0
        model, opt = create_gp(self.hyperparameters.num)
        self.gp_score = model
        self.score_opt = opt
        model_c, opt_c = create_gp(self.hyperparameters.num)
        self.gp_cost = model_c
        self.cost_opt = opt_c

    def suggest(self, fill: Optional[MutableMapping[str, object]] = None) -> Tuple[MutableMapping[str, object], Dict[str, object]]:
        _require_pyro()
        info: Dict[str, object] = {}
        self.suggestion_idx += 1
        if not self.success_observations and self.seed_with_search_center:
            best = self.hyperparameters.search_centers
            return self.hyperparameters.to_dict(best, fill), info
        if not self.seed_with_search_center and len(self.success_observations) < self.num_random_samples:
            suggestions = self.hyperparameters.sample(self.random_suggestions)
            suggestion = random.choice(suggestions)
            return self.hyperparameters.to_dict(suggestion, fill), info
        if self.resample_frequency and self.suggestion_idx % self.resample_frequency == 0:
            candidates, _ = pareto_points(self.success_observations)
            suggestions = np.stack([obs["input"] for obs in candidates])
            best_idx = np.random.randint(0, len(candidates))
            best = suggestions[best_idx]
            return self.hyperparameters.to_dict(best, fill), info
        import numpy as _np
        params = _np.array([obs["input"] for obs in self.success_observations])
        params_t = torch.from_numpy(params)
        y = _np.array([obs["output"] for obs in self.success_observations])
        min_score = float(_np.min(y))
        max_score = float(_np.max(y))
        y_norm = (y - min_score) / (abs(max_score - min_score) + 1e-6)
        self.gp_score.set_data(params_t, torch.from_numpy(y_norm))
        self.gp_score.train()
        _gp.util.train(self.gp_score, self.score_opt)
        self.gp_score.eval()
        c = _np.array([obs["cost"] for obs in self.success_observations])
        log_c = _np.log(c)
        log_c_min = float(_np.min(log_c))
        log_c_max = float(_np.max(log_c))
        log_c_norm = (log_c - log_c_min) / (log_c_max - log_c_min + 1e-6)
        self.gp_cost.mean_function = lambda x: 1  # type: ignore[assignment]
        self.gp_cost.set_data(params_t, torch.from_numpy(log_c_norm))
        self.gp_cost.train()
        _gp.util.train(self.gp_cost, self.cost_opt)
        self.gp_cost.eval()
        candidates, pareto_idxs = pareto_points(self.success_observations)
        suggestions = self.hyperparameters.sample(len(candidates) * self.suggestions_per_pareto, mu=np.stack([obs["input"] for obs in candidates]))
        suggestions_t = torch.from_numpy(suggestions)
        with torch.no_grad():
            gp_y_norm, _ = self.gp_score(suggestions_t)
            gp_log_c_norm, _ = self.gp_cost(suggestions_t)
        gp_y_norm = gp_y_norm.numpy()
        gp_log_c_norm = gp_log_c_norm.numpy()
        gp_y = gp_y_norm * (max_score - min_score) + min_score
        gp_log_c = gp_log_c_norm * (log_c_max - log_c_min) + log_c_min
        gp_c = np.exp(gp_log_c)
        max_c_mask = gp_c < self.max_suggestion_cost
        target = (1 + self.expansion_rate) * np.random.rand()
        weight = 1 - abs(target - gp_log_c_norm)
        suggestion_scores = self.hyperparameters.optimize_direction * max_c_mask * (gp_y_norm * weight)
        best_idx = int(np.argmax(suggestion_scores))
        info.update({
            "cost": float(gp_c[best_idx]),
            "score": float(gp_y[best_idx]),
            "rating": float(suggestion_scores[best_idx]),
        })
        best = suggestions[best_idx]
        return self.hyperparameters.to_dict(best, fill), info

    def observe(self, hypers: Mapping[str, object], score: float, cost: float, *, is_failure: bool = False) -> None:
        params = self.hyperparameters.from_dict(hypers)
        new_observation = {
            "input": params,
            "output": float(score),
            "cost": float(cost),
            "is_failure": is_failure,
        }
        if not self.success_observations:
            self.success_observations.append(new_observation)
            return
        success_params = np.stack([obs["input"] for obs in self.success_observations])
        dist = np.linalg.norm(params - success_params, axis=1)
        same = np.where(dist < 1e-6)[0]
        if len(same) > 0:
            self.success_observations[same[0]] = new_observation
        else:
            self.success_observations.append(new_observation)


def _carbs_params_from_puffer_sweep(
    sweep_config: Mapping[str, Mapping[str, object]],
    prefix: Tuple[str, ...] = (),
):
    from carbs import Param, LinearSpace, LogSpace, LogitSpace  # type: ignore
    param_spaces = {}
    for name, param in sweep_config.items():
        if name in {"method", "name", "metric", "max_score"}:
            continue
        if not isinstance(param, Mapping):
            continue
        path = prefix + (str(name),)
        if any(isinstance(param[k], Mapping) for k in param):
            param_spaces[name] = _carbs_params_from_puffer_sweep(param, path)  # nested
            continue
        distribution = param.get("distribution")
        kwargs = dict(min=float(param.get("min")), max=float(param.get("max")))
        if distribution == "uniform":
            space = LinearSpace(**kwargs)
        elif distribution in {"int_uniform", "uniform_pow2"}:
            space = LinearSpace(**kwargs, is_integer=True)
        elif distribution == "log_normal":
            space = LogSpace(**kwargs)
        elif distribution == "logit_normal":
            space = LogitSpace(**kwargs)
        else:
            raise ValueError(f"Invalid distribution: {distribution}")
        full_name = ".".join(path)
        param_spaces[name] = Param(name=full_name, space=space, search_center=param.get("mean"))
    return param_spaces


class Carbs:
    def __init__(
        self,
        sweep_config: Mapping[str, object],
        *,
        max_suggestion_cost: float = 3600,
        resample_frequency: int = 5,
        num_random_samples: int = 10,
    ) -> None:
        try:
            from carbs import CARBS, CARBSParams  # type: ignore
        except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("CARBS not available. Install 'carbs' or 'pufferlib[carbs]' to use this sweep strategy.") from exc

        self.hyperparameters = Hyperparameters(sweep_config)
        self._param_spaces = _carbs_params_from_puffer_sweep(sweep_config)
        flat_items = list(_unroll_nested_dict(self._param_spaces))
        flat_params = [item[1] for item in flat_items]
        self._flat_keys = [item[0] for item in flat_items]
        self._flat_names = [param.name for param in flat_params]

        carbs_params = CARBSParams(
            better_direction_sign=1,
            is_wandb_logging_enabled=False,
            resample_frequency=resample_frequency,
            num_random_samples=num_random_samples,
            max_suggestion_cost=max_suggestion_cost,
            is_saved_on_every_observation=False,
        )

        self.carbs = CARBS(carbs_params, flat_params)
        self._last_candidate: Optional[object] = None
        self._last_input: Optional[Mapping[str, float]] = None

    def suggest(self, fill: Optional[MutableMapping[str, object]] = None) -> Tuple[MutableMapping[str, object], Dict[str, object]]:
        candidate = self.carbs.suggest()
        raw = getattr(candidate, "suggestion", candidate)

        if isinstance(raw, Mapping):
            mapping = {str(key): raw[key] for key in raw}
        else:
            try:
                values = list(raw)  # type: ignore[arg-type]
            except TypeError as exc:
                raise TypeError("CARBS suggestion payload is not iterable or mapping") from exc
            if len(values) != len(self._flat_names):
                raise ValueError("CARBS suggestion length does not match configured parameters")
            mapping = {self._flat_names[idx]: float(values[idx]) for idx in range(len(values))}

        normalized: List[float] = []
        cleaned_mapping: Dict[str, float] = {}
        # Accept both slash and dot key styles
        mapping_norm: Dict[str, float] = {}
        for k, v in mapping.items():
            mapping_norm[str(k)] = float(v)
            mapping_norm[str(k).replace('/', '.')] = float(v)
        for name, space in self.hyperparameters.flat_spaces.items():
            if name not in mapping_norm:
                raise KeyError(f"CARBS suggestion missing hyperparameter '{name}'")
            value = mapping_norm[name]
            try:
                value_f = float(value)
            except (TypeError, ValueError) as exc:
                raise TypeError(f"CARBS suggestion for '{name}' is not numeric: {value!r}") from exc
            normalized.append(space.normalize(value_f))
            cleaned_mapping[name] = value_f

        suggestion_dict = self.hyperparameters.to_dict(normalized, fill)

        self._last_candidate = candidate
        self._last_input = cleaned_mapping

        info: Dict[str, object] = {"method": "carbs", "raw_params": dict(cleaned_mapping)}
        meta = getattr(candidate, "metadata", None)
        if isinstance(meta, Mapping):
            info.update(meta)
        return suggestion_dict, info

    def observe(self, hypers: Mapping[str, object], score: float, cost: float, *, is_failure: bool = False) -> None:
        if self._last_input is None or self._last_candidate is None:
            return
        try:
            from carbs import ObservationInParam  # type: ignore
        except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("CARBS not available. Install 'carbs' or 'pufferlib[carbs]' to use this sweep strategy.") from exc

        observation = ObservationInParam(
            input=self._last_input,
            output=float(score),
            cost=float(cost),
            is_failure=bool(is_failure),
        )
        self.carbs.observe(observation)
        self._last_input = None
        self._last_candidate = None

    # Persistence hooks are no-ops for external CARBS
    def state(self) -> Dict[str, object]:  # pragma: no cover - external
        return {}
    def load_state(self, data: Mapping[str, object]) -> None:  # pragma: no cover - external
        return


# ------------------------------
# Native efficient strategies
# ------------------------------

class EProtein:
    """Native, dependency-free GP-like strategy using kernel ridge regression.

    Approximates GP-UCB: fits RBF kernel ridge model to normalized inputs and
    selects suggestions that maximize mu(x) + kappa * sigma(x).
    """

    def __init__(self, sweep_config: Mapping[str, object], *, kappa: float = 1.2, l2: float = 1e-3, global_search_scale: float = 1.0, random_suggestions: int = 256) -> None:
        self.hyperparameters = Hyperparameters(sweep_config)
        self.kappa = float(kappa)
        self.l2 = float(l2)
        self.global_search_scale = float(global_search_scale)
        self.random_suggestions = int(random_suggestions)
        self.X: Optional[np.ndarray] = None  # shape (n, d)
        self.y: Optional[np.ndarray] = None  # shape (n,)
        self.alpha: Optional[np.ndarray] = None
        self.L: Optional[np.ndarray] = None  # Cholesky of (K + l2 I)
        self.length_scale: Optional[float] = None

    def _rbf(self, A: np.ndarray, B: np.ndarray, l: float) -> np.ndarray:
        # Pairwise squared distances
        A2 = np.sum(A*A, axis=1)[:, None]
        B2 = np.sum(B*B, axis=1)[None, :]
        d2 = np.maximum(A2 + B2 - 2 * A @ B.T, 0.0)
        return np.exp(-0.5 * d2 / (l*l))

    def _fit(self) -> None:
        if self.X is None or self.y is None or len(self.y) == 0:
            self.alpha = None; self.L = None; return
        X = self.X; y = self.y
        # Median distance heuristic for length scale
        if X.shape[0] >= 2:
            # sample pairs for efficiency
            idx = np.random.choice(X.shape[0], size=min(128, X.shape[0]), replace=False)
            Xs = X[idx]
            D = np.sqrt(np.maximum(0.0, np.sum((Xs[:, None, :] - Xs[None, :, :])**2, axis=2)))
            med = np.median(D[D>0]) if np.any(D>0) else 1.0
            self.length_scale = float(med) if med > 0 else 1.0
        else:
            self.length_scale = 1.0
        l = float(self.length_scale)
        K = self._rbf(X, X, l)
        K[np.diag_indices_from(K)] += self.l2
        try:
            self.L = np.linalg.cholesky(K)
            self.alpha = np.linalg.solve(self.L.T, np.linalg.solve(self.L, y))
        except np.linalg.LinAlgError:
            # Fallback to ridge (pinv)
            self.alpha = np.linalg.pinv(K) @ y
            self.L = None

    def _predict(self, Xc: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # Return (mu, sigma)
        if self.X is None or self.y is None or self.alpha is None or self.length_scale is None:
            mu = np.zeros(Xc.shape[0]); sigma = np.ones(Xc.shape[0])
            return mu, sigma
        k = self._rbf(Xc, self.X, float(self.length_scale))
        mu = k @ self.alpha
        if self.L is not None:
            v = np.linalg.solve(self.L, k.T)
            var = np.maximum(1.0 - np.sum(v*v, axis=0), 1e-6)
        else:
            # approximate variance
            var = np.maximum(1.0 - np.sum(k*k, axis=1) / max(1, self.X.shape[0]), 1e-6)
        return mu, np.sqrt(var)

    def suggest(self, fill: Optional[MutableMapping[str, object]] = None) -> Tuple[MutableMapping[str, object], Dict[str, object]]:
        # Candidate pool: random around centers + (optionally) around past bests
        centers = self.hyperparameters.search_centers[None, :]
        rs = self.hyperparameters.sample(self.random_suggestions, mu=centers, scale=self.global_search_scale)
        self._fit()
        mu, sigma = self._predict(rs)
        ucb = mu + self.kappa * sigma
        best_idx = int(np.argmax(ucb))
        best = rs[best_idx]
        return self.hyperparameters.to_dict(best, fill), {"method": "eprotein"}

    def observe(self, hypers: Mapping[str, object], score: float, cost: float, *, is_failure: bool = False) -> None:
        x = self.hyperparameters.from_dict(hypers)
        y = float(score)
        if self.X is None:
            self.X = x[None, :]; self.y = np.array([y], dtype=float)
        else:
            self.X = np.vstack([self.X, x[None, :]])
            self.y = np.concatenate([self.y, np.array([y], dtype=float)])

    def state(self) -> Dict[str, object]:  # pragma: no cover - simple JSON state
        return {
            "X": (self.X.tolist() if self.X is not None else []),
            "y": (self.y.tolist() if self.y is not None else []),
            "kappa": self.kappa,
            "l2": self.l2,
            "gscale": self.global_search_scale,
        }

    def load_state(self, data: Mapping[str, object]) -> None:  # pragma: no cover - simple JSON state
        try:
            X = np.array(data.get("X") or [])
            y = np.array(data.get("y") or [])
            if X.size and y.size and X.shape[0] == y.shape[0]:
                self.X = X; self.y = y
        except Exception:
            pass


class ECarbs:
    """Native, dependency-free cost-aware Pareto + local search.

    Maintains a Pareto set of (score,cost) and perturbs elite points to explore
    efficiently without external libraries.
    """

    def __init__(self, sweep_config: Mapping[str, object], *, global_search_scale: float = 1.0, elite_frac: float = 0.25, suggestions_per_round: int = 8) -> None:
        self.hyperparameters = Hyperparameters(sweep_config)
        self.global_search_scale = float(global_search_scale)
        self.elite_frac = float(elite_frac)
        self.suggestions_per_round = int(suggestions_per_round)
        self.success_observations: List[Dict[str, object]] = []

    def _elite(self) -> np.ndarray:
        if not self.success_observations:
            return self.hyperparameters.search_centers[None, :]
        obs = self.success_observations
        cand, idxs = pareto_points([{"output": float(o["output"]), "cost": float(o["cost"]) } for o in obs])
        X = np.array([obs[i]["input"] for i in idxs])
        # top K by output within Pareto
        k = max(1, int(len(X) * self.elite_frac))
        order = np.argsort([-obs[i]["output"] for i in idxs])[:k]
        return X[order]

    def suggest(self, fill: Optional[MutableMapping[str, object]] = None) -> Tuple[MutableMapping[str, object], Dict[str, object]]:
        elite = self._elite()
        n = max(self.suggestions_per_round, 1)
        mu = elite if len(elite.shape) == 2 else elite[None, :]
        cand = self.hyperparameters.sample(n, mu=mu, scale=self.global_search_scale)
        # Heuristic: pick first for diversity; otherwise pick one with highest projection on elite mean
        if cand.shape[0] == 1:
            best = cand[0]
        else:
            center = np.mean(mu, axis=0)
            scores = -np.linalg.norm(cand - center[None, :], axis=1)
            best = cand[int(np.argmax(scores))]
        return self.hyperparameters.to_dict(best, fill), {"method": "ecarbs"}

    def observe(self, hypers: Mapping[str, object], score: float, cost: float, *, is_failure: bool = False) -> None:
        x = self.hyperparameters.from_dict(hypers)
        self.success_observations.append({
            "input": x,
            "output": float(score),
            "cost": float(cost),
            "is_failure": bool(is_failure),
        })

    def state(self) -> Dict[str, object]:  # pragma: no cover - simple JSON state
        return {
            "observations": [
                {"input": list(map(float, obs.get("input", []))), "output": float(obs.get("output", 0.0)), "cost": float(obs.get("cost", 0.0)), "is_failure": bool(obs.get("is_failure", False))}
                for obs in self.success_observations
            ],
        }

    def load_state(self, data: Mapping[str, object]) -> None:  # pragma: no cover - simple JSON state
        try:
            self.success_observations = list(data.get("observations") or [])
        except Exception:
            pass


def get_strategy(name: str):
    key = name.lower()
    if key == "random":
        return RandomStrategy
    if key in {"pareto", "pareto_genetic", "paretogenetic"}:
        return ParetoGenetic
    if key == "protein":
        # Always return the Protein class; instantiation may error if deps missing
        return Protein
    if key == "carbs":
        # Always return the Carbs strategy class so callers can handle
        # dependency errors explicitly when instantiating.
        return Carbs
    if key in {"eprotein", "native_protein"}:
        return EProtein
    if key in {"ecarbs", "native_carbs"}:
        return ECarbs
    raise ValueError(f"Unknown sweep method: {name}")


__all__ = [
    "Hyperparameters",
    "RandomStrategy",
    "ParetoGenetic",
    "Protein",
    "Carbs",
    "pareto_points",
    "create_gp",
    "get_strategy",
]
