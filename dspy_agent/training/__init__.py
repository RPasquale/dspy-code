"""
Training and model management for DSPy agent.
"""

from .train_codegen import *
from .train_gepa import *
from .train_orchestrator import *
from .deploy_model import *
from .deploy import *
from .autogen_dataset import *
__all__ = [
    'run_gepa_codegen', 'run_gepa', 'run_gepa_with_val', 'evaluate_on_set',
    'run_gepa_orchestrator', 'run_gepa_orchestrator_with_val', 'evaluate_orchestrator',
    'DeploymentLogger', 'bootstrap_datasets', 'bootstrap_datasets_with_splits',
]

try:
    from .rl_sweep import run_sweep, load_sweep_config, SweepSettings, SweepOutcome, describe_default_hparams
except Exception:  # pragma: no cover - optional dependency (pufferlib/torch build)
    run_sweep = None  # type: ignore
    load_sweep_config = None  # type: ignore
    SweepSettings = None  # type: ignore
    SweepOutcome = None  # type: ignore
    describe_default_hparams = None  # type: ignore
else:
    __all__.extend([
        'run_sweep', 'load_sweep_config', 'SweepSettings', 'SweepOutcome',
        'describe_default_hparams'
    ])
