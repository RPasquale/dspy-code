import json
import types
import io
import os
import time
from pathlib import Path
from unittest.mock import patch

from rich.console import Console

from dspy_agent.cli import _record_gepa_outcome, AutoTrainingLoop
from dspy_agent.context.context_manager import ContextManager
from dspy_agent.rl.rlkit import ToolchainConfig, ToolchainExecutor, ToolAction
from dspy_agent.streaming.streamkit import TRAINER_SETTINGS_PATH, Trainer, LocalBus


def _make_progress_file(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    records = [
        {"step": 0, "score": 0.4},
        {"step": 1, "score": 0.55},
        {"step": 2, "score": 0.6},
    ]
    with path.open('w') as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    return path


def test_record_gepa_outcome_updates_settings_and_rl(tmp_path: Path):
    rl_cfg = tmp_path / '.dspy_rl.json'
    rl_cfg.write_text(json.dumps({
        "policy": "epsilon-greedy",
        "weights": {"pass_rate": 1.0},
    }, indent=2))

    progress_path = _make_progress_file(tmp_path / '.gepa_code' / 'progress.jsonl')

    detailed = types.SimpleNamespace(best_candidate={
        'id': 'candidate-1',
        'prompt': 'Always write tests before changes.',
        'reward_delta': 0.25,
        'verifier_weights': {'pass_rate': 1.2, 'lint_ok': 0.35},
    })
    optimized = types.SimpleNamespace(detailed_results=detailed)

    _record_gepa_outcome('code', optimized, tmp_path, progress_path)

    settings_path = tmp_path / TRAINER_SETTINGS_PATH.name
    assert settings_path.exists()
    settings = json.loads(settings_path.read_text())

    prompts = settings.get('prompts', {})
    assert 'patch' in prompts
    patch_registry = prompts['patch']
    assert patch_registry.get('active') == 'candidate-1'
    candidates = patch_registry.get('candidates', [])
    assert any(c.get('prompt') == 'Always write tests before changes.' for c in candidates)

    reward_weights = settings.get('reward_weights', {})
    assert reward_weights.get('lint_ok') == 0.35

    rl_data = json.loads(rl_cfg.read_text())
    assert rl_data['weights']['lint_ok'] == 0.35


def test_trainer_uses_prompt_registry(tmp_path: Path):
    settings = {
        'actions': ['run_tests', 'lint', 'build', 'patch'],
        'prompts': {
            'patch': {
                'active': 'prompt-123',
                'candidates': [
                    {
                        'id': 'prompt-123',
                        'prompt': 'Prefer small patches tied to failing tests.',
                        'reward_delta': 0.5,
                    }
                ],
            }
        },
        'reward_weights': {
            'pass_rate': 1.0,
            'coverage_delta': 0.4,
        },
    }
    (tmp_path / TRAINER_SETTINGS_PATH.name).write_text(json.dumps(settings, indent=2))

    trainer = Trainer(tmp_path, LocalBus(), containers=['app'], min_batch=1, interval_sec=60.0)

    action_args = trainer._build_action_args('Recent failing tests...')
    assert 'patch' in action_args
    patch_args = action_args['patch']
    assert patch_args.get('prompt') == 'Prefer small patches tied to failing tests.'
    weights, _, _ = trainer._reward_components()
    assert weights.get('coverage_delta') == 0.4


def test_trainer_shell_actions(tmp_path: Path):
    settings = {
        'actions': ['shell_ls', 'shell_run'],
        'shell_actions': {
            'shell_run': {'cmd': 'echo hello world'},
            'shell_cat': 'CHANGELOG.md',
        },
        'shell_timeout': 45,
    }
    (tmp_path / 'CHANGELOG.md').write_text('demo\n')
    (tmp_path / TRAINER_SETTINGS_PATH.name).write_text(json.dumps(settings, indent=2))

    trainer = Trainer(tmp_path, LocalBus(), containers=['app'], min_batch=1, interval_sec=60.0)
    assert trainer.shell_timeout == 45

    action_args = trainer._build_action_args('')
    assert 'shell_ls' in action_args
    assert action_args['shell_ls']['timeout'] == 45
    assert action_args['shell_run']['cmd'] == 'echo hello world'


def test_auto_training_loop_status_file(tmp_path: Path):
    console = Console(file=io.StringIO())
    loop = AutoTrainingLoop(tmp_path, None, console=console, modules=['code'], interval_sec=1, initial_delay_sec=0, ollama=False)

    dataset = tmp_path / '.dspy_data' / 'code_train.jsonl'
    dataset.parent.mkdir(parents=True, exist_ok=True)
    dataset.write_text('{"task":"t"}\n')

    fake_prog = types.SimpleNamespace(detailed_results=None)

    with patch('dspy_agent.cli.bootstrap_datasets', return_value={'code': dataset}), \
         patch('dspy_agent.cli._maybe_configure_lm', return_value=object()), \
         patch('dspy_agent.cli.run_gepa', return_value=fake_prog), \
         patch('dspy_agent.cli._record_gepa_outcome') as record_mock, \
         patch('dspy_agent.cli.AutoTrainingLoop._run_rl_training', return_value=0.42):
        loop._run_once()

    status_path = tmp_path / '.dspy_auto_status.json'
    assert status_path.exists()
    status = json.loads(status_path.read_text())
    assert status.get('status') == 'idle'
    assert status.get('module') == 'code'
    assert status.get('avg_reward') == 0.42
    record_mock.assert_called_once()


def test_trainer_group_advantage_updates(tmp_path: Path):
    settings = {
        'actions': ['run_tests', 'lint'],
        'group_advantage': True,
        'group_size': 2,
    }
    (tmp_path / TRAINER_SETTINGS_PATH.name).write_text(json.dumps(settings, indent=2))

    trainer = Trainer(tmp_path, LocalBus(), containers=['app'], min_batch=1, interval_sec=60.0)
    trainer._contexts = [{"context": ["error"], "timestamp": time.time()} for _ in range(3)]

    class DummyBandit:
        def __init__(self):
            self.next_action = 0
            self.updates = []

        def select(self, ctx=None):
            action = self.next_action
            self.next_action = (self.next_action + 1) % 2
            return action

        def update(self, action, reward, ctx=None):
            self.updates.append((action, reward))

    class DummyEnv:
        def __init__(self, *args, **kwargs):
            self._actions = ['run_tests', 'lint']

        @property
        def action_dim(self):
            return len(self._actions)

        @property
        def action_names(self):
            return list(self._actions)

        def reset(self):
            return [], {}

        def step(self, action):
            reward = 1.0 if action == 0 else 0.2
            return [], reward, True, False, {'tool': self._actions[action]}

    bandit = DummyBandit()
    orig = os.environ.get('RL_BACKGROUND_STEPS')
    os.environ['RL_BACKGROUND_STEPS'] = '2'
    try:
        with patch('dspy_agent.streaming.streamkit.RLToolEnv', DummyEnv), \
             patch('dspy_agent.streaming.streamkit.ToolchainExecutor'), \
             patch('dspy_agent.streaming.streamkit.detect_toolchain'), \
             patch('dspy_agent.streaming.streamkit._make_bandit', return_value=bandit):
            trainer._train_on_contexts()
    finally:
        if orig is None:
            del os.environ['RL_BACKGROUND_STEPS']
        else:
            os.environ['RL_BACKGROUND_STEPS'] = orig

    # First action should receive positive reward advantage, second zero
    assert bandit.updates[0][0] == 0 and bandit.updates[0][1] > 0
    assert bandit.updates[1][0] == 1 and bandit.updates[1][1] == 0.0


def test_toolchain_executor_quality_checks(tmp_path: Path):
    target = tmp_path / 'foo.py'
    target.write_text('print("hi")\n')
    config = ToolchainConfig(workspace=tmp_path)
    executor = ToolchainExecutor(config)

    patch_text = """--- foo.py\n+++ foo.py\n@@\n-print(\"hi\")\n+print(\"hello\")\n"""

    class DummyLocator:
        def __call__(self, **kwargs):
            return types.SimpleNamespace(file_candidates='foo.py', notes='')

    class DummyEdit:
        def __call__(self, **kwargs):
            return types.SimpleNamespace(patch=patch_text, rationale='ok')

    class DummyVerifier:
        def __call__(self, **kwargs):
            return types.SimpleNamespace(verdict='pass')

    class DummyPlanner:
        def __call__(self, **kwargs):
            return types.SimpleNamespace(commands=None)

    run_calls = []

    def fake_run(cmd, cwd, timeout):
        run_calls.append(cmd)
        if 'lint' in cmd:
            return False, 'lint fail', 0.01
        return True, '', 0.01

    with patch('dspy_agent.rl.rlkit.FileLocator', lambda: DummyLocator()), \
         patch('dspy_agent.rl.rlkit.CodeEdit', lambda use_cot=True: DummyEdit()), \
         patch('dspy_agent.rl.rlkit.PatchVerifier', lambda **kwargs: DummyVerifier()), \
         patch('dspy_agent.rl.rlkit.TestPlanner', lambda *a, **k: DummyPlanner()), \
         patch('dspy_agent.rl.rlkit.apply_unified_patch', lambda patch, ws: (True, 'applied')), \
         patch('dspy_agent.rl.rlkit.revert_unified_patch', lambda patch, ws: (True, 'reverted')), \
         patch('dspy_agent.rl.rlkit.summarize_patch', lambda patch: {'files': 1, 'added_lines': 1, 'removed_lines': 0}), \
         patch('dspy_agent.rl.rlkit._run', fake_run):
        result = executor(ToolAction.PATCH, {
            'task': 'Fix issue',
            'context': 'error',
            'quality_checks': {'lint': 'lint-cmd', 'syntax': 'syntax-cmd'},
        })

    metrics = result.metrics
    assert metrics.get('quality_lint') == 0.0
    assert 'quality_syntax' not in metrics
    assert metrics.get('pass_rate') == 0.0
    assert run_calls.count('lint-cmd') == 1


def test_context_manager_bundle(tmp_path: Path):
    log_dir = tmp_path / 'logs'
    log_dir.mkdir()
    (log_dir / 'app.log').write_text("ERROR something broke\n")
    history_dir = tmp_path / '.dspy_patches'
    history_dir.mkdir()
    entry = {
        'timestamp': time.time(),
        'prompt_id': 'p1',
        'result': 'success',
        'metrics': {'pass_rate': 1.0, 'blast_radius': 2.0},
        'high_confidence': True,
        'file_candidates': 'foo.py,bar.py',
    }
    (history_dir / 'history.jsonl').write_text(json.dumps(entry) + "\n")
    cm = ContextManager(tmp_path, log_dir)
    bundle = cm.build_patch_context('Fix issue')
    assert 'Recent errors' in bundle['text']
    assert bundle['patches'] and bundle['patches'][0]['prompt_id'] == 'p1'
    assert 'foo.py' in bundle.get('file_hints', '')
