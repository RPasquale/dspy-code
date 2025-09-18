from pathlib import Path

from dspy_agent.rl.rlkit import ToolAction, ToolchainConfig, ToolchainExecutor


def test_tool_action_aliases_cover_cli():
    assert ToolAction.from_any('ls') == ToolAction.SHELL_LS
    assert ToolAction.from_any('pwd') == ToolAction.SHELL_PWD
    assert ToolAction.from_any('cat') == ToolAction.SHELL_CAT
    assert ToolAction.from_any('cd') == ToolAction.SHELL_CD
    assert ToolAction.from_any('shell') == ToolAction.SHELL_RUN


def test_toolchain_executor_shell_ops(tmp_path: Path):
    (tmp_path / 'README.md').write_text('hello world\n')
    (tmp_path / 'subdir').mkdir()
    cfg = ToolchainConfig(workspace=tmp_path, shell_defaults={'shell_run': 'echo $PWD'})
    executor = ToolchainExecutor(cfg)

    ls_result = executor(ToolAction.SHELL_LS, {})
    assert ls_result.metrics['shell_exit_code'] == 0
    assert ls_result.metrics['cwd'] == str(tmp_path)

    pwd_result = executor(ToolAction.SHELL_PWD, {})
    assert pwd_result.metrics['shell_exit_code'] == 0
    assert Path(pwd_result.metrics['shell_stdout']).resolve() == tmp_path.resolve()

    cat_result = executor(ToolAction.SHELL_CAT, {'path': 'README.md'})
    assert cat_result.metrics['shell_exit_code'] == 0
    assert cat_result.metrics['shell_target'].endswith('README.md')

    bad_cd = executor(ToolAction.SHELL_CD, {'path': '../'})
    assert bad_cd.metrics['shell_cd_ok'] == 0.0

    good_cd = executor(ToolAction.SHELL_CD, {'path': 'subdir'})
    assert good_cd.metrics['shell_cd_ok'] == 1.0
    assert executor._is_within_workspace(Path(good_cd.metrics['cwd']))

    run_result = executor(ToolAction.SHELL_RUN, {'cmd': 'echo hi'})
    assert run_result.metrics['shell_exit_code'] == 0
    assert 'hi' in run_result.info.get('stdout', '')

