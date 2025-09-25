from pathlib import Path


def test_slurm_script_present_and_has_placeholders():
    p = Path('deploy/slurm/train_ddp.sbatch')
    assert p.exists()
    text = p.read_text()
    assert '#SBATCH -N ${NODES:-2}' in text
    assert '--nproc_per_node "${GPUS:-4}"' in text
    assert 'torchrun' in text


def test_trainer_stub_loadable():
    import importlib.util
    spec = importlib.util.spec_from_file_location('trainer_fsdp', 'rl/training/trainer_fsdp.py')
    mod = importlib.util.module_from_spec(spec)  # type: ignore
    assert spec and spec.loader
    spec.loader.exec_module(mod)  # type: ignore
    assert hasattr(mod, 'run_training')

