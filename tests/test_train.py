# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch
from click.testing import CliRunner

from mim.commands.install import cli as install
from mim.commands.train import cli as train
from mim.commands.uninstall import cli as uninstall


def setup_module():
    runner = CliRunner()
    result = runner.invoke(uninstall, ['mmcv-full', '--yes'])
    assert result.exit_code == 0
    result = runner.invoke(uninstall, ['mmcv', '--yes'])
    assert result.exit_code == 0
    result = runner.invoke(uninstall, ['mmcls', '--yes'])
    assert result.exit_code == 0


@pytest.mark.parametrize('gpus', [
    0,
    pytest.param(
        1,
        marks=pytest.mark.skipif(
            not torch.cuda.is_available(), reason='requires CUDA support')),
])
def test_train(gpus, tmp_path):
    runner = CliRunner()
    result = runner.invoke(install, ['mmcls>=1.0.0rc0', '--yes'])
    assert result.exit_code == 0
    result = runner.invoke(install, ['mmengine', '--yes'])
    assert result.exit_code == 0
    result = runner.invoke(install, ['mmcv>=2.0.0rc0', '--yes'])
    assert result.exit_code == 0

    result = runner.invoke(train, [
        'mmcls', 'tests/data/lenet5_mnist_2.0.py', f'--gpus={gpus}',
        f'--work-dir={tmp_path}'
    ])
    assert result.exit_code == 0

    result = runner.invoke(train, [
        'mmcls', 'tests/data/xxx.py', f'--gpus={gpus}',
        f'--work-dir={tmp_path}'
    ])
    assert result.exit_code != 0


def teardown_module():
    runner = CliRunner()
    result = runner.invoke(uninstall, ['mmcv-full', '--yes'])
    assert result.exit_code == 0
    result = runner.invoke(uninstall, ['mmcv', '--yes'])
    assert result.exit_code == 0
    result = runner.invoke(uninstall, ['mmcls', '--yes'])
    assert result.exit_code == 0
