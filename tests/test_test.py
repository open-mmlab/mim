# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch
from click.testing import CliRunner

from mim.commands.install import cli as install
from mim.commands.test import cli as test
from mim.commands.uninstall import cli as uninstall


def setup_module():
    runner = CliRunner()
    result = runner.invoke(uninstall, ['mmcv-full', '--yes'])
    assert result.exit_code == 0
    result = runner.invoke(uninstall, ['mmcv', '--yes'])
    assert result.exit_code == 0
    result = runner.invoke(uninstall, ['mmcls', '--yes'])
    assert result.exit_code == 0


@pytest.mark.parametrize('device', [
    'cpu',
    pytest.param(
        'cuda',
        marks=pytest.mark.skipif(
            not torch.cuda.is_available(), reason='requires CUDA support')),
])
def test_test(device):
    runner = CliRunner()
    result = runner.invoke(install, ['mmcls>=1.0.0rc0', '--yes'])
    assert result.exit_code == 0
    result = runner.invoke(install, ['mmengine', '--yes'])
    assert result.exit_code == 0
    result = runner.invoke(install, ['mmcv>=2.0.0rc0', '--yes'])
    assert result.exit_code == 0

    result = runner.invoke(test, [
        'mmcls', 'tests/data/lenet5_mnist.py', '--checkpoint',
        'tests/data/epoch_1.pth'
    ])
    assert result.exit_code == 0
    result = runner.invoke(test, [
        'mmcls', 'tests/data/xxx.py', '--checkpoint', 'tests/data/epoch_1.pth'
    ])
    assert result.exit_code != 0
    result = runner.invoke(test, [
        'mmcls', 'tests/data/lenet5_mnist.py', '--checkpoint',
        'tests/data/xxx.pth'
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
