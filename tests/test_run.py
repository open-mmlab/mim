# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch
from click.testing import CliRunner

from mim.commands.install import cli as install
from mim.commands.run import cli as run
from mim.commands.uninstall import cli as uninstall


def setup_module():
    runner = CliRunner()
    result = runner.invoke(uninstall, ['mmcv-full', '--yes'])
    assert result.exit_code == 0
    result = runner.invoke(uninstall, ['mmcls', '--yes'])
    assert result.exit_code == 0


@pytest.mark.parametrize('device,gpus', [
    ('cpu', 0),
    pytest.param(
        'cuda',
        1,
        marks=pytest.mark.skipif(
            not torch.cuda.is_available(), reason='requires CUDA support')),
])
def test_run(device, gpus, tmp_path):
    runner = CliRunner()
    result = runner.invoke(install, ['mmcls', '--yes'])
    assert result.exit_code == 0
    # Since `mminstall.txt` is not included in the distribution of
    # mmcls<=0.23.1, we need to install mmcv-full manually.
    result = runner.invoke(install, ['mmcv-full', '--yes'])
    assert result.exit_code == 0

    result = runner.invoke(run, [
        'mmcls', 'train', 'tests/data/lenet5_mnist.py', f'--gpus={gpus}',
        f'--work-dir={tmp_path}'
    ])
    assert result.exit_code == 0
    result = runner.invoke(run, [
        'mmcls', 'test', 'tests/data/lenet5_mnist.py',
        'tests/data/epoch_1.pth', f'--device={device}', '--metrics=accuracy'
    ])
    assert result.exit_code == 0
    result = runner.invoke(run, [
        'mmcls', 'xxx', 'tests/data/lenet5_mnist.py', 'tests/data/epoch_1.pth',
        f'--gpus={gpus}', '--metrics=accuracy'
    ])
    assert result.exit_code != 0
    result = runner.invoke(run, [
        'mmcls', 'test', 'tests/data/xxx.py', 'tests/data/epoch_1.pth',
        f'--device={device}', '--metrics=accuracy'
    ])
    assert result.exit_code != 0


def teardown_module():
    runner = CliRunner()
    result = runner.invoke(uninstall, ['mmcv-full', '--yes'])
    assert result.exit_code == 0
    result = runner.invoke(uninstall, ['mmcls', '--yes'])
    assert result.exit_code == 0
