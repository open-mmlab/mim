# Copyright (c) OpenMMLab. All rights reserved.
import torch
from click.testing import CliRunner

from mim.commands.install import cli as install
from mim.commands.test import cli as test
from mim.commands.uninstall import cli as uninstall


def setup_module():
    runner = CliRunner()
    result = runner.invoke(uninstall, ['mmcv-full', '--yes'])
    assert result.exit_code == 0
    result = runner.invoke(uninstall, ['mmcls', '--yes'])
    assert result.exit_code == 0


def test_test():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    runner = CliRunner()
    result = runner.invoke(install, ['mmcls', '--yes'])
    assert result.exit_code == 0

    result = runner.invoke(test, [
        'mmcls', 'tests/data/lenet5_mnist.py', '--checkpoint',
        'tests/data/epoch_1.pth', f'--device={device}', '--metrics=accuracy'
    ])
    assert result.exit_code == 0
    result = runner.invoke(test, [
        'mmcls', 'tests/data/xxx.py', '--checkpoint', 'tests/data/epoch_1.pth',
        f'--device={device}', '--metrics=accuracy'
    ])
    assert result.exit_code != 0
    result = runner.invoke(test, [
        'mmcls', 'tests/data/lenet5_mnist.py', '--checkpoint',
        'tests/data/xxx.pth', f'--device={device}', '--metrics=accuracy'
    ])
    assert result.exit_code != 0


def teardown_module():
    runner = CliRunner()
    result = runner.invoke(uninstall, ['mmcv-full', '--yes'])
    assert result.exit_code == 0
    result = runner.invoke(uninstall, ['mmcls', '--yes'])
    assert result.exit_code == 0
