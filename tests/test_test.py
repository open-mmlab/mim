# Copyright (c) OpenMMLab. All rights reserved.
import sys

import pytest
import torch
from click.testing import CliRunner

from mim.commands.install import cli as install
from mim.commands.test import cli as test


@pytest.mark.parametrize('device', [
    'cpu',
    pytest.param(
        'cuda',
        marks=pytest.mark.skipif(
            not torch.cuda.is_available(), reason='requires CUDA support')),
])
def test_test(device, tmp_path):
    sys.path.append(str(tmp_path))
    runner = CliRunner()
    result = runner.invoke(install, ['mmcls', '--yes', '-t', str(tmp_path)])
    assert result.exit_code == 0
    # Since mmcv-full not in mminstall.txt of mmcls, we install mmcv-full here.
    result = runner.invoke(
        install,
        ['mmcv-full', '--yes', '-t', str(tmp_path)])
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
