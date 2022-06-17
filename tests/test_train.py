# Copyright (c) OpenMMLab. All rights reserved.
import sys

import pytest
import torch
from click.testing import CliRunner

from mim.commands.install import cli as install
from mim.commands.train import cli as train


@pytest.mark.parametrize('gpus', [
    0,
    pytest.param(
        1,
        marks=pytest.mark.skipif(
            not torch.cuda.is_available(), reason='requires CUDA support')),
])
def test_train(gpus, tmp_path):
    sys.path.append(str(tmp_path))
    runner = CliRunner()
    result = runner.invoke(install, ['mmcls', '--yes', '-t', str(tmp_path)])
    assert result.exit_code == 0
    # Since mmcv-full not in mminstall.txt of mmcls, we install mmcv-full here.
    result = runner.invoke(
        install,
        ['mmcv-full', '--yes', '-t', str(tmp_path)])
    assert result.exit_code == 0

    result = runner.invoke(train, [
        'mmcls', 'tests/data/lenet5_mnist.py', f'--gpus={gpus}',
        f'--work-dir={tmp_path}'
    ])
    assert result.exit_code == 0

    result = runner.invoke(train, [
        'mmcls', 'tests/data/xxx.py', f'--gpus={gpus}',
        f'--work-dir={tmp_path}'
    ])
    assert result.exit_code != 0
