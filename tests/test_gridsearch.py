# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch
from click.testing import CliRunner

from mim.commands.gridsearch import cli as gridsearch
from mim.commands.install import cli as install


@pytest.mark.run(order=-1)
@pytest.mark.parametrize('gpus', [
    0,
    pytest.param(
        1,
        marks=pytest.mark.skipif(
            not torch.cuda.is_available(), reason='requires CUDA support')),
])
def test_gridsearch(gpus, tmp_path):
    runner = CliRunner()
    result = runner.invoke(install, ['mmcls', '--yes'])
    assert result.exit_code == 0
    # Since mmcv-full not in mminstall.txt of mmcls, we install mmcv-full here.
    result = runner.invoke(install, ['mmcv-full', '--yes'])
    assert result.exit_code == 0

    args1 = [
        'mmcls', 'tests/data/lenet5_mnist.py', f'--gpus={gpus}',
        f'--work-dir={tmp_path}', '--search-args', '--optimizer.lr 1e-3 1e-4'
    ]
    args2 = [
        'mmcls', 'tests/data/lenet5_mnist.py', f'--gpus={gpus}',
        f'--work-dir={tmp_path}', '--search-args',
        '--optimizer.weight_decay 1e-3 1e-4'
    ]
    args3 = [
        'mmcls', 'tests/data/xxx.py', f'--gpus={gpus}',
        f'--work-dir={tmp_path}', '--search-args', '--optimizer.lr 1e-3 1e-4'
    ]
    args4 = [
        'mmcls', 'tests/data/lenet5_mnist.py', f'--gpus={gpus}',
        f'--work-dir={tmp_path}', '--search-args'
    ]

    result = runner.invoke(gridsearch, args1)
    assert result.exit_code == 0

    result = runner.invoke(gridsearch, args2)
    assert result.exit_code == 0

    result = runner.invoke(gridsearch, args3)
    assert result.exit_code != 0

    result = runner.invoke(gridsearch, args4)
    assert result.exit_code != 0
