# Copyright (c) OpenMMLab. All rights reserved.

import torch
from click.testing import CliRunner

from mim.commands.gridsearch import cli as gridsearch
from mim.commands.install import cli as install
from mim.commands.uninstall import cli as uninstall


def setup_module():
    runner = CliRunner()
    result = runner.invoke(uninstall, ['mmcv-full', '--yes'])
    assert result.exit_code == 0
    result = runner.invoke(uninstall, ['mmcls', '--yes'])
    assert result.exit_code == 0


def test_gridsearch(tmp_path):
    gpus = 1 if torch.cuda.is_available() else 0

    runner = CliRunner()
    result = runner.invoke(install, ['mmcls', '--yes'])
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


def teardown_module():
    runner = CliRunner()
    result = runner.invoke(uninstall, ['mmcv-full', '--yes'])
    assert result.exit_code == 0
    result = runner.invoke(uninstall, ['mmcls', '--yes'])
    assert result.exit_code == 0
