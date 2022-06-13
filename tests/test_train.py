# Copyright (c) OpenMMLab. All rights reserved.
import torch
from click.testing import CliRunner

from mim.commands.install import cli as install
from mim.commands.train import cli as train
from mim.commands.uninstall import cli as uninstall


def setup_module():
    runner = CliRunner()
    result = runner.invoke(uninstall, ['mmcv-full', '--yes'])
    assert result.exit_code == 0
    result = runner.invoke(uninstall, ['mmcls', '--yes'])
    assert result.exit_code == 0


def test_train(tmp_path):
    gpus = 1 if torch.cuda.is_available() else 0

    runner = CliRunner()
    result = runner.invoke(install, ['mmcls', '--yes'])
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


def teardown_module():
    runner = CliRunner()
    result = runner.invoke(uninstall, ['mmcv-full', '--yes'])
    assert result.exit_code == 0
    result = runner.invoke(uninstall, ['mmcls', '--yes'])
    assert result.exit_code == 0
