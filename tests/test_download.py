# Copyright (c) OpenMMLab. All rights reserved.
import pytest
from click.testing import CliRunner

from mim.commands.download import download
from mim.commands.install import cli as install
from mim.commands.uninstall import cli as uninstall


def setup_module():
    runner = CliRunner()
    result = runner.invoke(uninstall, ['mmcv-full', '--yes'])
    assert result.exit_code == 0
    result = runner.invoke(uninstall, ['mmcls', '--yes'])
    assert result.exit_code == 0


def test_download(tmp_path):
    runner = CliRunner()
    result = runner.invoke(install, ['mmcv-full', '--yes'])
    assert result.exit_code == 0

    with pytest.raises(ValueError):
        # version is not allowed
        download('mmcls==0.11.0', ['resnet18_8xb16_cifar10'])

    with pytest.raises(RuntimeError):
        # mmcls is not installed
        download('mmcls', ['resnet18_8xb16_cifar10'])

    with pytest.raises(ValueError):
        # invalid config
        download('mmcls==0.11.0', ['resnet18_b16x8_cifar1'])

    runner = CliRunner()
    # mim install mmcls --yes
    result = runner.invoke(install, [
        'mmcls', '--yes', '-f',
        'https://github.com/open-mmlab/mmclassification.git'
    ])
    assert result.exit_code == 0

    # mim download mmcls --config resnet18_8xb16_cifar10
    checkpoints = download('mmcls', ['resnet18_8xb16_cifar10'])
    assert checkpoints == ['resnet18_b16x8_cifar10_20210528-bd6371c8.pth']
    checkpoints = download('mmcls', ['resnet18_8xb16_cifar10'])

    # mim download mmcls --config resnet18_8xb16_cifar10 --dest tmp_path
    checkpoints = download('mmcls', ['resnet18_8xb16_cifar10'], tmp_path)
    assert checkpoints == ['resnet18_b16x8_cifar10_20210528-bd6371c8.pth']


def teardown_module():
    runner = CliRunner()
    result = runner.invoke(uninstall, ['mmcv-full', '--yes'])
    assert result.exit_code == 0
    result = runner.invoke(uninstall, ['mmcls', '--yes'])
    assert result.exit_code == 0
