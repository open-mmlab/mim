# Copyright (c) OpenMMLab. All rights reserved.
import pytest
from click.testing import CliRunner

from mim.commands.download import download
from mim.commands.install import cli as install
from mim.commands.uninstall import cli as uninstall


def setup_module():
    runner = CliRunner()
    result = runner.invoke(uninstall, ['mmcv', '--yes'])
    assert result.exit_code == 0
    result = runner.invoke(uninstall, ['mmengine', '--yes'])
    assert result.exit_code == 0
    result = runner.invoke(uninstall, ['mmpretrain', '--yes'])
    assert result.exit_code == 0


def test_download(tmp_path):
    runner = CliRunner()
    result = runner.invoke(install, ['mmcv', '--yes'])
    assert result.exit_code == 0
    result = runner.invoke(install, ['mmengine', '--yes'])
    assert result.exit_code == 0

    with pytest.raises(ValueError):
        # version is not allowed
        download('mmpretrain==0.11.0', ['resnet18_8xb16_cifar10'])

    with pytest.raises(RuntimeError):
        # mmpretrain is not installed
        download('mmpretrain', ['resnet18_8xb16_cifar10'])

    with pytest.raises(ValueError):
        # invalid config
        download('mmpretrain', ['resnet18_b16x8_cifar1'])

    runner = CliRunner()
    # mim install mmpretrain
    result = runner.invoke(install, ['mmpretrain'])
    assert result.exit_code == 0

    # mim download mmpretrain --config resnet18_8xb16_cifar10
    checkpoints = download('mmpretrain', ['resnet18_8xb16_cifar10'])
    assert checkpoints == ['resnet18_b16x8_cifar10_20210528-bd6371c8.pth']
    checkpoints = download('mmpretrain', ['resnet18_8xb16_cifar10'])

    # mim download mmpretrain --config resnet18_8xb16_cifar10 --dest tmp_path
    checkpoints = download('mmpretrain', ['resnet18_8xb16_cifar10'], tmp_path)
    assert checkpoints == ['resnet18_b16x8_cifar10_20210528-bd6371c8.pth']


def teardown_module():
    runner = CliRunner()
    result = runner.invoke(uninstall, ['mmcv', '--yes'])
    assert result.exit_code == 0
    result = runner.invoke(uninstall, ['mmengine', '--yes'])
    assert result.exit_code == 0
    result = runner.invoke(uninstall, ['mmpretrain', '--yes'])
    assert result.exit_code == 0
