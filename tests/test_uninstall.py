# Copyright (c) OpenMMLab. All rights reserved.
from click.testing import CliRunner

from mim.commands.install import cli as install
from mim.commands.list import list_package
from mim.commands.uninstall import cli as uninstall


def setup_module():
    runner = CliRunner()
    result = runner.invoke(uninstall, ['mmcv-full', '--yes'])
    assert result.exit_code == 0
    result = runner.invoke(uninstall, ['mmcls', '--yes'])
    assert result.exit_code == 0


def test_uninstall():
    runner = CliRunner()

    # mim install mmdet mmsegmentation
    result = runner.invoke(install, ['mmdet', 'mmsegmentation', '--yes'])
    assert result.exit_code == 0

    # check if install success
    result = list_package()
    installed_packages = [item[0] for item in result]
    assert 'mmdet' in installed_packages
    assert 'mmsegmentation' in installed_packages
    assert 'mmcv-full' in installed_packages
    # `mim install mmsegmentation` will install mim extra requirements (via
    # mminstall.txt) automatically since PR#132, so we got installed mmcls here.  # noqa: E501
    assert 'mmcls' in installed_packages

    # mim uninstall mmdet mmsegmentation
    result = runner.invoke(uninstall, ['mmdet', 'mmsegmentation', '--yes'])
    assert result.exit_code == 0

    # check if uninstall success
    result = list_package()
    installed_packages = [item[0] for item in result]
    assert 'mmdet' not in installed_packages
    assert 'mmsegmentation' not in installed_packages

    # mim uninstall mmcls mmcv-full
    result = runner.invoke(uninstall, ['mmcv-full', 'mmmcls', '--yes'])
    assert result.exit_code == 0

    # check if uninstall success
    result = list_package()
    installed_packages = [item[0] for item in result]
    assert 'mmdet' not in installed_packages
    assert 'mmsegmentation' not in installed_packages


def teardown_module():
    runner = CliRunner()
    result = runner.invoke(uninstall, ['mmcv-full', '--yes'])
    assert result.exit_code == 0
    result = runner.invoke(uninstall, ['mmcls', '--yes'])
    assert result.exit_code == 0
