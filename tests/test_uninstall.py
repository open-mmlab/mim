# Copyright (c) OpenMMLab. All rights reserved.
import traceback

import pip
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

    import site
    import subprocess
    subprocess.check_call(['pip', 'list'])
    subprocess.check_call(['ls', site.getsitepackages()[0]])
    subprocess.check_call(['pip', 'freeze'])
    print(site.getsitepackages())

    # mim install mmsegmentation --yes
    result = runner.invoke(install, ['mmsegmentation', '--yes'])
    traceback.print_exception(*result.exc_info)
    print(result.output)
    print(pip.__version__)
    assert result.exit_code == 0

    # check if install success
    result = list_package()
    installed_packages = [item[0] for item in result]
    assert 'mmsegmentation' in installed_packages
    assert 'mmcv-full' in installed_packages
    # `mim install mmsegmentation` will install mim extra requirements (via
    # mminstall.txt) automatically since PR#132, so we got installed mmcls here.  # noqa: E501
    assert 'mmcls' in installed_packages

    # mim uninstall mmsegmentation --yes
    result = runner.invoke(uninstall, ['mmsegmentation', '--yes'])
    assert result.exit_code == 0

    # check if uninstall success
    result = list_package()
    installed_packages = [item[0] for item in result]
    assert 'mmsegmentation' not in installed_packages

    # mim uninstall mmcls mmcv-full --yes
    result = runner.invoke(uninstall, ['mmcls', 'mmcv-full', '--yes'])
    assert result.exit_code == 0

    # check if uninstall success
    result = list_package()
    installed_packages = [item[0] for item in result]
    assert 'mmcls' not in installed_packages
    assert 'mmcv-full' not in installed_packages


def teardown_module():
    runner = CliRunner()
    result = runner.invoke(uninstall, ['mmcv-full', '--yes'])
    assert result.exit_code == 0
    result = runner.invoke(uninstall, ['mmcls', '--yes'])
    assert result.exit_code == 0
