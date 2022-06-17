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


def test_list():
    runner = CliRunner()
    # mim install mmcls==0.23.0 --yes
    result = runner.invoke(install, ['mmcls==0.23.1', '--yes'])
    from mim.commands.install import install as install_
    install_(['mmcls==0.23.1'])
    assert result.exit_code == 0
    # mim list
    target = ('mmcls', '0.23.1',
              'https://github.com/open-mmlab/mmclassification')
    result = list_package()
    assert target in result


def teardown_module():
    runner = CliRunner()
    result = runner.invoke(uninstall, ['mmcv-full', '--yes'])
    assert result.exit_code == 0
    result = runner.invoke(uninstall, ['mmcls', '--yes'])
    assert result.exit_code == 0
