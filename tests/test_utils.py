# Copyright (c) OpenMMLab. All rights reserved.
from click.testing import CliRunner

from mim.commands.install import cli as install
from mim.commands.uninstall import cli as uninstall
from mim.utils import get_github_url, parse_home_page


def setup_module():
    runner = CliRunner()
    result = runner.invoke(uninstall, ['mmcv-full', '--yes'])
    assert result.exit_code == 0
    result = runner.invoke(uninstall, ['mmcls', '--yes'])
    assert result.exit_code == 0


def test_parse_home_page():
    runner = CliRunner()
    result = runner.invoke(install, ['mmcls', '--yes'])
    assert result.exit_code == 0
    assert parse_home_page(
        'mmcls') == 'https://github.com/open-mmlab/mmclassification'
    result = runner.invoke(uninstall, ['mmcls', '--yes'])
    assert result.exit_code == 0


def test_get_github_url():
    runner = CliRunner()
    result = runner.invoke(install, ['mmcls', '--yes'])
    assert result.exit_code == 0
    assert get_github_url(
        'mmcls') == 'https://github.com/open-mmlab/mmclassification.git'

    result = runner.invoke(uninstall, ['mmcls', '--yes'])
    assert result.exit_code == 0
    assert get_github_url(
        'mmcls') == 'https://github.com/open-mmlab/mmclassification.git'


def teardown_module():
    runner = CliRunner()
    result = runner.invoke(uninstall, ['mmcv-full', '--yes'])
    assert result.exit_code == 0
    result = runner.invoke(uninstall, ['mmcls', '--yes'])
    assert result.exit_code == 0
