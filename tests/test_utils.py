from click.testing import CliRunner

from mim.commands.install import cli as install
from mim.commands.uninstall import cli as uninstall
from mim.utils import get_github_url, parse_home_page


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
