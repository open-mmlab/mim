from click.testing import CliRunner

from mim.commands.install import cli as install
from mim.commands.uninstall import cli as uninstall
from mim.utils import read_installation_records


def setup_module():
    runner = CliRunner()
    result = runner.invoke(uninstall, ['mmcv-full', '--yes'])
    assert result.exit_code == 0
    result = runner.invoke(uninstall, ['mmcls', '--yes'])
    assert result.exit_code == 0


def test_read_installation_records():
    target = ('mmcls', '0.11.0',
              'https://github.com/open-mmlab/mmclassification.git')
    pkgs_info = read_installation_records()
    assert target not in pkgs_info

    runner = CliRunner()
    result = runner.invoke(install, ['mmcls==0.11.0', '--yes'])
    assert result.exit_code == 0
    target = ('mmcls', '0.11.0',
              'https://github.com/open-mmlab/mmclassification.git')
    pkgs_info = read_installation_records()
    assert target in pkgs_info

    result = runner.invoke(uninstall, ['mmcls', '--yes'])
    assert result.exit_code == 0


def test_write_installation_records():
    runner = CliRunner()
    result = runner.invoke(install, ['mmcls==0.11.0', '--yes'])
    assert result.exit_code == 0
    target = ('mmcls', '0.11.0',
              'https://github.com/open-mmlab/mmclassification.git')
    pkgs_info = read_installation_records()
    assert target in pkgs_info

    result = runner.invoke(install, ['mmcls==0.10.0', '--yes'])
    assert result.exit_code == 0
    new_target = ('mmcls', '0.10.0',
                  'https://github.com/open-mmlab/mmclassification.git')
    pkgs_info = read_installation_records()
    assert target not in pkgs_info and new_target in pkgs_info

    result = runner.invoke(uninstall, ['mmcls', '--yes'])
    assert result.exit_code == 0
