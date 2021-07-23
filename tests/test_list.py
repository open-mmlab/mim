from click.testing import CliRunner

from mim.commands.install import cli as install
from mim.commands.list import list_package


def test_list():
    runner = CliRunner()
    # mim install mmcls==0.11.0 --yes
    result = runner.invoke(install, ['mmcls==0.11.0', '--yes'])
    assert result.exit_code == 0
    # mim list
    target = ('mmcls', '0.11.0',
              'https://github.com/open-mmlab/mmclassification')
    result = list_package()
    print(result)
    assert target in result
