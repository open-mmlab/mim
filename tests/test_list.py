# Copyright (c) OpenMMLab. All rights reserved.
import sys

from click.testing import CliRunner

from mim.commands.install import cli as install
from mim.commands.list import list_package


def test_list(tmp_path):
    sys.path.append(str(tmp_path))
    runner = CliRunner()
    # mim install mmcls==0.23.0 --yes
    result = runner.invoke(
        install, ['mmcls==0.23.1', '--yes', '--target',
                  str(tmp_path)])
    assert result.exit_code == 0
    # mim list
    target = ('mmcls', '0.23.1',
              'https://github.com/open-mmlab/mmclassification')
    result = list_package()
    assert target in result
