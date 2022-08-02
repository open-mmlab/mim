# Copyright (c) OpenMMLab. All rights reserved.
from click.testing import CliRunner

from mim.cli import cli as mim_cli


def test_version():
    # test `mim --version` command
    runner = CliRunner()
    result = runner.invoke(mim_cli, ['--version'])
    assert result.exit_code == 0
