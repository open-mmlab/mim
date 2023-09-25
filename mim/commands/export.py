# Copyright (c) OpenMMLab. All rights reserved.
import sys
from typing import Optional

import click

from mim.click import CustomCommand
from mim.utils.mmpack.pack_cfg import export_from_cfg

PYTHON = sys.executable


@click.command(
    name='export',
    context_settings=dict(ignore_unknown_options=True),
    cls=CustomCommand)
@click.argument('config', type=str)
@click.option(
    '-p',
    '--export_root_dir',
    type=str,
    help='The root directory name of export packge')
@click.option(
    '-ft',
    '--fast_test',
    is_flag=True,
    help='The fast_test mode using few data for testing')
def cli(config: str,
        export_root_dir: Optional[str] = None,
        fast_test: bool = False) -> None:

    export_from_cfg(
        config, export_root_dir=export_root_dir, fast_test=fast_test)
