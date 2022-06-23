# Copyright (c) OpenMMLab. All rights reserved.
import sys
from typing import Any, List, Tuple, Union

import click

from mim.click import argument, get_installed_package, param2lowercase
from mim.utils import call_command


@click.command(
    'uninstall',
    context_settings=dict(ignore_unknown_options=True),
)
@argument(
    'args',
    autocompletion=get_installed_package,
    callback=param2lowercase,
    nargs=-1,
    type=click.UNPROCESSED)
@click.option(
    '-y',
    '--yes',
    'confirm_yes',
    is_flag=True,
    help='Don’t ask for confirmation of uninstall deletions.')
@click.option(
    '-r',
    '--requirement',
    'requirements',
    multiple=True,
    help='Uninstall all the packages listed in the given requirements '
    'file.  This option can be used multiple times.')
def cli(args: Tuple,
        confirm_yes: bool = False,
        requirements: Tuple = ()) -> None:
    """Uninstall package.

    Same as `pip uninstall`.

    \b
    Example:

        > mim uninstall mmcv-full
        > mim uninstall -y mmcv-full
        > mim uninstall mmdet mmcls

    Here we list several commonly used options.

    For more options, please refer to `pip uninstall --help`.
    """
    exit_code = uninstall(list(args), confirm_yes, requirements)
    exit(exit_code)


def uninstall(uninstall_args: Union[str, List],
              confirm_yes: bool = True,
              requirements: Tuple = ()) -> Any:
    """Uninstall package.

    Args:
        uninstall_args (str or list): A package name or a list of package names
            to uninstalled. You can also put some `pip uninstal` options here.
        confirm_yes (bool): Don’t ask for confirmation of uninstall deletions.
            Default: True.
        requirements (tuple): A tuple of requirements files to uninstalled.

    Returns:
        The status code return by `pip uninstall`.
    """
    if type(uninstall_args) is str:
        uninstall_args = [uninstall_args]

    if confirm_yes:
        uninstall_args.append('-y')  # type: ignore

    for requirement_file in requirements:
        uninstall_args += ['-r', requirement_file]  # type: ignore

    # Use the pip official recommend way to invoke `pip uninstall`:
    # https://pip.pypa.io/en/stable/user_guide/#using-pip-from-your-program
    pip_uninstall_cmd = [sys.executable, '-m', 'pip', 'uninstall']
    return call_command(pip_uninstall_cmd + uninstall_args)  # type: ignore
