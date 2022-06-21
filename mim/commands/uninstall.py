# Copyright (c) OpenMMLab. All rights reserved.
from typing import Any, Union

import click
from pip._internal.commands import create_command

from mim.click import argument, get_installed_package, param2lowercase


@click.command('uninstall')
@argument(
    'packages',
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
@click.option(
    '--root-user-action',
    'root_user_action',
    default='warn',
    type=click.Choice(['warn', 'ignore']),
    help='Action if pip is run as a root user. By default, a warning '
    'message is shown.',
)
def cli(packages: tuple,
        confirm_yes: bool = False,
        requirements: tuple = (),
        root_user_action: str = 'warn') -> None:
    """Uninstall package.

    Same as `pip uninstall`.

    Example:

        > mim uninstall mmcv-full
        > mim uninstall -y mmcv-full
        > mim uninstall mmdet mmcls
    """
    exit_code = uninstall(packages, confirm_yes, requirements,
                          root_user_action)
    exit(exit_code)


def uninstall(packages: Union[str, tuple],
              confirm_yes: bool = True,
              requirements: tuple = (),
              root_user_action: str = 'warn') -> Any:
    """Uninstall package.

    Args:
        packages (str or tuple): A package name or a tuple of package names to
            uninstalled.
        confirm_yes (bool): Don’t ask for confirmation of uninstall deletions.
            Default: True.
        requirements (tuple): A tuple of requirements files to uninstalled.
        root_user_action (str): The action if pip is run as a root user.
            The valid values are 'warn' and 'ignore'.

    Returns:
        The status code return by `pip uninstall`.
    """
    if type(packages) is str:
        uninstall_args = [packages]
    else:
        uninstall_args = list(packages)

    if confirm_yes:
        uninstall_args.append('-y')

    assert root_user_action in ('warn', 'ignore'), \
        f"Invalid root_user_action: {root_user_action}' (from 'warn', 'ignore')"  # noqa: E501
    uninstall_args += ['--root-user-action', root_user_action]

    for requirement_file in requirements:
        uninstall_args += ['-r', requirement_file]

    return create_command('uninstall').main(uninstall_args)  # type: ignore
