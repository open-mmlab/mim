import sys

import click

from mim.click import get_installed_package, param2lowercase
from mim.utils import PKG2MODULE, call_command, remove_installation_records


@click.command('uninstall')
@click.argument(
    'package', autocompletion=get_installed_package, callback=param2lowercase)
@click.option(
    '-y',
    '--yes',
    'confirm_yes',
    is_flag=True,
    help='Don’t ask for confirmation of uninstall deletions.')
def cli(package: str, confirm_yes: bool) -> None:
    """Uninstall package.

    Example:

        > mim uninstall mmcv-full
    """
    uninstall(package, confirm_yes)


def uninstall(package: str, confirm_yes=False) -> None:
    """Uninstall package.

    Args:
        package (str): The name of uninstalled package, such as mmcv-full.
        confirm_yes (bool): Don’t ask for confirmation of uninstall deletions.
            Default: True.
    """
    uninstall_cmd = ['python', '-m', 'pip', 'uninstall', package]
    if confirm_yes:
        uninstall_cmd.append('-y')

    call_command(uninstall_cmd)
    # if package is installed, importlib.import_module will import the
    # package and add it to sys.modules. However, if we uninstall the
    # package in the same process, is_installed will give a wrong result
    # because importlib.import_module will search package from sys.modules
    # first.
    sys.modules.pop(PKG2MODULE.get(package, package), None)

    remove_installation_records(package)
