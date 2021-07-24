import click

from mim.click import get_installed_package, param2lowercase
from mim.utils import call_command


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
