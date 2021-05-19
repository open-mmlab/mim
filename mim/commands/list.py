import pkg_resources
from typing import List

import click
from tabulate import tabulate

from mim.utils import read_installation_records


@click.command('list')
@click.option(
    '--all',
    is_flag=True,
    help='List packages of OpenMMLab projects or all the packages in the '
    'python environment.')
def cli(all: bool = True) -> None:
    """List packages.

    \b
    Example:
        > mim list
        > mim list --all
    """
    table_header = ['Package', 'Version', 'Source']
    table_data = list_package(all=all)
    click.echo(tabulate(table_data, headers=table_header, tablefmt='simple'))


def list_package(all: bool = False) -> List[List[str]]:
    """List packages.

    List packages of OpenMMLab projects or all the packages in the python
    environment.

    Args:
        all (bool): List all installed packages. If all is False, it just lists
            the packages installed by mim. Default: False.
    """
    if not all:
        pkgs_info = read_installation_records()
    else:
        pkgs_info = []
        for pkg in pkg_resources.working_set:
            pkgs_info.append([pkg.project_name, pkg.version])

    return pkgs_info
